#!/usr/bin/env python3
"""
Proton Affinity (PA) Calculator using PySCF DFT.

PA is defined as the negative enthalpy change for:
    B(g) + H+(g) -> BH+(g)
    PA = -DeltaH = H(B) + H(H+) - H(BH+)

where H(H+) = 5/2 RT (translational enthalpy only).

Pipeline:
    1. SMILES -> enumerate ALL protonation sites (N, O, S)
    2. For each site: generate protonated SMILES
    3. SMILES -> 3D coordinates via RDKit
    4. Geometry optimization via PySCF + geomeTRIC
    5. Hessian -> vibrational frequencies -> ZPE + thermal corrections
    6. PA = H(neutral) + H(H+) - H(protonated)
    7. Report the MOST FAVORABLE site (highest PA = most stable protonation)
"""
import numpy as np
import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from pyscf.tools import molden as molden_tool

# GPU acceleration: try gpu4pyscf, fall back to CPU
try:
    from gpu4pyscf.dft import rks as gpu_rks
    USE_GPU = True
    print("  [GPU] gpu4pyscf detected - using GPU acceleration")
except ImportError:
    USE_GPU = False
    print("  [CPU] gpu4pyscf not available - using CPU")

# ============================================================
# Physical Constants
# ============================================================
HARTREE_TO_KJMOL = 2625.5002
HARTREE_TO_KCALMOL = 627.5095
CM1_TO_HARTREE = 4.55634e-6      # 1 cm^-1 in Hartree
KB_SI = 1.380649e-23              # J/K
HARTREE_TO_J = 4.3597447222e-18   # J/Hartree
R_KJMOL = 8.314462e-3            # kJ/(mol*K)
FREQ_CONV = 5140.487             # sqrt(eigenvalue) -> cm^-1

# ZPE scale factors for common methods (Scott & Radom 1996 + updates)
ZPE_SCALE_FACTORS = {
    "B3LYP/6-31G**": 0.9613,
    "B3LYP/6-31G(d,p)": 0.9613,
    "B3LYP/def2-SVP": 0.9700,
    "B3LYP/def2-TZVP": 0.9850,
    "B3LYP/def2-TZVPP": 0.9850,
    "B3LYP/6-311+G(2d,2p)": 0.9850,
    "wB97X-D/def2-TZVP": 0.9750,
}


# ============================================================
# Automatic Protonation Site Enumeration
# ============================================================
def enumerate_protonation_sites(smiles):
    """
    Find ALL possible protonation sites on a molecule.

    Identifies heteroatoms (N, O, S, P) that can accept a proton
    based on formal charge (must be 0). Uses deepcopy approach for
    reliable SMILES generation and deduplicates equivalent sites.

    Returns:
        list of dicts: [{atom_idx, atom_symbol, neighbor_info,
                         protonated_smiles}, ...]
    """
    import copy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    raw_sites = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()

        # Consider N, O, S, P with formal charge 0
        if symbol not in ['N', 'O', 'S', 'P']:
            continue
        if atom.GetFormalCharge() != 0:
            continue

        # Describe the bonding environment
        neighbors = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = str(bond.GetBondType()).split('.')[-1]
            neighbors.append(f"{other.GetSymbol()}({bt})")
        neighbor_str = ", ".join(neighbors) if neighbors else "lone"

        # Create protonated form via deepcopy + SetFormalCharge
        try:
            mol_copy = copy.deepcopy(mol)
            atom_copy = mol_copy.GetAtomWithIdx(idx)
            atom_copy.SetFormalCharge(1)
            Chem.SanitizeMol(mol_copy)
            prot_smi = Chem.MolToSmiles(mol_copy, canonical=True)

            # Verify charge increased by 1
            orig_charge = Chem.GetFormalCharge(mol)
            new_charge = Chem.GetFormalCharge(mol_copy)
            if new_charge != orig_charge + 1:
                continue

            raw_sites.append({
                "atom_idx": idx,
                "atom_symbol": symbol,
                "neighbor_info": neighbor_str,
                "protonated_smiles": prot_smi,
            })
        except Exception:
            pass

    # Deduplicate: some atoms produce the same canonical SMILES
    seen = set()
    unique_sites = []
    for site in raw_sites:
        smi = site["protonated_smiles"]
        if smi not in seen:
            unique_sites.append(site)
            seen.add(smi)

    return unique_sites


# ============================================================
# Molecular structure utilities
# ============================================================
def smiles_to_pyscf_atom(smiles):
    """Convert SMILES to PySCF atom string via RDKit 3D embedding."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    for method in [AllChem.ETKDGv3, AllChem.ETKDG]:
        params = method()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == 0:
            break
    else:
        raise RuntimeError(f"Cannot generate 3D coords for {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)

    conf = mol.GetConformer()
    lines = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
    return "\n".join(lines)


def get_formal_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.GetFormalCharge(mol)


def is_linear(mol):
    if mol.natm <= 2:
        return True
    coords = mol.atom_coords()
    v1 = coords[1] - coords[0]
    n1 = np.linalg.norm(v1)
    if n1 < 1e-10:
        return False
    v1 = v1 / n1
    for i in range(2, mol.natm):
        vi = coords[i] - coords[0]
        ni = np.linalg.norm(vi)
        if ni > 1e-6:
            cross = np.cross(v1, vi / ni)
            if np.linalg.norm(cross) > 0.01:
                return False
    return True


# ============================================================
# Frequency & Thermochemistry
# ============================================================
def compute_frequencies(pyscf_mol, hessian_matrix):
    """
    Compute vibrational frequencies from PySCF Hessian.
    Returns array of vibrational frequencies in cm^-1.
    Imaginary frequencies shown as negative values.
    """
    natm = pyscf_mol.natm
    n3 = 3 * natm

    h = hessian_matrix.transpose(0, 2, 1, 3).reshape(n3, n3)
    h = (h + h.T) / 2

    masses = pyscf_mol.atom_mass_list(isotope_avg=True)
    mass_vec = np.repeat(masses, 3)
    sqrt_m = np.sqrt(mass_vec)
    h_mw = h / np.outer(sqrt_m, sqrt_m)

    eigenvalues = np.linalg.eigvalsh(h_mw)
    eigenvalues.sort()

    freqs_cm = np.array([
        np.sqrt(abs(ev)) * FREQ_CONV * (1 if ev >= 0 else -1)
        for ev in eigenvalues
    ])

    n_tr = 5 if is_linear(pyscf_mol) else 6
    return freqs_cm[n_tr:]


def compute_thermochemistry(e_elec, freqs_cm, temperature=298.15,
                            scale_factor=0.9850):
    """
    Compute thermodynamic quantities using IGRRHO model.
    (Ideal Gas, Rigid Rotor, Harmonic Oscillator)
    """
    T = temperature
    scaled = freqs_cm * scale_factor
    real_freqs = scaled[scaled > 50.0]

    # ZPE
    zpe = 0.5 * CM1_TO_HARTREE * np.sum(np.abs(real_freqs))

    # Vibrational thermal energy
    u = np.abs(real_freqs) * CM1_TO_HARTREE * HARTREE_TO_J / (KB_SI * T)
    u = np.clip(u, 0, 500)
    e_vib_thermal = CM1_TO_HARTREE * np.sum(
        np.abs(real_freqs) / (np.exp(u) - 1)
    )

    # Translational: 3/2 kT
    e_trans = 1.5 * KB_SI * T / HARTREE_TO_J
    # Rotational: 3/2 kT (nonlinear)
    e_rot = 1.5 * KB_SI * T / HARTREE_TO_J
    # PV = kT
    pv = KB_SI * T / HARTREE_TO_J

    h_total = e_elec + zpe + e_vib_thermal + e_trans + e_rot + pv
    n_imag = int(np.sum(freqs_cm < -50))

    return {
        "E_elec": e_elec,
        "ZPE": zpe,
        "ZPE_kjmol": zpe * HARTREE_TO_KJMOL,
        "H_total": h_total,
        "n_imaginary": n_imag,
        "frequencies_cm": freqs_cm,
    }


# ============================================================
# DFT Calculation Driver
# ============================================================
def _make_mf(mol, xc):
    """Create DFT mean-field object, using GPU if available."""
    if USE_GPU:
        mf = gpu_rks.RKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.grids.level = 4
    return mf


def _extract_properties(mf, mol):
    """Extract extra molecular properties from converged SCF (zero cost)."""
    props = {}

    # HOMO / LUMO energies
    occ = mf.mo_occ
    mo_e = mf.mo_energy
    occupied = np.where(occ > 0)[0]
    virtual = np.where(occ == 0)[0]
    if len(occupied) > 0:
        props["HOMO_Ha"] = float(mo_e[occupied[-1]])
        props["HOMO_eV"] = float(mo_e[occupied[-1]] * 27.2114)
    if len(virtual) > 0:
        props["LUMO_Ha"] = float(mo_e[virtual[0]])
        props["LUMO_eV"] = float(mo_e[virtual[0]] * 27.2114)
    if len(occupied) > 0 and len(virtual) > 0:
        gap = mo_e[virtual[0]] - mo_e[occupied[-1]]
        props["HOMO_LUMO_gap_eV"] = float(gap * 27.2114)

    # Dipole moment
    try:
        dip = mf.dip_moment(verbose=0)
        props["dipole_x"] = float(dip[0])
        props["dipole_y"] = float(dip[1])
        props["dipole_z"] = float(dip[2])
        props["dipole_debye"] = float(np.linalg.norm(dip))
    except Exception:
        pass

    # Basis and electron info
    props["n_basis"] = int(mol.nao)
    props["n_electrons"] = int(mol.nelectron)
    props["n_atoms"] = int(mol.natm)
    props["nuclear_repulsion_Ha"] = float(mol.energy_nuc())

    # Optimized coordinates
    BOHR_TO_ANG = 0.529177249
    coords_bohr = mol.atom_coords()
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords_ang = coords_bohr * BOHR_TO_ANG
    props["opt_coords_symbols"] = symbols
    props["opt_coords_angstrom"] = coords_ang.tolist()

    return props


# ============================================================
# Standard DFT File Output Functions
# ============================================================
def write_xyz(mol, filepath, comment=""):
    """Write geometry in XYZ format."""
    BOHR_TO_ANG = 0.529177249
    coords = mol.atom_coords() * BOHR_TO_ANG
    with open(filepath, "w") as f:
        f.write(f"{mol.natm}\n")
        f.write(f"{comment}\n")
        for i in range(mol.natm):
            sym = mol.atom_symbol(i)
            x, y, z = coords[i]
            f.write(f"{sym:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")


def write_molden_file(mf, mol, filepath):
    """Write Molden file for MO visualization (GaussView, Avogadro, etc.)."""
    try:
        with open(filepath, "w") as f:
            molden_tool.header(mol, f)
            molden_tool.orbital_coeff(mol, f, mf.mo_coeff,
                                      ene=mf.mo_energy, occ=mf.mo_occ)
    except Exception as e:
        print(f"  WARNING: Could not write Molden file: {e}")


def write_frequency_file(freqs_cm, filepath, scale_factor=0.9850):
    """Write vibrational frequencies to a text file."""
    with open(filepath, "w") as f:
        f.write(f"{'Mode':>5s}  {'Unscaled (cm-1)':>16s}  "
                f"{'Scaled (cm-1)':>16s}  {'Note'}\n")
        f.write("-" * 65 + "\n")
        for i, freq in enumerate(freqs_cm):
            scaled = freq * scale_factor
            note = "IMAGINARY" if freq < -50 else ""
            f.write(f"{i+1:5d}  {freq:16.4f}  {scaled:16.4f}  {note}\n")
        f.write(f"\nScale factor: {scale_factor}\n")
        f.write(f"Number of modes: {len(freqs_cm)}\n")
        n_imag = int(np.sum(freqs_cm < -50))
        f.write(f"Imaginary frequencies: {n_imag}\n")


def write_hessian_file(hessian_matrix, mol, filepath):
    """Write Hessian matrix (force constants) in a readable format."""
    natm = mol.natm
    n3 = 3 * natm
    h_flat = hessian_matrix.transpose(0, 2, 1, 3).reshape(n3, n3)

    with open(filepath, "w") as f:
        f.write(f"Hessian matrix (Hartree/Bohr^2)\n")
        f.write(f"Atoms: {natm}, Dimensions: {n3}x{n3}\n")
        f.write(f"Atom labels: ")
        for i in range(natm):
            sym = mol.atom_symbol(i)
            for ax in "XYZ":
                f.write(f" {sym}{i+1}{ax}")
        f.write("\n\n")
        # Write in blocks of 5 columns (Gaussian style)
        for col_start in range(0, n3, 5):
            col_end = min(col_start + 5, n3)
            f.write("         " +
                    "".join(f"{c+1:14d}" for c in range(col_start, col_end))
                    + "\n")
            for row in range(col_start, n3):
                f.write(f"{row+1:5d}  ")
                for col in range(col_start, min(col_end, row + 1)):
                    f.write(f"{h_flat[row, col]:14.8f}")
                f.write("\n")
            f.write("\n")


def write_log_summary(result, filepath, smiles="", charge=0,
                      basis="", xc=""):
    """Write a human-readable summary log (similar to Gaussian .log)."""
    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  DFT Calculation Summary\n")
        f.write(f"  Generated by PySCF {__import__('pyscf').__version__}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  SMILES:       {smiles}\n")
        f.write(f"  Charge:       {charge}\n")
        f.write(f"  Method:       {xc}/{basis}\n")
        f.write(f"  Engine:       {'GPU (gpu4pyscf)' if USE_GPU else 'CPU'}\n")
        f.write(f"  N atoms:      {result.get('n_atoms', '?')}\n")
        f.write(f"  N electrons:  {result.get('n_electrons', '?')}\n")
        f.write(f"  N basis:      {result.get('n_basis', '?')}\n\n")

        f.write("-" * 70 + "\n")
        f.write("  ENERGIES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  E(elec)         = {result.get('E_elec', 0):20.10f} Ha\n")
        f.write(f"  Nuclear repul.  = "
                f"{result.get('nuclear_repulsion_Ha', 0):20.10f} Ha\n")
        zpe = result.get("ZPE_kjmol")
        if zpe is not None:
            f.write(f"  ZPE             = {zpe:20.4f} kJ/mol\n")
        h = result.get("H_total")
        if h is not None:
            f.write(f"  H(total)        = {h:20.10f} Ha\n")
            f.write(f"                  = "
                    f"{h * HARTREE_TO_KJMOL:20.4f} kJ/mol\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("  ORBITAL ENERGIES\n")
        f.write("-" * 70 + "\n")
        homo = result.get("HOMO_eV")
        lumo = result.get("LUMO_eV")
        if homo is not None:
            f.write(f"  HOMO            = {homo:12.4f} eV\n")
        if lumo is not None:
            f.write(f"  LUMO            = {lumo:12.4f} eV\n")
        gap = result.get("HOMO_LUMO_gap_eV")
        if gap is not None:
            f.write(f"  HOMO-LUMO gap   = {gap:12.4f} eV\n")
        f.write("\n")

        dip = result.get("dipole_debye")
        if dip is not None:
            f.write("-" * 70 + "\n")
            f.write("  DIPOLE MOMENT\n")
            f.write("-" * 70 + "\n")
            f.write(f"  |mu|  = {dip:10.4f} Debye\n")
            f.write(f"  mu_x  = {result.get('dipole_x', 0):10.4f}\n")
            f.write(f"  mu_y  = {result.get('dipole_y', 0):10.4f}\n")
            f.write(f"  mu_z  = {result.get('dipole_z', 0):10.4f}\n\n")

        n_imag = result.get("n_imaginary")
        if n_imag is not None:
            f.write("-" * 70 + "\n")
            f.write("  VIBRATIONAL ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Imaginary frequencies: {n_imag}\n")
            freqs = result.get("frequencies_cm", [])
            if len(freqs) > 0:
                f.write(f"  Total modes: {len(freqs)}\n")
                f.write(f"  Lowest:  {min(freqs):10.1f} cm-1\n")
                real = [v for v in freqs if v > 0]
                if real:
                    f.write(f"  Highest: {max(real):10.1f} cm-1\n")
            f.write("\n")

        wt = result.get("wall_time")
        if wt is not None:
            f.write(f"  Wall time: {wt:.1f} s "
                    f"({wt/60:.1f} min)\n")
        f.write("\n  Normal termination.\n")


def run_dft(smiles, charge=None, basis="def2-TZVP", xc="B3LYP",
            do_opt=True, do_freq=True, verbose=3, scale_factor=None,
            output_dir=None, label=None):
    """Full DFT workflow: SMILES -> opt -> freq -> thermochemistry.

    If output_dir is set, saves standard DFT files:
      - {label}_initial.xyz     Initial geometry
      - {label}_optimized.xyz   Optimized geometry
      - {label}.molden          MO coefficients (Molden format)
      - {label}_freq.txt        Vibrational frequencies
      - {label}_hessian.txt     Hessian matrix
      - {label}.log             Human-readable summary
    """
    if charge is None:
        charge = get_formal_charge(smiles)

    # Auto-select ZPE scale factor
    if scale_factor is None:
        key = f"{xc}/{basis}"
        scale_factor = ZPE_SCALE_FACTORS.get(key, 0.9800)

    print(f"\n{'='*60}")
    print(f"  DFT: {smiles}  charge={charge}  {xc}/{basis}")
    print(f"  Engine: {'GPU' if USE_GPU else 'CPU'}")
    print(f"{'='*60}")

    # Set up file output
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if label is None:
            # Sanitize SMILES for filename
            label = smiles.replace("/", "_").replace("\\", "_") \
                          .replace("(", "").replace(")", "") \
                          .replace("[", "").replace("]", "") \
                          .replace("+", "p").replace("-", "m") \
                          .replace("#", "t").replace("=", "d")[:60]

    t0 = time.time()

    atom_str = smiles_to_pyscf_atom(smiles)
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=0,
                verbose=verbose)
    print(f"  Atoms: {mol.natm}  Electrons: {mol.nelectron}  "
          f"Basis functions: {mol.nao}")

    # Save initial geometry
    if output_dir is not None:
        write_xyz(mol, os.path.join(output_dir, f"{label}_initial.xyz"),
                  comment=f"Initial geometry: {smiles} {xc}/{basis}")

    mf = _make_mf(mol, xc)
    mf.kernel()
    print(f"  Initial E = {mf.e_tot:.10f} Ha")

    if do_opt:
        print("  Optimizing geometry...")
        from pyscf.geomopt.geometric_solver import optimize
        try:
            mol_eq = optimize(mf, maxsteps=200)
        except Exception as e:
            print(f"  WARNING: geomeTRIC failed ({e}), using initial geometry")
            mol_eq = mol

        mf_opt = _make_mf(mol_eq, xc)
        mf_opt.kernel()
        e_elec = mf_opt.e_tot
        active_mol = mol_eq
        active_mf = mf_opt
    else:
        e_elec = mf.e_tot
        active_mol = mol
        active_mf = mf

    print(f"  Final E = {e_elec:.10f} Ha")
    result = {"smiles": smiles, "charge": charge, "E_elec": e_elec}

    # Extract extra properties (HOMO, LUMO, dipole, coords) - free
    props = _extract_properties(active_mf, active_mol)
    result.update(props)

    # Save optimized geometry + Molden file
    if output_dir is not None:
        write_xyz(active_mol,
                  os.path.join(output_dir, f"{label}_optimized.xyz"),
                  comment=f"Optimized: {smiles} E={e_elec:.10f} Ha "
                          f"{xc}/{basis}")
        write_molden_file(active_mf, active_mol,
                          os.path.join(output_dir, f"{label}.molden"))

    if do_freq:
        print("  Computing Hessian (analytical)...")
        # Hessian: use CPU fallback if GPU hessian unavailable
        try:
            hess = active_mf.Hessian().kernel()
        except Exception:
            print("  GPU Hessian unavailable, falling back to CPU...")
            mf_cpu = dft.RKS(active_mol)
            mf_cpu.xc = xc
            mf_cpu.conv_tol = 1e-10
            mf_cpu.grids.level = 4
            mf_cpu.kernel()
            hess = mf_cpu.Hessian().kernel()

        freqs = compute_frequencies(active_mol, hess)

        # Save Hessian and frequency files
        if output_dir is not None:
            write_hessian_file(hess, active_mol,
                               os.path.join(output_dir,
                                            f"{label}_hessian.txt"))
            write_frequency_file(freqs,
                                 os.path.join(output_dir,
                                              f"{label}_freq.txt"),
                                 scale_factor=scale_factor)

        n_imag = int(np.sum(freqs < -50))
        print(f"  Vibrational frequencies (cm^-1):")
        for i, f in enumerate(freqs):
            tag = " ***IMAGINARY***" if f < -50 else ""
            print(f"    {i+1:3d}: {f:10.1f}{tag}")
        if n_imag > 0:
            print(f"  WARNING: {n_imag} imaginary frequency(ies)!")

        thermo = compute_thermochemistry(e_elec, freqs,
                                         scale_factor=scale_factor)
        result.update(thermo)
        # Save all frequencies as list (not numpy array) for JSON
        result["frequencies_cm"] = freqs.tolist()
        print(f"  ZPE = {thermo['ZPE_kjmol']:.2f} kJ/mol")
        print(f"  H_total = {thermo['H_total']:.10f} Ha")
    else:
        result["H_total"] = e_elec

    result["wall_time"] = time.time() - t0
    print(f"  Wall time: {result['wall_time']:.1f} s")

    # Write summary log
    if output_dir is not None:
        write_log_summary(result, os.path.join(output_dir, f"{label}.log"),
                          smiles=smiles, charge=charge,
                          basis=basis, xc=xc)
        print(f"  Output files saved to: {output_dir}/{label}_*")

    return result


# ============================================================
# Proton Affinity: Multi-Site Calculation
# ============================================================
def calculate_pa(smiles_neutral, basis="def2-TZVP", xc="B3LYP",
                 do_freq=True, protonated_smiles_override=None,
                 output_dir=None):
    """
    Calculate proton affinity by trying ALL protonation sites.

    For each heteroatom (N, O, S) that can accept a proton:
      1. Generate the protonated SMILES
      2. Optimize geometry + compute frequencies
      3. Calculate PA for that site

    Returns the MOST FAVORABLE site (highest PA = most stable protonation).

    Parameters:
        smiles_neutral: SMILES of base B
        basis, xc: level of theory
        do_freq: include ZPE + thermal corrections
        protonated_smiles_override: if given, skip auto-enumeration and
                                    use this specific protonated SMILES
        output_dir: if set, saves standard DFT files in subdirectories
                    (neutral/, site_1/, site_2/, ...)

    Returns:
        best_pa_kjmol, result_neutral, best_result_prot, all_sites_results
    """
    print(f"\n{'#'*60}")
    print(f"  PROTON AFFINITY CALCULATION")
    print(f"  Neutral:  {smiles_neutral}")
    print(f"  Level:    {xc}/{basis}")
    print(f"{'#'*60}")

    # --- Step 1: Compute neutral molecule ---
    neutral_outdir = (os.path.join(output_dir, "neutral")
                      if output_dir else None)
    result_neutral = run_dft(smiles_neutral, basis=basis, xc=xc,
                             do_freq=do_freq,
                             output_dir=neutral_outdir,
                             label="neutral")

    # --- Step 2: Enumerate protonation sites ---
    if protonated_smiles_override:
        sites = [{"atom_symbol": "?", "neighbor_info": "manual",
                  "protonated_smiles": protonated_smiles_override}]
    else:
        sites = enumerate_protonation_sites(smiles_neutral)

    if not sites:
        print("  ERROR: No protonation sites found!")
        return None, result_neutral, None, []

    print(f"\n  Found {len(sites)} protonation site(s):")
    for i, s in enumerate(sites):
        print(f"    Site {i+1}: {s['atom_symbol']} "
              f"({s['neighbor_info']}) -> {s['protonated_smiles']}")

    # --- Step 3: Calculate PA for each site ---
    T = 298.15
    h_proton_kjmol = 2.5 * R_KJMOL * T  # = 6.197 kJ/mol

    all_site_results = []
    best_pa = -1e10
    best_result = None
    best_site_info = None

    for i, site in enumerate(sites):
        print(f"\n  --- Site {i+1}/{len(sites)}: "
              f"{site['atom_symbol']} ({site['neighbor_info']}) ---")
        print(f"  Protonated SMILES: {site['protonated_smiles']}")

        try:
            site_outdir = (os.path.join(output_dir, f"site_{i+1}")
                           if output_dir else None)
            result_prot = run_dft(site["protonated_smiles"],
                                  basis=basis, xc=xc, do_freq=do_freq,
                                  output_dir=site_outdir,
                                  label=f"protonated_site{i+1}")

            dH = result_neutral["H_total"] - result_prot["H_total"]
            pa = dH * HARTREE_TO_KJMOL + h_proton_kjmol

            print(f"  -> PA at this site: {pa:.1f} kJ/mol")

            site_entry = {
                "site": site,
                "pa_kjmol": pa,
                "result": result_prot,
                "status": "OK",
            }
            all_site_results.append(site_entry)

            if pa > best_pa:
                best_pa = pa
                best_result = result_prot
                best_site_info = site

        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_site_results.append({
                "site": site, "pa_kjmol": None,
                "status": f"FAILED: {e}",
            })

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  PROTON AFFINITY RESULTS for {smiles_neutral}")
    print(f"  Level: {xc}/{basis}")
    print(f"{'='*60}")
    print(f"  H(neutral) = {result_neutral['H_total']:.10f} Ha")
    print(f"  H(H+)      = {h_proton_kjmol:.3f} kJ/mol")
    print()

    if len(all_site_results) > 1:
        print(f"  {'Site':<30s}  {'PA (kJ/mol)':>12s}  {'Status'}")
        print(f"  {'-'*55}")
        for sr in all_site_results:
            s = sr["site"]
            label = f"{s['atom_symbol']}({s['neighbor_info']})"
            if sr["status"] == "OK":
                marker = " <-- BEST" if sr["pa_kjmol"] == best_pa else ""
                print(f"  {label:<30s}  {sr['pa_kjmol']:12.1f}  "
                      f"OK{marker}")
            else:
                print(f"  {label:<30s}  {'---':>12s}  {sr['status']}")

    if best_site_info:
        print(f"\n  BEST SITE: {best_site_info['atom_symbol']} "
              f"({best_site_info['neighbor_info']})")
        print(f"  BEST PA = {best_pa:.1f} kJ/mol  "
              f"({best_pa/4.184:.1f} kcal/mol)")
    print(f"{'='*60}")

    return best_pa, result_neutral, best_result, all_site_results
