# mopac_calculator.py - UPDATED WITH CHARGE SUPPORT
import tempfile
import uuid
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pandas as pd
import re

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class MOPACCalculator:
    """
    MOPAC calculator for local use with charge support and 
    comprehensive property parsing.
    """

    def __init__(self, method="PM7"):
        self.method = method
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        self.keywords = "PRECISE GNORM=0.001 SCFCRT=1.D-8"
        self.proton_hof = 365.7  # kcal/mol (gas phase standard)
        self._check_mopac()

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up the temporary directory."""
        self.temp_dir_obj.cleanup()

    def _check_mopac(self):
        """Check if MOPAC is in the system PATH."""
        try:
            subprocess.run(["mopac"], capture_output=True, text=True, timeout=10)
            print("✅ MOPAC executable found in PATH.")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ MOPAC not found in PATH.")
            return False

    def smiles_to_3d(self, smiles, charge='auto'):
        """
        Convert SMILES to 3D coordinates with charge detection.
        
        Args:
            smiles: SMILES string
            charge: 'auto' to detect from SMILES, or specify int (0, +1, -1, etc.)
        
        Returns:
            tuple: (atoms, coordinates, charge) or (None, None, None)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                return None, None, None
            
            # Detect charge if auto
            if charge == 'auto':
                charge = Chem.GetFormalCharge(mol)
            
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None, None, None
            
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            except:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                except:
                    pass  # Use unoptimized
            
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) 
                             for i in range(mol.GetNumAtoms())])
            
            return atoms, coords, charge
            
        except Exception as e:
            print(f"❌ Failed to generate 3D structure for {smiles}: {e}")
            return None, None, None

    def write_mopac_input(self, atoms, coordinates, label, charge=0, multiplicity=None):
        """
        Write MOPAC input file with charge and multiplicity support.
        
        Args:
            atoms: List of atom symbols
            coordinates: Array of coordinates
            label: Job label
            charge: Molecular charge (default 0)
            multiplicity: Spin multiplicity (default None)
        """
        input_file = os.path.join(self.temp_dir, f"{label}.mop")
        
        # Build keyword line
        keyword_line = f"{self.method} {self.keywords}"
        
        # Add charge
        if charge != 0:
            keyword_line += f" CHARGE={charge}"
        
        # Add UHF for charged species or radicals
        if multiplicity is not None and multiplicity > 1:
            keyword_line += " UHF"
        elif charge != 0:
            keyword_line += " UHF"  # Use UHF for all charged species
        
        with open(input_file, 'w') as f:
            f.write(f"{keyword_line}\n")
            f.write(f"Calculation for {label} (charge={charge:+d})\n\n")
            for atom, coord in zip(atoms, coordinates):
                f.write(f"{atom:2s} {coord[0]:12.6f} 1 {coord[1]:12.6f} 1 {coord[2]:12.6f} 1\n")
        
        return input_file

    def run_mopac_calculation(self, input_file):
        """Run MOPAC calculation."""
        try:
            result = subprocess.run(
                ["mopac", input_file], 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            if result.returncode != 0:
                print(f"MOPAC failed: {result.stderr[:200]}")
                return False
            return True
        except Exception as e:
            print(f"Error running MOPAC: {e}")
            return False

    def parse_mopac_output(self, output_file):
        """Parse MOPAC output file for all properties."""
        properties = {}
        try:
            with open(output_file, 'r') as f:
                content = f.read()

            # 1. Heat of formation (CRITICAL for PA!)
            hof_match = re.search(
                r"FINAL\s+HEAT\s+OF\s+FORMATION\s*=\s*([-+]?\d+\.\d+)", 
                content, re.IGNORECASE
            )
            if hof_match: 
                properties['heat_of_formation'] = float(hof_match.group(1))

            # 2. Dipole moment
            dipole_match = re.search(
                r"SUM\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", 
                content
            )
            if dipole_match:
                properties['dipole_x'] = float(dipole_match.group(1))
                properties['dipole_y'] = float(dipole_match.group(2))
                properties['dipole_z'] = float(dipole_match.group(3))
                properties['dipole_moment'] = float(dipole_match.group(4))

            # 3. HOMO/LUMO energies (closed shell)
            homo_lumo_match = re.search(
                r"HOMO\s+LUMO\s+ENERGIES\s*\(EV\)\s*=\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", 
                content
            )
            if homo_lumo_match:
                properties['homo_ev'] = float(homo_lumo_match.group(1))
                properties['lumo_ev'] = float(homo_lumo_match.group(2))
                properties['gap_ev'] = properties['lumo_ev'] - properties['homo_ev']
                properties['spin_state'] = 'closed_shell'
            else:
                # Try SOMO/LUMO (open shell)
                alpha_match = re.search(
                    r"ALPHA\s+SOMO\s+LUMO\s*\(EV\)\s*=\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", 
                    content
                )
                beta_match = re.search(
                    r"BETA\s+SOMO\s+LUMO\s*\(EV\)\s*=\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", 
                    content
                )
                if alpha_match and beta_match:
                    properties['alpha_somo_ev'] = float(alpha_match.group(1))
                    properties['alpha_lumo_ev'] = float(alpha_match.group(2))
                    properties['beta_somo_ev'] = float(beta_match.group(1))
                    properties['beta_lumo_ev'] = float(beta_match.group(2))
                    properties['homo_ev'] = max(properties['alpha_somo_ev'], 
                                               properties['beta_somo_ev'])
                    properties['lumo_ev'] = min(properties['alpha_lumo_ev'], 
                                               properties['beta_lumo_ev'])
                    properties['gap_ev'] = properties['lumo_ev'] - properties['homo_ev']
                    properties['spin_state'] = 'open_shell'

            # 4. Ionization potential
            ip_match = re.search(
                r"IONIZATION\s+POTENTIAL\s*=\s*([-+]?\d+\.\d+)", 
                content, re.IGNORECASE
            )
            if ip_match: 
                properties['ionization_potential'] = float(ip_match.group(1))

            # 5. COSMO properties
            cosmo_area_match = re.search(
                r"COSMO\s+AREA\s*=\s*([-+]?\d+\.\d+)", 
                content, re.IGNORECASE
            )
            if cosmo_area_match: 
                properties['cosmo_area'] = float(cosmo_area_match.group(1))
            
            cosmo_volume_match = re.search(
                r"COSMO\s+VOLUME\s*=\s*([-+]?\d+\.\d+)", 
                content, re.IGNORECASE
            )
            if cosmo_volume_match: 
                properties['cosmo_volume'] = float(cosmo_volume_match.group(1))

            # 6. Molecular weight
            mw_match = re.search(
                r"MOLECULAR\s+WEIGHT\s*=\s*([-+]?\d+\.\d+)", 
                content, re.IGNORECASE
            )
            if mw_match: 
                properties['molecular_weight'] = float(mw_match.group(1))

            # 7. Point group
            pg_match = re.search(
                r"POINT\s+GROUP:\s*([A-Za-z0-9]+)", 
                content, re.IGNORECASE
            )
            if pg_match: 
                properties['point_group'] = pg_match.group(1)

            # 8. Filled levels
            filled_match = re.search(
                r"NO\.\s+OF\s+FILLED\s+LEVELS\s*=\s*(\d+)", 
                content, re.IGNORECASE
            )
            if filled_match: 
                properties['filled_levels'] = int(filled_match.group(1))

            # 9. Molecular charge
            charge_match = re.search(
                r"CHARGE\s+ON\s+SYSTEM\s*=\s*([-+]?\d+)", 
                content, re.IGNORECASE
            )
            if charge_match: 
                properties['charge'] = int(charge_match.group(1))

            # 10. Electron counts (for open shell)
            alpha_match = re.search(
                r"NO\.\s+OF\s+ALPHA\s+ELECTRONS\s*=\s*(\d+)", 
                content
            )
            beta_match = re.search(
                r"NO\.\s+OF\s+BETA\s+ELECTRONS\s*=\s*(\d+)", 
                content
            )
            if alpha_match and beta_match:
                alpha_electrons = int(alpha_match.group(1))
                beta_electrons = int(beta_match.group(1))
                properties['alpha_electrons'] = alpha_electrons
                properties['beta_electrons'] = beta_electrons
                properties['unpaired_electrons'] = abs(alpha_electrons - beta_electrons)
                properties['multiplicity'] = properties['unpaired_electrons'] + 1

            # 11. Total energy
            if 'heat_of_formation' in properties:
                properties['total_energy_kcal_mol'] = properties['heat_of_formation']
                properties['total_energy_ev'] = properties['heat_of_formation'] * 0.043363

            # 12. Computation time
            comp_time_match = re.search(
                r"COMPUTATION\s+TIME\s*=\s*([\d.]+)", 
                content, re.IGNORECASE
            )
            if comp_time_match: 
                properties['computation_time'] = float(comp_time_match.group(1))

        except Exception as e:
            print(f"❌ Error parsing MOPAC output: {e}")
        
        return properties

    def cleanup_files(self, label):
        """Clean up temporary files."""
        extensions = ['.mop', '.out', '.arc', '.aux', '.log', '.end']
        for ext in extensions:
            try:
                os.remove(os.path.join(self.temp_dir, f"{label}{ext}"))
            except OSError:
                pass

    def calculate_properties(self, smiles, charge='auto', cleanup=True):
        """
        Calculate MOPAC properties for a single SMILES string with charge support.
        
        Args:
            smiles: SMILES string
            charge: 'auto' to detect, or int (0, +1, -1, etc.)
            cleanup: Whether to clean up temporary files
        
        Returns:
            dict: Properties including success, error, charge, etc.
        """
        label = f"mol_{uuid.uuid4().hex[:8]}"

        atoms, coords, detected_charge = self.smiles_to_3d(smiles, charge)
        if atoms is None:
            return {
                'success': False, 
                'error': 'Failed to generate 3D structure', 
                'smiles': smiles
            }

        input_file = self.write_mopac_input(atoms, coords, label, detected_charge)
        
        if not self.run_mopac_calculation(input_file):
            if cleanup: 
                self.cleanup_files(label)
            return {
                'success': False, 
                'error': 'MOPAC calculation failed', 
                'smiles': smiles,
                'input_charge': detected_charge
            }

        output_file = os.path.join(self.temp_dir, f"{label}.out")
        properties = self.parse_mopac_output(output_file)
        
        if not properties or 'heat_of_formation' not in properties:
            if cleanup: 
                self.cleanup_files(label)
            return {
                'success': False, 
                'error': 'Failed to parse properties', 
                'smiles': smiles,
                'input_charge': detected_charge
            }

        properties.update({
            'success': True, 
            'smiles': smiles, 
            'num_atoms': len(atoms),
            'input_charge': detected_charge
        })
        
        if cleanup:
            self.cleanup_files(label)
        
        return properties

    def calculate_proton_affinity(self, smiles_neutral, smiles_protonated, cleanup=True):
        """
        Calculate gas-phase proton affinity.
        
        PA = HOF(neutral) + HOF(H+) - HOF(protonated)
        
        Args:
            smiles_neutral: SMILES of neutral molecule
            smiles_protonated: SMILES of protonated form
            cleanup: Whether to clean up files
        
        Returns:
            dict: Results including PA and component energies
        """
        results = {
            'success': False,
            'smiles_neutral': smiles_neutral,
            'smiles_protonated': smiles_protonated
        }
        
        # Calculate neutral
        props_neutral = self.calculate_properties(smiles_neutral, charge=0, cleanup=cleanup)
        if not props_neutral['success']:
            results['error'] = f"Neutral calc failed: {props_neutral.get('error')}"
            return results
        
        hof_neutral = props_neutral.get('heat_of_formation')
        if hof_neutral is None:
            results['error'] = "HOF not found for neutral"
            return results
        
        # Calculate protonated
        props_protonated = self.calculate_properties(smiles_protonated, charge=1, cleanup=cleanup)
        if not props_protonated['success']:
            results['error'] = f"Protonated calc failed: {props_protonated.get('error')}"
            return results
        
        hof_protonated = props_protonated.get('heat_of_formation')
        if hof_protonated is None:
            results['error'] = "HOF not found for protonated"
            return results
        
        # Calculate PA
        pa = hof_neutral + self.proton_hof - hof_protonated
        
        results.update({
            'success': True,
            'hof_neutral': hof_neutral,
            'hof_protonated': hof_protonated,
            'hof_proton': self.proton_hof,
            'proton_affinity_kcal_mol': pa,
            'proton_affinity_kj_mol': pa * 4.184,
            'proton_affinity_ev': pa * 0.043363,
            'properties_neutral': props_neutral,
            'properties_protonated': props_protonated
        })
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_pm7_properties(smiles, charge='auto', method="PM7", cleanup=True):
    """One-line function to calculate properties with charge support."""
    with MOPACCalculator(method=method) as calculator:
        return calculator.calculate_properties(smiles, charge=charge, cleanup=cleanup)

def calculate_pm7_proton_affinity(smiles_neutral, smiles_protonated, method="PM7", cleanup=True):
    """One-line function to calculate proton affinity."""
    with MOPACCalculator(method=method) as calculator:
        return calculator.calculate_proton_affinity(smiles_neutral, smiles_protonated, cleanup=cleanup)

def calculate_pm7_dataframe(df, smiles_column='smiles', charge_column=None, method="PM7", cleanup=True):
    """Calculate properties for a pandas DataFrame with charge support."""
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found")
    
    results = []
    with MOPACCalculator(method=method) as calculator:
        for i, row in df.iterrows():
            smiles = row[smiles_column]
            charge = row[charge_column] if charge_column and charge_column in df.columns else 'auto'
            print(f"\n--- Molecule {i+1}/{len(df)} ---")
            results.append(calculator.calculate_properties(smiles, charge=charge, cleanup=cleanup))
    
    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)