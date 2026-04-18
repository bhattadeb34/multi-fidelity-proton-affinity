# run_pm7_parallel.py - UPDATED WITH CHARGE SUPPORT AND PROTON AFFINITY
# ====================================================================
# PARALLEL PM7 CALCULATOR WITH PROTON AFFINITY CALCULATIONS
# Optimized for HPC job arrays
# ====================================================================

import os
import json
import time
import copy
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import rdkit.Chem as Chem
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import from updated mopac_calculator
from mopac_calculator import calculate_pm7_properties, calculate_pm7_proton_affinity


def check_system_resources():
    """Check available system resources."""
    cpu_count = os.cpu_count()
    return {'cpu_count': cpu_count}


class ParallelPM7Calculator:
    """PM7 calculator with charge support and proton affinity calculations."""
    
    def __init__(self, project_name: str, n_processes: int):
        self.project_name = project_name
        self.n_processes = n_processes
        self.output_dir = Path.cwd() / self.project_name
        self.output_dir.mkdir(exist_ok=True)
        
        # Load existing data
        self.protonation_map = self._load_json("protonation_map.json", {})
        self.properties_neutral = self._load_json("properties_neutral.json", {})
        self.properties_protonated = self._load_json("properties_protonated.json", {})
        self.proton_affinities = self._load_json("proton_affinities.json", {})
        self.progress = self._load_json("progress.json", {
            "completed_smiles": [], 
            "total_processed": 0, 
            "successful_neutral": 0,
            "successful_protonated": 0,
            "successful_pa": 0,
            "failed": 0
        })
        self.failed_molecules = self._load_json("failed_molecules.json", [])
        
        print(f"✅ Initialized calculator with {self.n_processes} processes")
        print(f"📁 Results directory: {self.output_dir}")

    def _load_json(self, filename: str, default):
        """Load JSON file."""
        filepath = self.output_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f: 
                    return json.load(f)
            except: 
                pass
        return default

    def _save_json(self, data, filename: str):
        """Save JSON file atomically."""
        filepath = self.output_dir / filename
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w') as f: 
            json.dump(data, f, indent=2)
        temp_path.rename(filepath)

    def _save_checkpoint(self):
        """Save all checkpoint files."""
        self._save_json(self.protonation_map, "protonation_map.json")
        self._save_json(self.properties_neutral, "properties_neutral.json")
        self._save_json(self.properties_protonated, "properties_protonated.json")
        self._save_json(self.proton_affinities, "proton_affinities.json")
        self._save_json(self.progress, "progress.json")
        self._save_json(self.failed_molecules, "failed_molecules.json")

    def generate_protonated_forms(self, smiles: str) -> List[Dict]:
        """
        Generate protonated forms with charge information.
        
        Returns:
            List of dicts with 'smiles', 'site_index', 'site_element'
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: 
                return []
            
            protonated_forms = []
            
            # Protonate N, O, S, P atoms
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                
                # Only protonate if neutral and is N/O/S/P
                if atom.GetSymbol() in ['N', 'O', 'S', 'P'] and atom.GetFormalCharge() == 0:
                    try:
                        mol_copy = copy.deepcopy(mol)
                        atom_copy = mol_copy.GetAtomWithIdx(atom_idx)
                        atom_copy.SetFormalCharge(1)  # Add +1 charge
                        Chem.SanitizeMol(mol_copy)
                        
                        prot_smiles = Chem.MolToSmiles(Chem.AddHs(mol_copy), canonical=True)
                        
                        # Avoid duplicates
                        if not any(p['smiles'] == prot_smiles for p in protonated_forms):
                            protonated_forms.append({
                                'smiles': prot_smiles,
                                'site_index': atom_idx,
                                'site_element': atom.GetSymbol()
                            })
                    except:
                        continue
            
            return protonated_forms
            
        except: 
            return []

    def process_molecules_parallel(self, smiles_list: List[str]):
        """
        Process molecules with charge-aware calculations and PA computation.
        """
        # Filter already processed
        remaining = sorted(list(set(
            s for s in smiles_list 
            if s not in self.progress['completed_smiles']
        )))
        
        if not remaining:
            print("✅ All molecules already processed!")
            return

        print(f"\n{'='*70}")
        print(f"PROCESSING {len(remaining)} NEW MOLECULES")
        print(f"{'='*70}")
        
        # Step 1: Generate protonation map
        print("\n📋 Step 1: Generating protonated forms...")
        for neutral_smiles in tqdm(remaining, desc="Generating"):
            if neutral_smiles not in self.protonation_map:
                self.protonation_map[neutral_smiles] = self.generate_protonated_forms(neutral_smiles)
        
        # Step 2: Build task list
        print("\n📋 Step 2: Building calculation task list...")
        tasks = []
        
        for neutral_smiles in remaining:
            # Task for neutral molecule
            if neutral_smiles not in self.properties_neutral:
                tasks.append({
                    'smiles': neutral_smiles,
                    'charge': 0,
                    'type': 'neutral',
                    'parent_neutral': neutral_smiles
                })
            
            # Tasks for protonated forms
            prot_forms = self.protonation_map.get(neutral_smiles, [])
            for prot_info in prot_forms:
                prot_smiles = prot_info['smiles']
                if prot_smiles not in self.properties_protonated:
                    tasks.append({
                        'smiles': prot_smiles,
                        'charge': 1,
                        'type': 'protonated',
                        'parent_neutral': neutral_smiles,
                        'site_index': prot_info['site_index'],
                        'site_element': prot_info['site_element']
                    })
        
        if not tasks:
            print("✅ All required calculations already complete!")
            return
        
        print(f"📊 Total tasks: {len(tasks)} ({sum(1 for t in tasks if t['type']=='neutral')} neutral, {sum(1 for t in tasks if t['type']=='protonated')} protonated)")
        
        # Step 3: Run calculations in parallel
        print(f"\n⚡ Step 3: Running PM7 calculations ({self.n_processes} workers)...")
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            future_to_task = {
                executor.submit(calculate_molecule_worker, task): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="PM7 Calculations") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        
                        if result.get('success', False):
                            # Store result based on type
                            if task['type'] == 'neutral':
                                self.properties_neutral[task['smiles']] = result
                                self.progress['successful_neutral'] += 1
                            elif task['type'] == 'protonated':
                                # Add site info to result
                                result['site_index'] = task['site_index']
                                result['site_element'] = task['site_element']
                                result['parent_neutral'] = task['parent_neutral']
                                self.properties_protonated[task['smiles']] = result
                                self.progress['successful_protonated'] += 1
                        else:
                            self.failed_molecules.append({
                                'smiles': task['smiles'],
                                'type': task['type'],
                                'error': result.get('error', 'Unknown'),
                                'parent_neutral': task.get('parent_neutral')
                            })
                            self.progress['failed'] += 1
                            
                    except Exception as e:
                        self.failed_molecules.append({
                            'smiles': task['smiles'],
                            'type': task['type'],
                            'error': f'Worker crashed: {e}',
                            'parent_neutral': task.get('parent_neutral')
                        })
                        self.progress['failed'] += 1
                    
                    pbar.update(1)
                    
                    # Save checkpoint every 100 calculations
                    if pbar.n % 100 == 0:
                        self._save_checkpoint()
        
        # Step 4: Calculate proton affinities
        print(f"\n🧪 Step 4: Calculating proton affinities...")
        pa_calculated = 0
        
        for neutral_smiles in tqdm(remaining, desc="PA Calculation"):
            # Check if neutral calculation succeeded
            if neutral_smiles not in self.properties_neutral:
                continue
            
            neutral_props = self.properties_neutral[neutral_smiles]
            hof_neutral = neutral_props.get('heat_of_formation')
            
            if hof_neutral is None:
                continue
            
            # Calculate PA for each protonation site
            prot_forms = self.protonation_map.get(neutral_smiles, [])
            pa_results = []
            
            for prot_info in prot_forms:
                prot_smiles = prot_info['smiles']
                
                if prot_smiles not in self.properties_protonated:
                    continue
                
                prot_props = self.properties_protonated[prot_smiles]
                hof_protonated = prot_props.get('heat_of_formation')
                
                if hof_protonated is None:
                    continue
                
                # Calculate PA: HOF(neutral) + HOF(H+) - HOF(protonated)
                pa_kcal_mol = hof_neutral + 365.7 - hof_protonated
                
                pa_results.append({
                    'protonated_smiles': prot_smiles,
                    'site_index': prot_info['site_index'],
                    'site_element': prot_info['site_element'],
                    'hof_neutral': hof_neutral,
                    'hof_protonated': hof_protonated,
                    'pa_kcal_mol': pa_kcal_mol,
                    'pa_kj_mol': pa_kcal_mol * 4.184,
                    'pa_ev': pa_kcal_mol * 0.043363
                })
                pa_calculated += 1
            
            if pa_results:
                self.proton_affinities[neutral_smiles] = {
                    'neutral_smiles': neutral_smiles,
                    'protonation_sites': pa_results,
                    'max_pa': max(r['pa_kcal_mol'] for r in pa_results),
                    'min_pa': min(r['pa_kcal_mol'] for r in pa_results),
                    'num_sites': len(pa_results)
                }
        
        self.progress['successful_pa'] = pa_calculated
        
        # Update completed list
        for smiles in remaining:
            if smiles not in self.progress['completed_smiles']:
                self.progress['completed_smiles'].append(smiles)
        
        self.progress['total_processed'] = len(self.progress['completed_smiles'])
        
        # Final checkpoint
        self._save_checkpoint()
        
        print(f"\n{'='*70}")
        print(f"CHUNK COMPLETE!")
        print(f"{'='*70}")
        print(f"✅ Neutral calculations:     {self.progress['successful_neutral']}")
        print(f"✅ Protonated calculations:  {self.progress['successful_protonated']}")
        print(f"✅ Proton affinities:        {self.progress['successful_pa']}")
        print(f"❌ Failed calculations:      {self.progress['failed']}")
        print(f"📊 Total molecules processed: {self.progress['total_processed']}")
        print(f"{'='*70}\n")


def calculate_molecule_worker(task: Dict) -> Dict:
    """
    Worker function for parallel PM7 calculations with charge support.
    
    Args:
        task: Dict with 'smiles', 'charge', 'type', etc.
    
    Returns:
        Dict with results
    """
    smiles = task['smiles']
    charge = task['charge']
    
    for attempt in range(2):  # 2 attempts
        try:
            result = calculate_pm7_properties(smiles, charge=charge, cleanup=True)
            
            if result and result.get('success'):
                return result
            
        except Exception as e:
            if attempt == 1:  # Last attempt
                return {
                    'smiles': smiles,
                    'success': False,
                    'error': f'Exception: {str(e)}'
                }
        
        time.sleep(1)  # Brief pause before retry
    
    return {
        'smiles': smiles,
        'success': False,
        'error': 'Failed after 2 attempts'
    }


def estimate_parallel_runtime(num_molecules, n_processes):
    """Estimate runtime based on benchmarks."""
    avg_time_per_calc_seconds = 144.0  # Average PM7 calculation time
    avg_calcs_per_mol = 4.1  # Average: 1 neutral + ~3 protonated forms
    
    total_calcs = num_molecules * avg_calcs_per_mol
    parallel_time_hrs = (total_calcs * avg_time_per_calc_seconds / n_processes) / 3600
    
    return parallel_time_hrs


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Parallel PM7 Calculator with Proton Affinity"
    )
    parser.add_argument('input_file', type=str, 
                       help="Path to CSV file with SMILES")
    parser.add_argument('--smiles_column', type=str, default='smiles',
                       help="SMILES column name")
    parser.add_argument('--project_name', type=str, default='pm7_results',
                       help="Output directory name")
    parser.add_argument('--n_processes', type=int, default=None,
                       help="Number of processes (default: all cores)")
    parser.add_argument('--force', action='store_true',
                       help="Skip confirmation")
    parser.add_argument('--total_chunks', type=int, default=1,
                       help="Total chunks")
    parser.add_argument('--current_chunk', type=int, default=1,
                       help="Current chunk (1-based)")
    
    args = parser.parse_args()
    
    system_info = check_system_resources()
    n_processes = args.n_processes or system_info['cpu_count']
    
    print(f"\n{'='*70}")
    print(f"PM7 CALCULATOR WITH PROTON AFFINITY - CHUNK {args.current_chunk}/{args.total_chunks}")
    print(f"{'='*70}")
    print(f"Processes: {n_processes}")
    print(f"Input: {args.input_file}")
    print(f"Project: {args.project_name}")
    print(f"{'='*70}\n")

    # Load data
    try:
        df = pd.read_csv(args.input_file)
        smiles_list = df[args.smiles_column].dropna().unique().tolist()
        print(f"✅ Loaded {len(smiles_list)} unique molecules")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Calculate chunk
    total_smiles = len(smiles_list)
    chunk_size = (total_smiles + args.total_chunks - 1) // args.total_chunks
    start_index = (args.current_chunk - 1) * chunk_size
    end_index = min(start_index + chunk_size, total_smiles)
    smiles_for_chunk = smiles_list[start_index:end_index]

    if not smiles_for_chunk:
        print("✅ No molecules in this chunk. Exiting.")
        return

    print(f"📊 Chunk range: {start_index} to {end_index-1}")
    print(f"📊 Molecules in chunk: {len(smiles_for_chunk)}")

    estimated_hours = estimate_parallel_runtime(len(smiles_for_chunk), n_processes)
    print(f"⏱️  Estimated time: ~{estimated_hours:.1f} hours\n")
    
    start_time = time.time()
    
    try:
        calculator = ParallelPM7Calculator(
            project_name=args.project_name, 
            n_processes=n_processes
        )
        calculator.process_molecules_parallel(smiles_for_chunk)
        
        elapsed_hrs = (time.time() - start_time) / 3600
        print(f"\n✅ CHUNK {args.current_chunk}/{args.total_chunks} FINISHED in {elapsed_hrs:.2f} hours")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()