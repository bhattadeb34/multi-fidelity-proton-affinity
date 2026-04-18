"""
featurize/
==========
Feature generation pipeline for proton affinity ML models.

Modules
-------
  fp_maccs.py      — MACCS structural key fingerprints (167 bits)
  fp_morgan.py     — Morgan circular fingerprints, radius=2, count (1024)
  desc_rdkit.py    — 210 RDKit molecular descriptors × 3 states = 630
  desc_pm7.py      — 13 PM7 quantum descriptors × 3 states = 39
  desc_mordred.py  — 1613 Mordred 2D + 213 3D descriptors × 3 states
  desc_site.py     — 6 site-level descriptors (element, position, n_sites)
  build_features.py — orchestrator, run directly

Usage
-----
  python featurize/build_features.py --dataset all
  python featurize/build_features.py --dataset nist --no-3d
"""
