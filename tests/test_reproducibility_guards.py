"""Unit tests for safeguards that do not require scientific recomputation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReproducibilityGuardTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("rdkit") is not None,
        "RDKit is required to import the target builder",
    )
    def test_target_builder_refuses_existing_bundle(self) -> None:
        module = load_module(
            "build_targets",
            PROJECT / "scripts" / "calculations" / "build_targets.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "nist1155_ml.parquet").touch()
            with self.assertRaises(FileExistsError):
                module.ensure_output_is_safe(output, force=False)
            module.ensure_output_is_safe(output, force=True)

    def test_complete_release_validator_is_read_only_by_default(self) -> None:
        source = (
            PROJECT / "scripts" / "analysis" / "validate_complete_release.py"
        ).read_text()
        self.assertIn("No files are written by default", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("subprocess.run([\"mopac", source)

    def test_pip_lock_is_sorted_and_hash_pinned(self) -> None:
        module = load_module(
            "generate_pip_lock",
            PROJECT / "scripts" / "reproducibility" / "generate_pip_lock.py",
        )
        report = {
            "install": [
                {
                    "metadata": {"name": "Z_pkg", "version": "2"},
                    "download_info": {"archive_info": {"hashes": {"sha256": "b" * 64}}},
                },
                {
                    "metadata": {"name": "a.pkg", "version": "1"},
                    "download_info": {"archive_info": {"hashes": {"sha256": "a" * 64}}},
                },
            ]
        }
        self.assertEqual(
            module.lock_lines(report),
            [
                f"a.pkg==1 --hash=sha256:{'a' * 64}",
                f"Z_pkg==2 --hash=sha256:{'b' * 64}",
            ],
        )
        minimal = module.minimal_resolution(report)
        self.assertEqual([item["name"] for item in minimal["packages"]], ["a.pkg", "Z_pkg"])

    def test_release_manifest_is_deterministic(self) -> None:
        script = PROJECT / "scripts" / "reproducibility" / "build_release_manifest.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.txt").write_text("b")
            (root / "a.txt").write_text("a")
            subprocess.run([sys.executable, str(script), str(root)], check=True)
            lines = (root / "SHA256SUMS").read_text().splitlines()
            self.assertEqual([line.split("  ", 1)[1] for line in lines], ["a.txt", "b.txt"])
            refused = subprocess.run(
                [sys.executable, str(script), str(root)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)

    def test_artifact_layout_is_separate_from_data_and_code(self) -> None:
        module = load_module(
            "artifact_store",
            PROJECT / "scripts" / "reproducibility" / "artifact_store.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            code = release / "code"
            data = release / "data"
            code.mkdir()
            data.mkdir()
            layout = module.ReleaseLayout.discover(code_root=code)
            self.assertEqual(layout.code, code.resolve())
            self.assertEqual(layout.data, data.resolve())
            self.assertEqual(layout.artifacts, (release / "artifacts").resolve())
            self.assertNotEqual(layout.artifacts, layout.data)

    def test_checkpoint_exporter_validates_before_publish(self) -> None:
        source = (
            PROJECT
            / "scripts"
            / "reproducibility"
            / "export_model_checkpoints.py"
        ).read_text()
        validation = source.index("np.allclose(observed, wanted")
        publication = source.index("shutil.copytree(staging, destination)")
        self.assertLess(validation, publication)


if __name__ == "__main__":
    unittest.main(verbosity=2)
