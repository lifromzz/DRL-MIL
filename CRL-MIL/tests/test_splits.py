"""CLI regression tests; run from CRL-MIL with unittest discovery."""

from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


class SplitCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="crl_splits_")
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)

    def run_script(self, filename, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "process" / filename),
                                 *map(str, args)], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def check_folds(self, folder, expected, patient_lookup=None):
        tests = []
        expected_labels = expected.set_index("path")["STAS"].to_dict()
        for fold in range(5):
            frames = [pd.read_csv(folder / f"fold{fold}_{split}.csv") for split in SPLITS]
            sets = [set(frame.path) for frame in frames]
            self.assertEqual(set.union(*sets), set(expected.path))
            for i in range(3):
                self.assertEqual(len(frames[i]), len(sets[i]))
                self.assertEqual(frames[i].set_index("path")["STAS"].to_dict(),
                                 {name: expected_labels[name] for name in sets[i]})
                for j in range(i + 1, 3):
                    self.assertFalse(sets[i] & sets[j])
                    if patient_lookup is not None:
                        self.assertFalse({patient_lookup[n] for n in sets[i]} &
                                         {patient_lookup[n] for n in sets[j]})
            tests.extend(frames[2].path)
        self.assertEqual(Counter(tests), Counter(expected.path))

    def test_slide_cli_matches_published_folds(self):
        for dataset, published in [("BRACS", "BRACS_"), ("cptac_lung", "cptac_lung")]:
            with self.subTest(dataset=dataset):
                manifest = ROOT / "process/metadata" / f"{dataset}.csv"
                output = self.tmp / dataset
                self.run_script("create_kflod.py", "--input-csv", manifest, "--output-dir", output)
                self.check_folds(output, pd.read_csv(manifest))
                for fold in range(5):
                    for split in SPLITS:
                        name = f"fold{fold}_{split}.csv"
                        pd.testing.assert_frame_equal(pd.read_csv(output / name),
                                                      pd.read_csv(ROOT / "data" / published / name))

    def test_cptac_patient_cli_and_strict_mapping(self):
        manifest = ROOT / "process/metadata/cptac_lung.csv"
        expected = pd.read_csv(manifest)
        exceptions = pd.read_csv(ROOT / "process/cptac_patient_map.csv").set_index("path").patient_id.to_dict()
        lookup = {name: exceptions.get(name, name.rsplit("-", 1)[0]) for name in expected.path}
        for strict in [False, True]:
            with self.subTest(strict=strict):
                out = self.tmp / str(strict)
                options = ["--cptac-patient-map", ROOT / "process/cptac_patient_map.csv", "--strict-cptac"] if strict else []
                self.run_script("create_patient_split.py", "--dataset", "cptac", "--cptac-csv",
                                manifest, "--output-root", out, *options)
                folder = out / "cptac_lung_patient"
                self.check_folds(folder, expected, lookup)
                for fold in range(5):
                    for split in SPLITS:
                        name = f"fold{fold}_{split}.csv"
                        actual = pd.read_csv(folder / name)
                        if strict:
                            self.assertEqual(actual.patient_id.tolist(), actual.path.map(lookup).tolist())
                        else:
                            pd.testing.assert_frame_equal(actual, pd.read_csv(ROOT / "data/cptac_lung_patient" / name))

    def test_bracs_patient_cli_with_synthetic_workbook(self):
        # Exercise the documented Excel sheet/columns without gated BRACS data.
        labels = {"UDH": 0, "DCIS": 1, "IC": 2, "ADH": 3, "PB": 4, "N": 5, "FEA": 6}
        rows = []
        for index, label in enumerate(labels):
            for patient in range(10):
                patient_id = index * 10 + patient
                for suffix, slide_label in [("a", "N"), ("b", label)]:
                    rows.append({"WSI Filename": f"synthetic_{patient_id}_{suffix}",
                                 "Patient Id": patient_id, "WSI label": slide_label})
        table = pd.DataFrame(rows)
        workbook = self.tmp / "BRACS.xlsx"
        table.to_excel(workbook, sheet_name="WSI_Information", index=False)
        self.run_script("create_patient_split.py", "--dataset", "bracs", "--bracs-xlsx",
                        workbook, "--output-root", self.tmp)
        expected = pd.DataFrame({"path": table["WSI Filename"], "STAS": table["WSI label"].map(labels)})
        lookup = table.set_index("WSI Filename")["Patient Id"].to_dict()
        self.check_folds(self.tmp / "BRACS_patient", expected, lookup)

    def test_duplicate_slides_and_unknown_patients_are_rejected(self):
        duplicate = self.tmp / "duplicate.csv"
        pd.DataFrame({"path": ["same", "same"], "STAS": [0, 0]}).to_csv(duplicate, index=False)
        result = self.run_script("create_kflod.py", "--input-csv", duplicate, "--output-dir", self.tmp / "bad", success=False)
        self.assertIn("unique slide", result.stderr)
        unknown = self.tmp / "unknown.csv"
        pd.DataFrame({"path": ["unmapped_slide"], "STAS": [1]}).to_csv(unknown, index=False)
        result = self.run_script("create_patient_split.py", "--dataset", "cptac", "--cptac-csv",
                                 unknown, "--strict-cptac", "--output-root", self.tmp / "bad", success=False)
        self.assertIn("Cannot infer CPTAC patient", result.stderr)


if __name__ == "__main__":
    unittest.main()
