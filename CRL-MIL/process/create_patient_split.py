"""Patient-level train/val/test splits for BRACS and CPTAC-lung.

Each patient ID (BRACS Patient Id, or CPTAC case id) is assigned to only one of
train, validation, and test. Slide labels are unchanged. Stratification uses
one case-level label:

- BRACS: the most severe slide label of that patient
  (N < PB < UDH < FEA < ADH < DCIS < IC).
- CPTAC: the tumor subtype on that case. Label 0 is adjacent normal and is
  not used as the case label, because tumor cases also contain normal slides.
"""

import argparse
import os
import re
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
N_SPLITS = 5
VAL_RATIO = 0.2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")

BRACS_STAS = {"UDH": 0, "DCIS": 1, "IC": 2, "ADH": 3, "PB": 4, "N": 5, "FEA": 6}
BRACS_SEVERITY = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]


def cptac_case_id(slide_name, patient_map=None, strict=False):
    name = str(slide_name)
    if patient_map is not None and name in patient_map:
        return str(patient_map[name])
    matched = re.match(r"^(C3[LN]-\d+)-\d+$", name)
    if matched:
        return matched.group(1)
    if strict:
        raise ValueError(f"Cannot infer CPTAC patient for {name!r}; supply --cptac-patient-map")
    return name  # Historical fallback; this is not a verified patient identifier.


def most_severe(labels):
    rank = {label: index for index, label in enumerate(BRACS_SEVERITY)}
    return max(labels, key=lambda label: rank[label])


def assign_patients(patients, case_labels):
    """Return a list of (fold, split_name, patient_ids)."""
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    assignments = []
    for fold, (trainval_index, test_index) in enumerate(splitter.split(patients, case_labels)):
        trainval = patients.iloc[trainval_index]
        test = patients.iloc[test_index]
        train, val = train_test_split(
            trainval,
            test_size=VAL_RATIO,
            stratify=trainval["case_label"],
            random_state=SEED,
        )
        assignments.append((fold, "train", train["patient_id"].tolist()))
        assignments.append((fold, "val", val["patient_id"].tolist()))
        assignments.append((fold, "test", test["patient_id"].tolist()))
    return assignments


def write_folds(slides, assignments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for fold, split_name, patient_ids in assignments:
        selected = slides[slides["patient_id"].isin(patient_ids)].copy()
        selected = selected.sort_values(["patient_id", "path"])
        out_path = os.path.join(output_dir, f"fold{fold}_{split_name}.csv")
        selected[["path", "STAS", "patient_id"]].to_csv(out_path, index=False)

        overlap_ok = selected["patient_id"].nunique() == len(patient_ids)
        if not overlap_ok:
            raise RuntimeError(f"{output_dir} fold {fold} {split_name} lost a patient")
        summary_rows.append(
            {
                "fold": fold,
                "split": split_name,
                "n_slides": len(selected),
                "n_patients": selected["patient_id"].nunique(),
                "slide_label_counts": selected["STAS"].value_counts().sort_index().to_dict(),
                "case_label_counts": (
                    selected.drop_duplicates("patient_id")["case_label"].value_counts().sort_index().to_dict()
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "patient_split_summary.csv"), index=False)
    return summary_rows


def assert_no_patient_leak(slides, assignments):
    for fold in range(N_SPLITS):
        groups = {}
        for assigned_fold, split_name, patient_ids in assignments:
            if assigned_fold == fold:
                groups[split_name] = set(patient_ids)
        shared = (
            (groups["train"] & groups["val"])
            | (groups["train"] & groups["test"])
            | (groups["val"] & groups["test"])
        )
        if shared:
            raise RuntimeError(f"fold {fold} has patients in more than one split: {sorted(shared)[:5]}")
        covered = groups["train"] | groups["val"] | groups["test"]
        if covered != set(slides["patient_id"]):
            raise RuntimeError(f"fold {fold} does not cover every patient")
    test_ids = [patient for _, split, ids in assignments if split == "test" for patient in ids]
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(slides["patient_id"]):
        raise RuntimeError("Each patient must appear in exactly one test fold")


def validate_slides(slides):
    if slides[["path", "STAS", "patient_id"]].isna().any().any():
        raise ValueError("Slide IDs, labels, and patient IDs must not be missing")
    if slides["path"].duplicated().any():
        raise ValueError("Expected one row per unique slide ID")


def build_bracs(bracs_xlsx, data_dir=DATA_DIR):
    table = pd.read_excel(bracs_xlsx, sheet_name="WSI_Information")
    if table[["WSI Filename", "Patient Id", "WSI label"]].isna().any().any():
        raise ValueError("BRACS metadata contains missing slide IDs, patient IDs, or labels")
    if not table["WSI label"].isin(BRACS_STAS).all():
        raise ValueError("Unmapped BRACS label")
    slides = pd.DataFrame(
        {
            "path": table["WSI Filename"].astype(str),
            "STAS": table["WSI label"].map(BRACS_STAS).astype(int),
            "patient_id": table["Patient Id"].astype(int),
            "wsi_label": table["WSI label"].astype(str),
        }
    )
    validate_slides(slides)
    case_label = (
        slides.groupby("patient_id")["wsi_label"]
        .agg(lambda labels: most_severe(list(labels)))
        .rename("case_label")
    )
    slides = slides.merge(case_label, on="patient_id", how="left")
    patients = slides[["patient_id", "case_label"]].drop_duplicates().sort_values("patient_id")
    assignments = assign_patients(patients, patients["case_label"])
    assert_no_patient_leak(slides, assignments)
    output_dir = os.path.join(data_dir, "BRACS_patient")
    summary = write_folds(slides, assignments, output_dir)
    return output_dir, summary


def build_cptac(cptac_csv, data_dir=DATA_DIR, patient_map_csv=None, strict=False):
    table = pd.read_csv(cptac_csv)
    if table[["path", "STAS"]].isna().any().any():
        raise ValueError("CPTAC metadata contains missing slide IDs or labels")
    if not table["STAS"].isin([0, 1, 2]).all():
        raise ValueError("CPTAC labels must be 0 (normal), 1, or 2 (tumor subtypes)")
    slides = pd.DataFrame(
        {
            "path": table["path"].astype(str),
            "STAS": table["STAS"].astype(int),
        }
    )
    patient_map = {}
    if patient_map_csv is not None:
        mapping = pd.read_csv(patient_map_csv, dtype=str)
        if mapping[["path", "patient_id"]].isna().any().any() or mapping["path"].duplicated().any():
            raise ValueError("Patient map needs one unique, nonmissing path/patient_id pair per row")
        patient_map = mapping.set_index("path")["patient_id"].to_dict()
    unresolved = [name for name in slides["path"]
                  if name not in patient_map and not re.fullmatch(r"C3[LN]-\d+-\d+", name)]
    if unresolved and not strict:
        warnings.warn(f"{len(unresolved)} CPTAC slide IDs have no supplied patient mapping. "
                      "Reproducing the original fallback: each is treated as a separate patient. "
                      "See process/cptac_patient_map.csv for the four TCIA-verified exceptions; "
                      "use --cptac-patient-map and --strict-cptac for explicit official IDs.", stacklevel=2)
    slides["patient_id"] = slides["path"].map(
        lambda name: cptac_case_id(name, patient_map, strict))
    validate_slides(slides)
    tumor_labels = (
        slides.loc[slides["STAS"] != 0]
        .groupby("patient_id")["STAS"]
        .agg(lambda labels: tuple(sorted(set(int(label) for label in labels))))
    )
    if any(len(labels) != 1 for labels in tumor_labels):
        raise RuntimeError("A CPTAC case has more than one tumor subtype")
    if set(tumor_labels.index) != set(slides["patient_id"]):
        missing = set(slides["patient_id"]) - set(tumor_labels.index)
        raise RuntimeError(f"CPTAC cases without a tumor slide: {sorted(missing)[:5]}")
    case_label = tumor_labels.map(lambda labels: labels[0]).rename("case_label")
    slides = slides.merge(case_label, on="patient_id", how="left")
    patients = slides[["patient_id", "case_label"]].drop_duplicates().sort_values("patient_id")
    assignments = assign_patients(patients, patients["case_label"])
    assert_no_patient_leak(slides, assignments)
    output_dir = os.path.join(data_dir, "cptac_lung_patient")
    summary = write_folds(slides, assignments, output_dir)
    return output_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bracs", "cptac", "all"], required=True)
    parser.add_argument("--bracs-xlsx", help="BRACS.xlsx containing sheet WSI_Information")
    parser.add_argument("--cptac-csv", help="CSV with path (bare slide ID) and STAS columns")
    parser.add_argument("--cptac-patient-map", help="Optional CSV with path,patient_id for exceptional IDs or all slides")
    parser.add_argument("--strict-cptac", action="store_true", help="Reject slide IDs without an inferred or supplied patient ID")
    parser.add_argument("--output-root", default=DATA_DIR)
    args = parser.parse_args()
    if args.dataset in ("bracs", "all") and not args.bracs_xlsx:
        parser.error("--bracs-xlsx is required for BRACS")
    if args.dataset in ("cptac", "all") and not args.cptac_csv:
        parser.error("--cptac-csv is required for CPTAC")
    results = []
    if args.dataset in ("bracs", "all"):
        results.append(build_bracs(args.bracs_xlsx, args.output_root))
    if args.dataset in ("cptac", "all"):
        results.append(build_cptac(args.cptac_csv, args.output_root,
                                  args.cptac_patient_map, args.strict_cptac))
    for output_dir, summary in results:
        print(output_dir)
        for row in summary:
            print(
                f"  fold{row['fold']} {row['split']}: "
                f"{row['n_slides']} slides, {row['n_patients']} patients, "
                f"cases {row['case_label_counts']}"
            )


if __name__ == "__main__":
    main()
