# Five-fold data splits

These are portable versions of the original `create_patient_split.py` and
`create_kflod.py` (historical spelling retained). Dataset locations are command-line
arguments; the original seed, stratification, row order, and splitting rules are
preserved. Input validation rejects missing identifiers and duplicate slides.

## Which split is which?

| Script | Splitting unit | Outer test split | Inner validation split |
| --- | --- | --- | --- |
| `create_patient_split.py` | Patient/case, followed by expansion to all slides | `StratifiedKFold(5, shuffle=True, random_state=42)` on case labels | Stratified on case labels; 20% of the remaining patients |
| `create_kflod.py` | One CSV row = one unique slide | `KFold(5, shuffle=True, random_state=42)`; **not stratified** | Stratified on slide `STAS`; 20% of the remaining slides |

With five folds, the approximate train/validation/test proportions are
**64% / 16% / 20%** of the splitting units. Patient-level slide counts can differ
from these proportions because patients have different numbers of slides.
Every patient (or slide for the slide protocol) appears in exactly one test fold.
Train/validation membership can change between folds. This is not nested CV.

Patient-level splitting keeps all slides of one patient together **within each
fold**. The slide-level script does not enforce this restriction: a patient can
appear in more than one partition. This distinction concerns partitioning, not
whether the downstream model predicts a slide or a patient.

## Inputs and label conventions

- Slide-level CSV: columns `path,STAS`; `path` is a unique slide identifier,
  without the feature file's `.pt` suffix. Preserve the input row order to
  reproduce historical slide splits. Additional columns are retained.
- BRACS patient metadata: the official `BRACS.xlsx`, sheet `WSI_Information`,
  with `WSI Filename`, `Patient Id`, and `WSI label`.
- CPTAC patient metadata: CSV with `path,STAS`, using bare slide IDs such as
  `C3L-00001-21` rather than full paths or `.svs` filenames.

BRACS slide labels retain `UDH=0, DCIS=1, IC=2, ADH=3, PB=4, N=5, FEA=6`.
The **case stratification label** is the most severe slide label under
`N < PB < UDH < FEA < ADH < DCIS < IC`; the slide labels are not replaced.
CPTAC case stratification uses the unique nonzero tumor subtype. Label `0`
denotes adjacent normal; normal and tumor slides from one case remain together.
Cases with conflicting tumor subtypes or without a tumor slide are rejected.

### CPTAC identifiers checked against TCIA

For standard `C3L-<digits>-<digits>` / `C3N-<digits>-<digits>` slide IDs, the final
numeric component is removed to obtain the case ID. The historical code retains
an unrecognized slide ID as a singleton grouping key. For the four nonstandard
IDs in the original 2,136-slide input, the official TCIA mapping confirms four
different patients, each represented by one slide in this input:

| Slide ID | Official case ID |
| --- | --- |
| `9f905736-f662-41d6-b3ac-16758d_D1_D1` | `11LU013` |
| `93e30fd5-e57e-4503-a175-863c7d_D1_D1` | `11LU016` |
| `5a84eae1-197e-4463-ad65-59becc_D1_D1` | `11LU022` |
| `2f2e5477-42a4-4906-a943-bf7f80_D1_D1` | `11LU035` |

Source: [TCIA CPTAC-LUAD image/subject table](https://pathdb.cancerimagingarchive.net/listofimages/528?page=0)
and [official CPTAC metadata](https://pathdb.cancerimagingarchive.net/system/files/collectionmetadata/202407/cptac_metadata_07-09-2024.csv),
checked on 2026-11-08. The four mappings are included in `cptac_patient_map.csv`.
Thus the original grouping has 441 cases, and these four fallback IDs do not
introduce patient leakage in the supplied cohort. Do not assume that singleton
fallback is valid for other unrecognized filenames.

The default reproduces historical grouping IDs and fold membership, with an
explicit warning for nonstandard IDs. For explicit official IDs, supply
`--cptac-patient-map process/cptac_patient_map.csv --strict-cptac`.
Strict mode rejects any remaining unrecognized ID. A mapping CSV may cover only
exceptions or all slides and must contain unique `path,patient_id` rows.
**Using official IDs changes the sorted patient order and can change the fold
assignments; it is a separate split version, not a replacement for published
historical folds.**

## Usage

Run from `CRL-MIL/`:

```bash
python -m pip install -r process/requirements.txt

# Slide-level, five folds; use a distinct output folder for each dataset.
python process/create_kflod.py --input-csv /path/to/BRACS.csv --output-dir data/BRACS_slide
python process/create_kflod.py --input-csv /path/to/cptac_lung.csv --output-dir data/cptac_lung_slide

# Patient-level; preserve historical fold assignments.
python process/create_patient_split.py --dataset bracs --bracs-xlsx /path/to/BRACS.xlsx --output-root data
python process/create_patient_split.py --dataset cptac --cptac-csv /path/to/cptac_lung.csv --output-root data

# Optional explicit official IDs; write this distinct split version elsewhere.
python process/create_patient_split.py --dataset cptac --cptac-csv /path/to/cptac_lung.csv --cptac-patient-map process/cptac_patient_map.csv --strict-cptac --output-root data/official_ids
```

Each dataset produces `fold0_train.csv`, `fold0_val.csv`, `fold0_test.csv`,
through fold 4. Patient-level outputs are under `BRACS_patient/` and
`cptac_lung_patient/`, contain `path,STAS,patient_id`, and include a
`patient_split_summary.csv` with slide/patient/label counts. Existing files at
the chosen output location are overwritten. Metadata and newly generated folds
are not bundled with these scripts; provide your own local input tables.

## Validation on the original inputs

All 60 generated CSVs (two datasets x two splitting units x five folds x three
partitions) were checked against the original scripts with identical results.
Every fold covers all slides, has disjoint slide partitions, and contributes
exactly one test assignment per slide across five folds. Patient splits have
disjoint grouping IDs in every fold. BRACS contains 547 slides from 189 patients;
CPTAC-Lung contains 2,136 slides from 441 case groups, including the four
TCIA-verified cases above.

For comparison, slide-level folds contain patients in multiple partitions:
79-90 patients per fold for BRACS and 349-367 for CPTAC-Lung. This is expected
for slide-level evaluation and must not be described as patient-independent.
All 2,136 CPTAC slide-to-case assignments were checked against official TCIA
data: 2,134 occur in the linked metadata CSV, while `C3N-02282-25` and
`C3N-02282-30` are listed under `C3N-02282` in the
[current TCIA image table](https://pathdb.cancerimagingarchive.net/?order=imageid&page=1908&search_api_fulltext=&sort=asc).
Using these official case IDs to audit the historical patient folds also gives
zero within-fold patient overlap. The explicit mapping/strict mode was tested
separately and its output IDs agree with TCIA.

Validation used pandas 2.3.3 and scikit-learn 1.6.1. This audit covers generated
splits; it does not retrospectively establish which CSVs were used by a
particular training run or overwrite the repository's existing data files.
