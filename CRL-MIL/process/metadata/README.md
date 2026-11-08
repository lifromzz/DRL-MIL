# Cohort label manifests

`BRACS.csv` (547 slides) and `cptac_lung.csv` (2,136 slides) are the original
`path,STAS` inputs for the split generators. Their slide/label pairs were checked
against the already-published train/validation/test CSVs in `data/BRACS_` and
`data/cptac_lung`. They add no new cohort records, images, or clinical attributes.
The original row order is retained because shuffled `KFold` operates on row
indices: reordering a manifest changes the resulting slide-level folds.

Label definitions:

- BRACS: `UDH=0, DCIS=1, IC=2, ADH=3, PB=4, N=5, FEA=6`.
- CPTAC-Lung: `normal=0, LUAD=1, LUSC/LSCC=2`.

Use these manifests directly for both slide-level workflows and the CPTAC
patient-level workflow. BRACS patient grouping additionally needs the official
`BRACS.xlsx` workbook, sheet `WSI_Information`, containing `WSI Filename`,
`Patient Id`, and `WSI label`. Obtain it through the
[BRACS download portal](https://www.bracs.icar.cnr.it/download/) after completing
the dataset provider's registration/access process. The
[BRACS dataset description](https://www.bracs.icar.cnr.it/details/) documents the
workbook's patient/label metadata. Patient IDs cannot be inferred from the
`BRACS_###` slide identifier alone.

These are the cohorts used by this repository, not automatically updated copies
of all available BRACS/TCIA slides. If you use another cohort, provide your own
manifest and patient metadata and save its splits under a different directory.
