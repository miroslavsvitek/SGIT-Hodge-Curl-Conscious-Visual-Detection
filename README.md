# Reproducible Hodge-Curl Relational Geometry of Conscious Visual Detection

This repository accompanies manuscript **v0.6**:

**Miroslav Svítek. _Reproducible Hodge-Curl Relational Geometry of Conscious Visual Detection_. Manuscript draft v0.6 (2026).**

The repository provides the manuscript source and PDF, the exact frozen analysis scripts/configurations used for the exploratory discovery and independent replication, and the supplementary numerical objects required to reconstruct the frozen Hodge geometry.

## Scientific scope

The manuscript introduces an empirical core of **Spatiotemporal Geometric Information Theory (SGIT)**. Pairwise chronology-sensitive EEG relations are represented as antisymmetric edge flows and decomposed with Hodge theory into gradient, curl, and harmonic components. The primary observable is the temporal trajectory of normalized Hodge-curl modal energy.

The principal inferential result is permutation-based. Within each permutation draw, condition labels are swapped within subject and **all leave-one-subject-out (LOSO) templates are recomputed**. The standardized LOSO effect size `g_z` is therefore reported only descriptively because LOSO templates overlap across held-out subjects.

Main reported curl results:

| Dataset | N | median D | positive subjects | descriptive g_z | permutation p |
|---|---:|---:|---:|---:|---:|
| Exploratory discovery | 30 | 7.6270e-05 | 19/30 | 0.562 | 0.0004 |
| Frozen replication | 35 | 1.0875e-05 | 24/35 | 0.575 | 0.0064 |

The matched Hodge-gradient representation was unsupported in both datasets. The manuscript does **not** claim that the present sensor-space effect is a causal mechanism, a universal biomarker of consciousness, or a subject-level classifier.

## Repository contents

- `manuscript_v0.6.tex` — LaTeX source of manuscript v0.6.
- `manuscript_v0.6.pdf` — compiled manuscript v0.6.
- `references.bib` — bibliography.
- `figure2_permutation_nulls.pdf` — Figure 2 permutation-null density visualization.
- `frozen_analysis/discovery/` — frozen discovery scripts, protocol, and configuration.
- `frozen_analysis/replication/` — frozen replication scripts, protocol, configuration, and channel-selection amendment.
- `supplementary/` — exact frozen montage, edge/triangle lists, selected Hodge bases, mode ranks, and complete Hodge object.
- `CITATION.cff` — citation metadata for GitHub and citation-aware tools.
- `.zenodo.json` — metadata intended for GitHub–Zenodo release archiving.
- `ZENODO_DESCRIPTION.md` — human-readable Zenodo record description.
- `MANIFEST.md` — repository inventory and reproducibility notes.

## Third-party EEG data

Participant EEG data are **not redistributed** in this repository. The analyses use publicly available third-party datasets:

1. Discovery dataset: https://doi.org/10.17045/STHLMUNI.26879077
2. Frozen replication dataset: https://doi.org/10.17045/STHLMUNI.5418967

Please cite the original dataset and study publications when reusing those data.

## Reproducibility

The key reproducibility principle is **frozen transfer**. The replication used the same discovery-derived graph, oriented edge/triangle complex, numerical Hodge bases, selected modes, temporal windows, normalization, LOSO statistic, permutation algorithm, direction of inference, and decision threshold. The replication graph and Hodge basis were not rebuilt from the replication EEG.

The repository contains no replacement or modified participant data. To rerun the analyses, obtain the source datasets from their original repositories and follow the frozen protocol files under `frozen_analysis/`.

## Funding acknowledgement

This work was supported by the European Union under the project **Robotics and advanced industrial production** (No. CZ.02.01.01/00/22_008/0004590) and by the Slovak Grant Agency VEGA (No. 1/0779/25).

## Citation and DOI

The first public GitHub release is intended to be archived through Zenodo. After Zenodo assigns the release DOI, this section and the manuscript Data Availability statement should be updated with the final DOI and repository URL.

## License

Unless otherwise stated, the manuscript text, documentation, figures, and supplementary numerical objects are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Analysis source code in `frozen_analysis/` is additionally available under the **MIT License**. Third-party EEG datasets remain under the terms set by their original repositories and are not redistributed here.
