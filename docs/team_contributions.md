# Team Contributions

Team LunaForge, National Institute of Technology Patna — Bharatiya Antariksh Hackathon (BAH) 2026, Challenge 8.

This document distinguishes exactly what is verified and included in **this repository** from what is proposed team-wide work owned by other members. Nothing outside the "Verified in this repository" column below has been implemented, reproduced, or claimed as code in this repository.

## Team Roster

| Member | Role (per proposal) | Primary Literature (per proposal) |
|---|---|---|
| Aditya Narayan | Remote Sensing — radar processing (MIDAS), CPR/DOP computation, hydrogen analysis | Spudis et al. (2013) |
| **Anjali Saini** | Data Engineer & GIS — QGIS terrain analysis, DEM processing, PSR mapping | Mazarico et al. (2011) |
| Prince | AI/ML Engineer — data fusion, clustering, A*/RRT path planning | Campbell et al. (2006) |
| Himanshu | Resource Analyst — ice confidence, volume estimation, dashboard | Paige et al. (2010) |

*(Roles and literature as stated in the team's BAH 2026 proposal document.)*

## Module-by-Module Attribution

| LIRAF Module (per proposal) | Owner (per proposal) | Verified in this repository? |
|---|---|---|
| Radar Processing (MIDAS, CPR/DOP, ejecta filter) | Aditya Narayan | **No** — not included; no radar-processing code, notebook, or output exists in this repository |
| **Terrain Analysis** (DEM, horizon angles, illumination, PSR mapping, slope/hillshade, ice-trap detection, landing-site selection) | **Anjali Saini** | **Yes** — fully verified from `notebooks/terrain_analysis/PSR_Mapping_original.ipynb`, cleaned and packaged into `src/terrain_analysis/` |
| Data Fusion (GMM clustering, multi-source integration) | Prince | **No** — not included; no fusion code, notebook, or output exists in this repository |
| Mission Planning — Rover Traverse (A*/RRT path planning) | Prince | **No** — not included |
| Mission Planning — Ice Volume Estimation (dielectric mixing model) | Himanshu | **No** — not included |
| OHRC-based landing-safety assessment (boulder detection, crater morphology) | (per proposal, contributes to Terrain/Radar context) | **No** — not verified in any uploaded file; not claimed here |

## What "Verified" Means Here

Every function in `src/terrain_analysis/` was extracted from, and diff-checked against, the actual working code in `PSR_Mapping_original.ipynb`. Nothing in that package was written from a description of what the pipeline *should* do — it was extracted from code that already ran and produced the outputs referenced in [Results](../README.md#results). Where the original notebook had ambiguities or internal inconsistencies (e.g. two different search-radius configurations for horizon-angle computation, two independently-computed slope rasters), those are documented rather than silently resolved — see `docs/methodology.md` and `docs/architecture.md`.

## What Is Explicitly Not Claimed

- No CPR, DOP, or hydrogen-abundance map exists in this repository, despite appearing in the team's proposal and pitch deck. Those belong to Aditya Narayan and Himanshu's respective modules.
- No GMM clustering, ice-confidence map, or fused multi-source output exists here. That belongs to Prince's Data Fusion module.
- No rover traverse path (A*/RRT) or ice-volume estimate exists here. That belongs to Prince and Himanshu's Mission Planning module.
- Any resemblance between this repository's landing-site outputs and the team's final combined LIRAF mission-planning deliverable is coincidental to the shared proposal design — this repository's `landing_site_scoring.py` uses only DEM, illumination, and this module's own ice-trap detection as inputs, not the team's fused ice-confidence map.

## Cross-Team Data Flow (For Context, Not Implementation)

Per the proposal's architecture, this module's outputs (`psr_mask.npy`, `illumination_fraction.npy`, `candidate_ice_trap_catalogue.json`) are intended to feed into the team's Data Fusion and Mission Planning stages. That integration code is not part of this repository and is not represented here as if it were.