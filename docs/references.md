# References

Scientific literature cited by this pipeline's methodology (see `docs/methodology.md` for where each is applied), reproduced from the notebook's own citations and the team's BAH 2026 proposal reference list. No citation below has been added beyond what the notebook or proposal already references.

1. Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., & Torrence, M. H. (2011). *Illumination Conditions of the Lunar Polar Regions Using LOLA Topography.* — Primary methodology reference for horizon-angle modeling, illumination-fraction computation, and PSR definition (Sections 1–3 of `docs/methodology.md`); also the source of the ~1,180 km² Faustini PSR area used as a validation reference point.

2. Paige, D. A., Siegler, M. A., Zhang, J. A., et al. (2010). *Diviner Lunar Radiometer Observations of Cold Traps in the Moon's Polar Regions.* — Thermal cold-trap criteria used in candidate ice-trap detection (Section 4).

3. Hayne, P. O., et al. (2015). Cold-trap modeling referenced in candidate ice-trap detection (Section 4). *(Full citation as referenced in the original notebook; title/venue not independently re-verified beyond the notebook's in-code citation.)*

4. Rubanenko, L., Venkatraman, J., & Paige, D. A. (2019). Micro cold-trap theory, used for the ice-proximity criterion in candidate ice-trap detection and landing-site scoring (Sections 4–5). *(Full citation as referenced in the original notebook.)*

5. Watson, K., Murray, B. C., & Brown, H. (1961). Classical cold-trap theory, foundational reference for candidate ice-trap detection (Section 4). *(Full citation as referenced in the original notebook.)*

6. Arvidson, R. E., et al. (2002). Landing-hazard slope threshold, used as the 10° hard-exclusion criterion in landing-site selection (Section 5). *(Full citation as referenced in the original notebook.)*

7. Kreslavsky, M. A., & Head, J. W. (2000). Terrain-roughness metric (local slope standard deviation), used as the roughness criterion in landing-site selection (Section 5). *(Full citation as referenced in the original notebook.)*

8. Archinal, B. A., et al. (2018). *Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015.* Celestial Mechanics and Dynamical Astronomy. — Source of the IAU WGCCRE Moon rotation model implemented in `build_icrs_to_mcmf_rotation` for the ICRS-to-MCMF frame transform (Section 2).

9. Spudis, P. D., Bussey, D. B. J., Baloga, S. M., et al. (2013). *Evidence for Water Ice on the Moon: Results from Mini-RF Radar Observations.* — Cited in the team's proposal as the primary literature reference for the Radar Processing module (owned by Aditya Narayan; not implemented in this repository).

10. Raney, R. K., et al. (2012). *The Mini-RF Radar Instrument and Bistatic Observations of the Lunar Surface.* — Cited in the team's proposal reference list (Radar Processing module).

11. Campbell, B. A., Carter, L. M., & Campbell, D. B. (2006). *Radar Backscatter Characteristics of Rough Surfaces and Implications for Lunar Ice Detection.* — Cited in the team's proposal as the primary literature reference for the Data Fusion / AI-ML module (owned by Prince; not implemented in this repository).

12. Feldman, W. C., Maurice, S., Lawrence, D. J., et al. (1998). *Fluxes of Fast and Epithermal Neutrons from Lunar Prospector: Evidence for Enhanced Hydrogen at the Lunar Poles.* — Cited in the team's proposal reference list (Lunar Prospector hydrogen data, not part of this repository's terrain-analysis scope).

13. ISRO. *Bharatiya Antariksh Hackathon (BAH) 2026 – Challenge Statement.* — Challenge 8: Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning.

## Datasets Referenced (Not All Used in This Repository)

- NASA Planetary Data System (PDS). *Lunar Reconnaissance Orbiter (LRO) Mini-RF Data Archive.* — Radar module (not implemented here).
- NASA Planetary Data System (PDS). *Lunar Orbiter Laser Altimeter (LOLA) Digital Elevation Model.* — Source of the DEM this repository's terrain analysis is built on. See `docs/datasets.md`.
- Lunar Prospector Mission. *Neutron Spectrometer Hydrogen Abundance Dataset.* — Data Fusion module (not implemented here).
- Chandrayaan-2 Mission Documentation. *Dual Frequency Synthetic Aperture Radar (DFSAR) and Orbiter High Resolution Camera (OHRC), ISRO.* — Radar/OHRC modules (not implemented here).

## Note on Citation Verification

Items 3–7 above are reproduced exactly as cited in the original notebook's in-code comments and the team's proposal document; their full bibliographic details (volume, page numbers, journal) were not independently re-verified against the published literature during this documentation pass, since doing so was outside the scope of the notebook-to-repository conversion. If you need exact bibliographic entries for formal citation, verify each against the published source before use.