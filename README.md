# SETI Selection Function v1.19

**Author:** Maurizio Massaro  
**Affiliation:** Independent Researcher  
**ORCID:** https://orcid.org/0009-0000-4967-5015  
**Repository:** https://github.com/MaurizioJakarta/seti-source-selection  

Version 1.19 is the referee-hardening release. It preserves the source-level
selection-function framework while correcting four assumptions that matter for
physical interpretation: catalog entries versus physical stellar systems,
receiver-beam chromaticity, survey-global nuisance variables, and cross-epoch
beacon phase coherence. It also relabels the Margot 94% recovery-derived
two-scan quantities as conditional marginal-recovery scenarios rather than
measured end-to-end event-efficiency bounds.

The Margot reconstruction still contains 47 Gaia EDR3 catalog sources. A
literature audit identifies five confirmed bound pairs within that set, giving
**42 physical systems** in the central system-level interpretation. The
correlated weak-scintillation catalog-source benchmark remains
`eta95 = 0.0719612`; after collapsing only those confirmed multiples, the
single-transmitter-per-system bracket is `0.0804387--0.0804641`, with the
equal-member benchmark `0.0804514`.

The GBT main beam is now available as a chromatic Gaussian,
`FWHM(nu)=8.4 arcmin*(1.42 GHz/nu)`. At the fiducial `1.35e13 W` EIRP the
hard-detectable count remains 46/47 at both band edges, so the headline
catalog-source continuous benchmark is unchanged.

Physical-system beaming is recalculated rather than reusing the 47-source
result. For the central marginal-recovery independence scenario and an equal
component-location model, the infinite-EIRP one-lobe orientation floor is
`beta = 32.3699 deg`; with `beta <= 90 deg`, the exact minimum radiated
narrowband power for any nontrivial 95% system-prevalence bound is
`4.64454e11 W` at `beta = 83.6633 deg`.

For the recovered Price/Ma GBT-L processing subset, repeated epochs can now be
evaluated with either a phase-coherent periodic clock or an independent
phase-reset model. At `P=1 h, q=0.5`, the temporal effective counts are
`30.9428` and `31.0083`, respectively; at `P=24 h, q=0.5`, they are
`386.279` and `402.786`.

The population likelihood also supports mixtures over a **shared
survey-global nuisance state** outside the product over systems.

Validation: `pytest` passes **95/95 tests**.

## Key v1.19 additions

- physical-system grouping and occupancy models;
- chromatic GBT beam functions;
- global-nuisance mixture likelihoods;
- coherent/phase-reset support for repeated Price/Ma epochs;
- exact system-level beaming recalculation;
- new regression tests and release consistency audits.

## Repository versus archival package

GitHub contains the implementation, tests, scripts, metadata, and compact
derived products needed to inspect and reproduce the analysis workflow.
Large generated selection surfaces and the complete frozen reproducibility
bundle are intended for the archival Zenodo release associated with v1.19.

No software license has yet been selected. Until a license is added, copyright
remains with the author and reuse rights are not granted beyond those provided
by applicable law.

## Citation

Citation metadata are provided in `CITATION.cff`. The archival DOI will be
added after the Zenodo v1.19 record is created.

## Tests

From the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

Expected result for v1.19:

```
95 passed
```
