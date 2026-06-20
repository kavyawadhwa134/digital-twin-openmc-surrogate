# Validation Benchmark Options

This project currently uses OpenMC-generated pin-cell data for workflow validation. For stronger scientific validation, use the following benchmark ladder.

## Immediate Project Benchmarks

- OpenMC pin-cell sweep: local response-surface test for `keff`.
- XSBench: kernel-level benchmark for cross-section lookup performance.
- OpenMC/ENDF-B-VIII.0 HDF5 lookup: direct reference for microscopic cross-section prediction.

## Recommended External Benchmarks

- XSBench, ANL-CESAR: https://github.com/ANL-CESAR/XSBench
- ECP XSBench summary: https://proxyapps.exascaleproject.org/app/xsbench/
- OECD/NEA C5G7 MOX benchmark: https://www.oecd-nea.org/science/wprs/eg3drtb/NEA-C5G7MOX.PDF
- OECD/NEA time-dependent C5G7 benchmark: https://www.oecd-nea.org/jcms/pl_32145/deterministic-time-dependent-neutron-transport-benchmark-without-spatial-homogenisation-c5g7-td
- VERA Core Physics Benchmark Progression Problems: https://vera.ornl.gov/technical-reports/
- VERA benchmark specification PDF: https://corephysics.com/docs/CASL-U-2012-0131-004.pdf
- ICSBEP criticality safety benchmarks: https://www.oecd-nea.org/jcms/pl_20291/international-criticality-safety-benchmark-evaluation-project-icsbep-handbook
- IRPhE reactor physics benchmarks: https://www.oecd-nea.org/jcms/pl_20279/international-handbook-of-evaluated-reactor-physics-benchmark-experiments-irphe

## Suggested Use

- Use `XSBench` for the claim about replacing or accelerating cross-section lookup kernels.
- Use `VERA` pin-cell and assembly problems for pin-cell response-surrogate validation.
- Use `C5G7` for transport-method comparison and multigroup benchmark structure.
- Use `ICSBEP` or `IRPhE` only when the workflow is mature enough for experimental/integral validation.
