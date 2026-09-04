# Water O-O RDF comparison

This folder contains the LAMMPS input, compact final RDF table, experimental
references, and analysis scripts for the TRACE-V2 water model. Generated
checkpoints, trajectories, and the full `h2o.rdf` time series are intentionally
not tracked; create them locally before rerunning the analysis.

## What the script checks

Run from this folder:

```bash
python analyze_water_rdf.py
```

The script reads all complete blocks in `h2o.rdf`, writes the final cumulative
RDF to `rdf_analysis/rdf_trace_final.csv`, and saves
`rdf_analysis/rdf_oo_trace_vs_experiment.png`.

The current LAMMPS input uses:

```lammps
compute myRDF all rdf 100 1 1 1 2 2 2
fix 2 all ave/time 100 1 100 c_myRDF[*] file h2o.rdf mode vector ave running
```

with `type 1 = H` and `type 2 = O`. The correct RDF mapping is:

```text
g_HH = c_myRDF[2]
g_HO = c_myRDF[4]
g_OO = c_myRDF[6]
```

Because `ave running` is used, the final RDF block is the cumulative RDF average
over the completed trajectory. Do not average all RDF blocks again unless the
LAMMPS input is changed to write independent block averages.

## Experimental O-O reference

For ambient liquid water O-O RDF, use:

Skinner, L. B. et al., "Benchmark oxygen-oxygen pair-distribution function of
ambient water from x-ray diffraction measurements with a wide Q-range,"
J. Chem. Phys. 138, 074506 (2013).

This is also the O-O experimental benchmark used for the 295.1 K comparison in
the NEP-MB-pol water paper. The O-H and H-H references in that comparison are
from Soper's neutron-diffraction RDF data, but this test focuses on O-O.

Add the experimental data as:

```text
experimental_goo_skinner_295K.csv
```

with two columns:

```csv
r_A,g_OO
2.000,0.000
...
```

If the Skinner SI file is available locally, convert it with:

```bash
python convert_skinner_si.py \
  --input /path/to/Ambient_water_xray_data.txt \
  --output experimental_goo_skinner_295K.csv
```

Then rerun:

```bash
python analyze_water_rdf.py --experimental experimental_goo_skinner_295K.csv
```

The summary file reports the first O-O peak position/height and an interpolated
O-O RDF RMSE over the common 2.2-6.0 Angstrom interval.

## Manuscript partial-RDF figure

To produce the three-panel manuscript figure, including O-O, O-H, and H-H
partial RDFs, run:

```bash
python analyze_water_rdf.py \
  --experimental experimental_goo_skinner_295K.csv \
  --manuscript-figure ../../../docs/figures/trace_water_partial_rdfs.png
```

The O-H and H-H Soper curves used for the current manuscript figure are
figure-coordinate digitizations of the supplied NEP-MB-pol figure, not raw
Soper tabulations. Generate them once with:

```bash
python digitize_soper_nep_mbpol_figure.py \
  --figure /path/to/nep_mbpol_rdf_figure.png \
  --output-dir experimental_digitized
```

Then include them in the manuscript figure:

```bash
python analyze_water_rdf.py \
  --experimental experimental_goo_skinner_295K.csv \
  --experimental-oh experimental_digitized/experimental_goh_soper_298K_digitized.csv \
  --experimental-hh experimental_digitized/experimental_ghh_soper_298K_digitized.csv \
  --manuscript-figure ../../../docs/figures/trace_water_partial_rdfs.png
```

The O-H panel clips the intramolecular covalent peak at $g_{OH}=4$ so that its
intermolecular features remain visible.
