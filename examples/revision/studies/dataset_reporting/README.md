# Dataset reporting inventory

Comments: 2.6, 2.8.4. Newly written integration utility, not a recovered historical script.

`audit_inputs.py` counts processed Patch-seq sweeps/cells and optionally reads NWB trial metadata. It does not estimate delays or perform Comment 2.5 reliability analyses. Run through the central launcher as documented in `docs/revision-running.md`.

Counts describe the specified files, not automatically the manuscript inclusion set. Inputs and hashes are recorded in the output. Exact stimulation units, conversion factors and protocol matching require their separate source audits.
