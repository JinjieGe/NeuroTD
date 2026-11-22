# Scope and limits of the revision evidence

## Scope

The revision bundle covers the analyses listed in the [comment-to-code map](revision-evidence-map.md). Comment 2.5, including cross-trial and cross-sweep reliability, ICC, split-half, and BASC analyses, is outside this release. Private correspondence, document exports, raw NWB files, and large window checkpoints are excluded.

## Historical materials

1. **Window selection (2.3).** An early signal sweep reported selections at 65, 129, 257, 513, and 1025 samples. Its full scripts and configuration would be needed to reproduce that particular scan (historical folders `window_bias_rebuttal` and `window_score_methods`). The included 1,200 repetitions and search-range comparison cover the specified archived designs.
2. **Dataset inventory (2.6).** The historical `reviewer5_closure.zip` provided tables and figures. The repository includes a new metadata inventory utility alongside the existing iEEG and Patch-seq entry points.
3. **Earlier statistical audits (2.4).** The `audit_v1` through `audit_v3` scripts refer to historical source hashes, NWB assets, and window checkpoints. Reproduction requires matching versions of those inputs. The final regional comparisons use `session_inference/code/calibrated_welch.py`.
4. **Figure 4b and expression data.** The revised Figure 4 shows regional delay distributions and simultaneous intervals. Reconstructing the earlier heatmap would require the original channel, trial and aggregation choices. Reconciling the gene counts requires the original expression feature list.

## Statistical and biological interpretation

- Added noise can increase small estimated delay magnitudes and reduce large ones. The template experiment cannot rule out a regional SNR confound in the observed hippocampal differences. Shifting the same template guarantees shared events by construction.
- A shared region label does not establish a shared hemisphere, bundle, or neuron. Some anatomical labels require further verification.
- The Welch-type analysis estimates variability from trial summaries within three fixed sessions. In stress tests with shared group offsets outside the calibration library, false-positive rates can be high. Generalization to new sessions or participants requires additional data.
- The outer monotonic transform in the submitted score q and its aggregation of squared residuals are separate operations. The implemented E can rank candidates differently from q.
- The matched-control analysis reapplies Holm to stored rank p-values and aggregates per-record tracking errors. Its familywise false-positive rates apply to the simulated constant-delay controls. They do not estimate what fraction of real delay fluctuations are artifacts.

[Validation records](revision-validation.md) distinguish regenerated outputs from archived results. Source and packaged hashes are recorded separately in the manifests.
