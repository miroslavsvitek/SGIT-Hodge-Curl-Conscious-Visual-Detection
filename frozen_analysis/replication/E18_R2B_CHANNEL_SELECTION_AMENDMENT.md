# E18-R2B technical channel-selection amendment

R2A resolved the 65-channel schema without computing any HGFT or condition-level neural outcome.

Exact R2A finding:

- `clean.label` count = 65;
- all 64 frozen HGFT-X2 reference labels are present;
- missing frozen labels = 0;
- exactly one extra label = `Nz`;
- frozen 64 reference labels occupy the exact frozen order in the representative file.

The only licensed change is therefore:

> for every subject, require the full exact frozen 64-label set, require the only extra label to be
> `Nz`, select the 64 frozen labels by name in the frozen reference order, and ignore `Nz`.

This is a technical file-schema adapter only. The HGFT-X2 graph, modes, lag, windows, Hellinger
trajectory, LOSO statistic, permutation test, alpha, and subject/trial gates are unchanged.

Frozen literal replication SHA-256:

`c28e1808fb9353d9c9e37b234edf0ba709683c262ac5ebb5925d8bf2d2354cf7`

R1 QC SHA-256:

`f82cb0c0486a0ecaa17869caf36c00e6cab3646e3ddd08f64300600ec7a4d168`

R2B channel-amendment SHA-256:

`1749613286bd5aa2d93cd0113ce0aedcb2502643e82fb105be4a1afdefde93e3`
