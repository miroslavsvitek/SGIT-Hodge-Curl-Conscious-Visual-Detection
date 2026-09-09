# SGIT HGFT-X2 — Relational Hodge-GFT trajectory discovery

Frozen config SHA-256:

`21e8889fc7bd60421bb429311652d9c02db093c30246bb71e4cf0a06a1107a08`

## Purpose

HGFT-X2 implements the "GFT on the relations" variant.

Unlike GFT-X1:

EEG -> sensor GFT -> mode relations

HGFT-X2 uses:

EEG -> **oriented time-resolved edge relations** -> **Hodge/edge GFT** ->
gradient/curl/harmonic relational modes -> temporal spectral trajectory.

Experiment 1 is already opened, so this is exploratory discovery only.
Experiment 2 / VAN3 is hard-forbidden.

## 1. Sensor simplicial complex

Using all valid EEG channels, construct the frozen symmetric 6-nearest-neighbor sensor graph.

Every undirected graph edge is oriented by ascending node index i<j.

Every 3-clique is promoted to a triangle (2-simplex), oriented i<j<k.

Node-edge incidence:

B1.

Edge-triangle incidence:

B2.

The boundary identity must hold numerically:

B1 B2 = 0.

The Hodge operators are:

L_grad = B1^T B1

L_curl = B2 B2^T

L1 = L_grad + L_curl.

The positive-eigenvalue eigenvectors of L_grad span gradient-like edge flow.
The positive-eigenvalue eigenvectors of L_curl span circulatory/curl-like edge flow.
The remaining orthogonal complement is harmonic.

## 2. Relational edge signal

The relation is constructed **before** any GFT.

For oriented sensor-graph edge i->j and one-sample lag:

f_ij(t)
=
[x_i(t-1)x_j(t)-x_j(t-1)x_i(t)]
/
[sqrt((x_i(t-1)^2+x_i(t)^2)(x_j(t-1)^2+x_j(t)^2))+eps].

At 256 Hz the lag is 3.90625 ms.

This is:
- signed and antisymmetric under edge reversal;
- chronology-sensitive;
- amplitude-normalized;
- time resolved at the EEG sample scale.

It can be viewed as an oriented lead-lag/rotational relational flow in a two-sample delay embedding.

## 3. Hodge-GFT

Project f(t) onto the edge-space Hodge eigenmodes.

The primary hypothesis concerns the **curl subspace**, because curl modes represent closed
circulatory patterns on the relational network rather than a simple node potential gradient.

Twelve curl modes are selected at evenly spaced positive-eigenvalue ranks across the available
curl spectrum. Twelve gradient modes are selected analogously as a specificity control.

Selection depends only on the graph, not awareness labels or EEG outcomes.

## 4. Awareness trajectory

Primary conditions:

UNSEEN = detect/Nothing

SEEN = detect/Something.

For each condition and window:

W1 = 80-240 ms

W2 = 240-400 ms

W3 = 400-560 ms

compute mean squared coefficient of each selected curl mode across retained trials and time samples.

Convert the selected-mode energy vector to a probability distribution:

p_m = E_m / sum_m E_m

and use sqrt(p_m), i.e. Hellinger geometry.

Concatenate W1/W2/W3 curl spectra into the subject condition trajectory.

This removes total curl magnitude from the primary geometry. Therefore HGFT-X2 does not ask merely:

"Is there more curl energy in SEEN?"

It asks:

"Is the **distribution and temporal redistribution of circulatory relational flow across curl modes**
reproducibly different for SEEN and UNSEEN?"

## 5. Primary test

For each held-out subject, build SEEN and UNSEEN curl-trajectory templates from all other subjects.

Use the same symmetric two-condition LOSO cosine-template score as GFT-X1.

Candidate discovery requires:

1. N_evaluable >= 24
2. median curl-template score > 0
3. 9999-swap one-sided p <= 0.025.

Gradient-template performance is reported as a specificity control but is not part of the primary gate.

## 6. Secondary biological quantities

Report:

- curl fraction:
  E_curl / (E_gradient + E_curl + E_harmonic)

by condition and window;

- gradient trajectory score;

- consensus SEEN-minus-UNSEEN curl spectral difference;

- top curl modes with sensor-edge loadings;

- simplicial-complex diagnostics and harmonic dimension.

A positive primary result would suggest a reproducible **circulatory relational reorganization**
associated with subjective detection experience.

It would still be exploratory and would require an untouched Experiment-2 replication.

## 7. Nuisance control

Source EEG remains the official preprocessed K17 epoch FIF.

No extra filtering or rereferencing.

Before edge construction, within subject and label-blind:

- baseline-standardize each sensor using -90..0 ms across all retained valid detection trials;
- residualize each sensor x time sample against:
  - intercept,
  - chunk-centered stimlevel,
  - stimulus identity,
  - objective correctness if uniquely source-resolvable.

Only after this is the oriented relational flow f_ij(t) constructed.

## 8. Synthetic sanity

Before human data, code must demonstrate:

- a curl-topology condition difference yields a clearly positive curl-template score;
- pure global scaling of the same curl topology does not;
- a gradient-only topology difference does not produce a curl-template score;
- that gradient-only difference is visible in the gradient control.

If this fails, human HGFT-X2 is not run.

## Claim boundary

Positive:
exploratory evidence for awareness-specific temporal organization of Hodge-curl relational modes.

Negative:
close this exact lag-flow / clique-complex / selected-curl-spectrum formulation on Experiment 1.

Experiment 2 remains untouched in either case.
