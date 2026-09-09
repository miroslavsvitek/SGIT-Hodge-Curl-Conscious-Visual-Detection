# SGIT HGFT E18-R0 — literal matched replication protocol lock

Frozen **before any E18 EEG header or sample access**.

Replication config SHA-256:

`c28e1808fb9353d9c9e37b234edf0ba709683c262ac5ebb5925d8bf2d2354cf7`

HGFT-X2 discovery config SHA-256:

`21e8889fc7bd60421bb429311652d9c02db093c30246bb71e4cf0a06a1107a08`

Reference Hodge complex SHA-256:

`ec54f49377b9a97427b46f10305e8995af5fb7d9628dbd17e7d6f915de3944b6`

## Primary question

Does the unchanged HGFT-X2 Hodge-curl relational trajectory distinguish:

`PAS1 = no visual experience`

from

`PAS2 = weak visual experience`

in the independent Eklund & Wiens 2018 LARGE-Gabor threshold dataset?

## Frozen candidate cohort

All 35 subjects that passed E18-A2R1 behavioral feasibility enter technical neural QC.

No subject is selected from the published neural outcome or from any HGFT result.

A subject is evaluable only if, after frozen official technical/quality gates, at least:

- 25 finite critical PAS1 trials;
- 25 finite critical PAS2 trials;

remain.

Group gate:

`N >= 24`.

## Neural source

Primary source is the official final preprocessed individual FieldTrip file:

`Fp<subject>s<large_session>_final.mat`.

The source preprocessing is accepted as dataset-native preprocessing. No new filtering,
re-reference, interpolation, or resampling is allowed.

Known source-native processing includes:
- 0.1-Hz high-pass;
- downsample to 250 Hz;
- photodiode-anchored epochs -0.1..+0.6 s;
- bad-channel interpolation;
- blink ICA removal;
- average reference;
- baseline correction;
- official artifact handling.

The 250-Hz source rate is retained. HGFT-X2's discrete relation remains exactly a **one-sample lag**.
The data must not be resampled to 256 Hz.

## Exact transferred Hodge geometry

Use the exact frozen Experiment-1 geometry:

- 64 named channels in the frozen order;
- 222 oriented edges;
- 216 triangles;
- exact curl basis;
- exact gradient basis;
- exact 12 selected curl modes;
- exact 12 selected gradient modes.

The graph must not be rebuilt.

## Conditions

Critical trial codes: 1,2,3,4 only.

Primary:

- UNSEEN = `Subj_resp==1` = PAS1;
- SEEN = `Subj_resp==2` = PAS2.

PAS3 is excluded from the primary contrast but may enter label-blind nuisance fitting.

## Relational signal

Exactly the frozen normalized antisymmetric lag flow:

\[
f_{ij}(t)=
\frac{x_i(t-1)x_j(t)-x_j(t-1)x_i(t)}
{\sqrt{(x_i^2(t-1)+x_i^2(t))(x_j^2(t-1)+x_j^2(t))}+\varepsilon}.
\]

## Windows

- W1 = 80–240 ms
- W2 = 240–400 ms
- W3 = 400–560 ms

For each window, selected curl-mode energies are normalized to probabilities, square-root transformed
(Hellinger geometry), concatenated across windows, and unit normalized exactly as HGFT-X2.

## Primary test

Same symmetric LOSO condition-template score.

Support requires:

1. technical PASS;
2. N>=24;
3. median curl-template score > 0;
4. 9999 subject swaps, one-sided p<=0.025.

Seed: `90520274`.

Gradient is a secondary specificity control.

## No rescue

No PAS pooling, small-Gabor rescue, mode/window/edge selection, graph rebuild, resampling, threshold
relaxation, or alpha change is allowed after neural outcome.

A negative result closes this exact literal matched replication.
