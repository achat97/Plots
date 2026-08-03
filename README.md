# Sonar Pulse Detection — Pipeline Guide

Detects, classifies, and localizes active-sonar pulses (CW / LFM / HFM, both sweep directions)
in ocean noise. A CNN–LSTM encoder-decoder reads a 5 s spectrogram and emits a list of pulses,
each as (type, t_start, t_stop, f1, f2).

The pipeline has 10 steps. Steps 1–4 build the data, 5 finds hyperparameters, 6–7 train and
evaluate, 8 analyzes behavior, 9 runs on real recordings, and 10 measures the detected pulses
precisely.

---

## Step 1 — Recording → noise segments  (`wav_spectogram.py`)

**What it does:** reads a `.wav`, resamples it to `FS` (50 kHz), cuts it into 5 s segments, and
z-scores each segment. These segments are the ocean-noise backgrounds everything is built on.

**Requirement:** the recording must be **natively sampled at ≥ 50 kHz**. Upsampling cannot create
content above the recording's own Nyquist, so the top of the spectrogram would be empty — the
script refuses such files.

**How to run:** it is not a CLI script — uncomment the chain at the bottom, set the filename, run
once per recording, re-comment:

```python
file_name = "hat01"                                  # prefix for the output files
data = read_resample("hat01.wav", file_name)          # resample to FS (from data_config.py)
segments, _ = split(data, FS, TIME_MAX, file_name)    # cut into 5 s pieces
normalized = noise_normalize(segments, file_name)     # z-score each -> hat01_normalized.npy
```

The `*_normalized.npy` file is what the next step consumes.

---

## Step 2 — Generate the dataset  (`pulses.py`)

**What it does:** takes the normalized segments and builds training examples. Every segment is
included once as a *noise-only* example; on top of that, `n_pulse_examples` examples are created
by drawing a random segment (with replacement) and injecting 1–4 synthetic pulses into it.

**What a pulse is:** random type (CW / LFM / HFM), random sweep direction (half of all swept
pulses descend), SNR ~ U(−15, 40) dB relative to the noise, frequency band inside 1–24.99 kHz
with 100–4000 Hz bandwidth, plus reverberation echoes and a broadband source-noise patch sitting
`delta` ~ U(15, 30) dB *below* the chirp (hull/machinery noise that fades together with the
pulse). All ranges live in `data_config.py`.

**How to run** (bottom of `pulses.py`, or a Python shell):

```python
generate_train(np.load("hat01_normalized.npy"), 15000, prefix="hat01_", seed=0)
```

| argument | meaning |
|---|---|
| `data` | the normalized segments from step 1 |
| `n_pulse_examples=15000` | number of pulse examples to create (segments reused with replacement); `0` gives a pure noise-only part |
| `prefix="hat01_"` | output files are `hat01_INPUT.npy`, `hat01_AXES.npz`, `hat01_TARGET.npy`, `hat01_META.npy` |
| `seed=0` | makes the part reproducible — **use a different seed per recording**, or every part gets identical pulses |

**Outputs:** `INPUT` = dB spectrograms (memory-mapped), `AXES` = time/frequency axes, `TARGET` =
the pulse rows `[cw,lfm,hfm,eos, t_start,t_stop,f1,f2]` per example, `META` = one row
`[snr, distance, n_pulses, seg_idx]` per example.

**Extra noise-only data** (recommended, from a *different* recording):

```python
generate_train(np.load("hat02_normalized.npy"), 0, prefix="hat02n_", seed=1)
```

---

## Step 3 — Merge the parts  (`merge_datasets.py`)

**What it does:** concatenates the per-recording parts into the one dataset the rest of the
pipeline reads, and adds a `wav_id` column to META (0, 1, 2… in the order you list the parts) so
each example remembers which recording it came from — the split in step 6 needs this.

```bash
python merge_datasets.py --prefixes hat01_,hat02n_ --out ""
```

| flag | meaning |
|---|---|
| `--prefixes` | comma-separated part prefixes from step 2; order defines `wav_id` |
| `--out ""` | output prefix; empty writes the plain `INPUT.npy` / `TARGET.npy` / `META.npy` / `AXES.npz` all later steps expect by default |
| `--chunk 512` | (optional) spectrograms copied per read — only a memory bound |

Run this even with a single part if you skipped extra noise (`--prefixes hat01_`).

---

## Step 4 — Pad the targets (one-time)

**Why:** the decoder trains on fixed-length target sequences; `TARGET.npy` rows have variable
length. **How:** uncomment the two lines near the top of `train_model.py`, run once, re-comment —
or in a shell:

```python
y = np.load("TARGET.npy", allow_pickle=True)
save_padded_sequences(y, "padded_sequences.npy", target_length=CONFIG["max_length"])
```

---

## Step 5 — Hyperparameter search

**Data rule:** the search runs on a **separate tuning dataset** built from *other recordings*
(repeat steps 1–4 into `INPUT_tune.npy`, `padded_sequences_tune.npy`, `META_tune.npy`), so tuning
never touches the data you train and test on.

**5a. Search.** Two equivalent scripts — `tune_hparams.py` (Ray: parallel trials + early
stopping) or `random_search.py` (plain loop, use when Ray won't start):

```bash
python random_search.py --input INPUT_tune.npy --target padded_sequences_tune.npy \
       --meta META_tune.npy --num-samples 40 --epochs 25 --subset 8000
```

| flag | meaning |
|---|---|
| `--input / --target` | the tuning dataset files |
| `--meta` | tuning META; makes the internal train/val split segment-disjoint (recommended; omit for the old example-level split) |
| `--num-samples 40` | how many random configurations to try |
| `--epochs 25` | training epochs per configuration |
| `--subset 8000` | cap on training examples per configuration (0 = all) — keeps the search fast |
| `--ar-max-examples 1000` | validation examples used for the per-epoch AR score (0 = all) — bounds the cost of generation inside each trial |
| `--seed 0` | seed for sampling configs and the split |

Ray version adds: `--grace 5` (epochs a trial is protected before ASHA may stop it), `--no-asha`
(disable early stopping), `--gpu-per-trial 1`.

**Output:** `tuning_results.csv`, one row per configuration, ranked by **AR score** (the
validation criterion — see Background). The teacher-forced `val_loss` is logged alongside.

**5b. Rerun the best under several seeds** — separates real architecture differences from luck:

```bash
python rerun_top_configs.py --csv tuning_results.csv --top-k 3 --seeds 0,1,2 \
       --epochs 25 --meta META_tune.npy
```

| flag | meaning |
|---|---|
| `--csv` | the results file from 5a |
| `--top-k 3` | how many of the best configurations to retrain |
| `--seeds 0,1,2` | each configuration is retrained once per seed; you get mean ± std |

It prints the winner as a ready **CONFIG dict → paste it into `model.py`'s `CONFIG`**.

---

## Step 6 — Train  (`train_model.py`)

**What it does:** reads `INPUT.npy` / `padded_sequences.npy` / `META.npy`, makes the
**segment-disjoint** train/val/test split (see Background), trains on *train*, and after every
epoch scores the model on *val* with the AR score — the epoch with the best AR score is saved.
*Test* is never touched here.

```bash
python train_model.py --epochs 50 --batch-size 32 --lr 1e-3
```

| flag | default | meaning |
|---|---|---|
| `--epochs` | 50 | training epochs |
| `--batch-size` | 32 | batch size |
| `--lr` | 1e-3 | learning rate |
| `--weight-decay` | 0 | L2 regularization (AdamW) |
| `--grad-clip` | 1.0 | gradient-norm clip |
| `--optimizer` | adam | `adam` or `adamw` |
| `--ar-max-examples` | 0 | val examples for the per-epoch AR score (0 = all) |

**Outputs:** `best_model_epoch_E_arscore_S.pth` — contains the weights **and** the CONFIG it was
trained with, so every later script rebuilds the exact architecture and split from the checkpoint
alone (plus the winning epoch's component metrics under `val_metrics`). Also
`train_losses.npy`, `val_losses.npy` (teacher-forced, logging only), `val_ar_scores.npy` (the
selection curve), and `val_components.csv` — one row per epoch with F1, precision, recall,
count-exact, **count_bias** (drifting positive = the model is starting to over-emit pulses),
type accuracy, and the four MAEs, so you can judge checkpoints by their component profile
instead of the scalar alone.

---

## Step 7 — Evaluate on the test split  (`evaluate.py`)

```bash
python evaluate.py --checkpoint best_model_epoch_37_arscore_0.2140.pth --plots
```

| flag | meaning |
|---|---|
| `--checkpoint` | the `.pth` from step 6 (required) |
| `--input / --target / --meta` | dataset files (defaults: `INPUT.npy`, `padded_sequences.npy`, `META.npy`) |
| `--quiet` | skip the per-sample prediction/truth printout |
| `--plots` | save the figure set to `eval_plots/` |

Reports pulse-count accuracy, detection precision/recall/F1, type accuracy, per-quantity
regression error, and the test AR score — **these are the numbers to quote**, computed only on
segments the model never saw.

---

## Step 8 — Behavior analyses

**SNR analysis** — recall/precision vs SNR, LFM/HFM accuracy vs curvature, false-positive autopsy:

```bash
python snr_analysis.py --checkpoint best_model_...pth
```
`--split test` (default; `all` mixes in training data → optimistic) · `--snr-bins
-15,-10,-5,0,5,10,20,30,40` (bin edges, match the SNR range) · `--high-snr 25` (threshold for the
high-SNR curvature slice; keep below SNR_MAX) · `--outdir snr_analysis`.

**Offset sweep** — how robust the model is to a global level shift of the input:

```bash
python offset_sweep.py --checkpoint best_model_...pth --offsets=-20,-10,-5,0,5,10
```
Note the `=` (a leading `-` confuses the shell). `--split test` default; CSV + F1/recall-vs-offset
plot in `--outdir offset_sweep`.

**False alarms per hour** — the deployment number:

```bash
python false_alarm_sweep.py --wav quiet1.wav quiet2.wav --checkpoint best_model_...pth
```
The recordings must be **pulse-free** (every detection counts as a false alarm) and **not used in
training** (the model is quieter on memorized backgrounds). `--thresholds 0,0.2,...,0.99` sweeps
the same `--min-confidence` you would set in step 9; pair the FA/hour at a threshold with recall
from the SNR analysis to pick an operating point. CSV + plot in `--outdir fa_sweep`.

**Ablations** — retrain controlled variants on the tuning data:

```bash
python ablation_study.py --studies freq,bidir,kernel --seeds 0,1,2 --epochs 25 --meta META_tune.npy
```
`--studies`: `freq` = how many CNN blocks halve the frequency axis, `bidir` = bi- vs
unidirectional encoder, `kernel` = CNN kernel shape. One CSV row per (variant, seed).

---

## Step 9 — Detect pulses in a real recording  (`detect_pulses.py`)

```bash
python detect_pulses.py --wav recording.wav --checkpoint best_model_...pth --plot --xlsx
```

**What it does:** resamples the recording to the checkpoint's rate, slides a 5 s window
(z-scoring each, exactly like training), predicts per window, merges detections of the same pulse
across overlapping windows, prints the table, saves `<wav>_detections.csv`.

| flag | default | meaning |
|---|---|---|
| `--gate` | 0.5 | max start-time gap (s) to merge two window-detections of the same pulse |
| `--freq-gate` | 500 | detections merge only if their bands, padded by this (Hz), overlap — keeps simultaneous pulses in different bands separate |
| `--min-confidence` | 0 | drop detections with `confidence_det` below this (pick via step 8's FA sweep) |
| `--hop` | window/2 | window hop in seconds |
| `--fs` | from ckpt | only for old checkpoints without a stored config |
| `--plot / --xlsx` | off | annotated spectrogram PNG / Excel export |

Each detection carries `confidence_det` = 1 − P(EOS) ("something is there") and
`confidence_type` = P(predicted type). Heuristic scores, not calibrated probabilities.

---

## Step 10 — Refine the detected pulses  (`refine_detections.py`)

**What it does:** measures each detected pulse directly in the spectrogram, at full frequency
resolution. The network localises a pulse but estimates its frequencies from a feature map whose
frequency axis has been halved in every CNN block, so its band carries an error of a few hundred
Hz. Once the time span is known, the ~24 Hz STFT bins can be used directly: the script tracks the
ridge of peak energy through the pulse, fits the linear (LFM) and hyperbolic (HFM) sweep laws, and
reports whichever fits better.

```bash
python refine_detections.py --wav recording.wav --detections recording_detections.csv --plot
```

| flag | default | meaning |
|---|---|---|
| `--wav / --detections` | — | the recording and the CSV from step 9 (both required) |
| `--out` | `<detections>_refined.csv` | output CSV |
| `--context` | 0.5 | seconds of signal kept on each side of a pulse |
| `--pad-hz` | 1500 | width added to each side of the predicted band when searching |
| `--snr-db` | *auto* | ridge threshold above the median level of a time bin — by default chosen per pulse from 4/6/8/12/16 dB |
| `--rel-db` | *auto* | level range kept below the strongest column — by default chosen per pulse from 12/20/30 dB |
| `--min-points` | 8 | fewest ridge points accepted for a refinement |
| `--max-rmse-bins` | 3.0 | largest accepted fit residual, in frequency bins |
| `--min-coverage` | 0.3 | smallest fraction of the detected span the tracked ridge must cover |
| `--min-margin` | 0.1 | smallest residual gap between the two sweep laws for the LFM/HFM distinction to be trusted; below this the type is reported as `swept` |
| `--plot / --max-plots / --outdir` | off / 20 / `refined` | overview of all refinements plus a spectrogram per pulse |

**Thresholds tune themselves.** A faint pulse needs a permissive ridge threshold to yield any
track at all; a loud pulse beside an interfering tone needs a strict one. No single setting suits
both, so the script tries several per pulse and keeps the one whose sweep fit has the lowest
residual — an internal quality measure that needs no labels. Costs about 0.1 s per pulse. Pass
`--snr-db` or `--rel-db` explicitly to fix either one.

**What it gives you** (added as columns beside the network's own, which are never overwritten):

- `refined_t_start_s`, `refined_t_stop_s` (+ `_hms`) and `refined_duration_s` — the pulse's span,
  taken from the first and last tracked point
- `refined_f1_hz`, `refined_f2_hz` with `refined_f1_se_hz`, `refined_f2_se_hz` — the ± to quote
- `refined_direction` — up, down or flat, from the sign of the fitted slope. The network gets this
  wrong whenever the sweep is narrower than its own frequency error, since the direction is then
  the sign of a difference smaller than the noise on either endpoint
- `refined_type` — LFM, HFM or CW from the fit; `swept` when the two laws fit equally well
- `refined_bandwidth_hz`, `refined_slope_hz_per_s`, `refined_rmse_hz`, `refined_n_points`
- `refined_snr_db`, `refined_rel_db` — the thresholds chosen for that pulse
- `refined` = 0 with a reason in `refined_note` when the ridge was too short or too scattered to
  fit; those pulses keep the network's values

**Times are refined too, and the fit never extrapolates.** The tracked ridge shows where the pulse
actually starts and stops, so those points become the reported span and the sweep is only ever
evaluated between them. Reading the fit at the network's span instead would extrapolate whenever
that span is the wider of the two, and the error grows with the sweep rate: on a 10 kHz/s sweep, a
span 0.2 s too wide at each end puts the reported endpoints 2 kHz off. A ridge covering less than
`--min-coverage` of the detected span is rejected rather than reported.

**CW pulses get one frequency.** A constant tone has a single centre frequency, but the decoder's
four regression heads are independent, so the network can report two endpoints hundreds of Hz
apart for a CW detection. The fit constrains them to one tone: both endpoints are set to the mean,
and the standard error of that mean is reported.

**It never adds or removes detections** — recall, false alarms and the operating threshold are all
decided in steps 8 and 9. This step only measures what was already found.

---

**Figures.** All plotting shares `plot_style.py`: one colour per pulse type (CW/LFM/HFM), sentence-case labels with units in brackets, linear axes for rates and counts, and every reference line carrying a legend entry. Change fonts or colours there once and every figure follows.

## Background — the concepts the pipeline leans on

**Normalization.** Every model input is the dB spectrogram of a z-scored 5 s window *including
its pulses* — identical convention in training and deployment. SNR is defined against the
pre-injection noise RMS (z-scoring rescales globally, so the ratio is untouched).

**Segment-disjoint split.** Examples are grouped by the background segment they were built on
(META's `wav_id`, `seg_idx`). The split shuffles *segments* and deals each group — the segment's
noise-only copy plus every pulse example on it — entirely to one of train/val/test. Backgrounds
are reused ~100×; an example-level split would put the same background on both sides of the test
boundary. Every wav still contributes to all three splits, and split fractions apply to segments,
so example counts are approximate.

**AR score** (lower = better, 0 = perfect) — five weighted terms, each capped at 1:
`w_det·(1−F1) + w_count·(1−exact-count) + w_type·(1−type-acc) + w_time·(time-MAE/0.5 s) +
w_freq·(freq-MAE/2500 Hz)`. Computed from *free-running generation*, because that is how the
model is deployed; a teacher-forced loss can rank models differently. Weights and MAE scales
live in `CONFIG` (`ar_weights`, `ar_time_scale`, `ar_freq_scale`) and are stored in every
checkpoint; raise `count` if selection favors over-emitting models. The scales mean "an MAE this
large = one full unit of error" — the old full-range normalization (5 s / 25 kHz) made regression
nearly invisible, so **AR values from before this change are not comparable**. Used for
checkpoint selection, tuning, reruns, and ablations alike; teacher-forced losses are logged next
to it so the correlation can be checked.

**Two stages.** The network finds pulses and localises them in time; the refinement stage measures
their frequencies. This split is deliberate: learning handles what it is good at (finding events
in noise, handling a variable number of them), and classical estimation handles what it is good at
(measuring parameters precisely once the signal is known to be there). Established passive-acoustic
software is built the same way, with an energy detector feeding a contour stage; here a learned
detector replaces the energy detector.

**Constants** (`data_config.py`): `FS` 50000 · `FREQ_MAX` 25000 · `TIME_MAX` 5.0 ·
`PULSE_F_MIN/MAX` 1000/24990 · `PULSE_BW_MIN/MAX` 100/4000 · `SNR_MIN/MAX` −15/40 ·
`PATCH_DELTA_MIN/MAX` 15/30 · `NPERSEG/NOVERLAP` 2048/1024 (Δf ≈ 24.4 Hz, Δt ≈ 20.5 ms).

---

## Changelog (this revision)

Segment-disjoint split (#2) · downward sweeps (#7) · pulse-coupled source-noise patch (#3-mod) ·
`false_alarm_sweep.py` (#6) · reweighted AR score (configurable `ar_weights` + tightened MAE
scales; old AR values not comparable) · per-epoch validation component log (`val_components.csv`)
· memory-safe detection plotting · contour refinement of detected pulses (`refine_detections.py`, with per-pulse threshold selection and single-frequency CW reporting) · consistent figure style across all scripts (`plot_style.py`). Earlier revision: post-injection normalization, AR-score selection,
frequency-gated matching/merging, SNR U(−15,40), seeded generation, val split, test-only
analyses, dual detection confidences, upsampling guard. **Old datasets and checkpoints are not
comparable — regenerate and retrain.**
