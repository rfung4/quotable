# evals

The evaluation harness for quotable: a corpus manifest with licences, reference
transcripts, and a matrix runner that measures every input kind, Whisper size and device
the README promises. It is not part of the package; `pip install quotable` installs none
of it. Media, references and results are gitignored (`evals/media/`, `evals/refs/`,
`evals/results/`).

## Running

Prerequisites: `uv sync`, the `ffmpeg` CLI (generates the local media from
`tests/fixtures/jfk.flac`), a CUDA GPU for the oracle and the `cuda` cells,
`OPENAI_API_KEY` for the TTS dialogues, and for the diarize cells `uv sync --extra diarize`
plus `QUOTABLE_HF_TOKEN` (or `HF_TOKEN`) with the pyannote terms accepted.

```bash
uv run python -m evals.build_corpus
uv run python -m evals.make_tts
uv run python -m evals.oracle
uv run python -m evals.run
uv run python -m evals.score
uv run python -m evals.scenarios
uv run python -m evals.mcp_probe
uv run python -m evals.report
```

In order: generate the local media and validate the manifest against the live sources;
synthesise the TTS dialogues (each turn is cached, so a re-run never re-bills); download
each URL and feed entry's audio once into `evals/media/cache/` and write the `large-v3`
references; run the matrix; score it; run the long-run scenarios on an idle GPU;
probe the MCP surface of the built wheel; render the tables. Each has `--help`, and each but
the runner has a `--check` or `--validate` that only verifies what exists.

## The matrix

`python -m evals.run` expands `corpus.json` into cells:

- **short** entries: Whisper × `tiny`, `base`, `small`, `turbo` × `cpu` (int8) and `cuda`
  (float16), on the local file or the cached audio
- **long** entries: Whisper `turbo` · `cuda` only
- a `captions` cell wherever the entry ships captions (YouTube) or a transcript tag (feed)
- `readme_row` entries: `base` · `cpu` as well (already in the grid for short entries)
- `diarize` entries: one `turbo` · `cuda` cell with diarization
- every URL and feed entry: one **real-path** cell, Whisper `turbo` · `cuda` on the live URL
  or feed plus pinned guid — the only cells that exercise the download

Cells run with the product's defaults: no `language`, so Whisper detects it and the
captions path takes its default track.

Options: `--matrix smoke` runs the single cell `jfk.flac|whisper|tiny|cpu|false|false`;
`--only <glob>` (repeatable) filters on `cell_id`; `--redo` re-runs cells that already have
a row; `--out <dir>` replaces `evals/results/`. A re-run skips every cell that has a row,
except `skipped` rows, which are retried because they record something that has not
happened yet.

Every cell runs in its own process, `python -m evals.run --cell <id> --out <dir> --row <file>`,
so model load, memory and a CUDA fallback are per cell. The child writes its row
to the `--row` file, never to stdout. A cell still running after 900 s plus the audio's
length (four times the length on `cpu`) is killed and recorded as a `crash`.

## Results

`<out>/cells.jsonl` holds one JSON object per line, one line per `cell_id`. Each cell also
leaves `<out>/cache/<slug>/` (its own `QUOTABLE_OUTPUT_DIR`) and `<out>/logs/<slug>.log`
(its stdout and stderr), where `<slug>` is the `cell_id` with `|` replaced by `__`.

| Key | Meaning |
|-----|---------|
| `cell_id` | `{entry}\|{mode}\|{model}\|{device}\|{diarize}\|{real}`; `-` for model and device on captions cells; flags are `true` / `false` |
| `entry` | Manifest id |
| `mode` | `whisper` or `captions` |
| `model`, `device`, `compute_type` | `null` on captions cells; `int8` on `cpu`, `float16` on `cuda` |
| `diarize` | Diarization requested |
| `real_path` | Whisper on the live URL or feed. Captions cells also reach the live source but download no audio, so they are `false` |
| `status` | `ok`, `error`, `crash`, `fail` or `skipped` (below) |
| `reason` | Short cause on every status but `ok`; on `error`, the exception type |
| `error` | `{type, message, fix}` on `error`; `{type, message, traceback}` on `crash` (the log's tail when the process died) |
| `load_s` | Whisper model load, plus the pyannote pipeline on diarize cells, timed before the call; `null` on captions cells |
| `wall_s` | The first `pipeline.transcribe(force=True)`: resolve, download (real-path cells), decode, VAD, inference, diarization, store |
| `audio_s` | The transcript's duration: decoded audio on Whisper cells, the source's declared duration on captions cells (a feed can declare it wrongly) |
| `realtime` | `audio_s / wall_s` |
| `cached_s` | The immediate second call, without `force`; on URL and feed sources it includes the metadata probe |
| `segments` | Segment count |
| `words` | Whitespace tokens across the segment texts — not word timings, which manual captions never carry |
| `progress_calls` | Progress callbacks during the first call (`0` on captions cells) |
| `first_progress_s` | Seconds from the call's start to the first progress callback; `null` without one |
| `max_rss_mb` | Peak RSS of the cell's process (`ru_maxrss` from `os.wait4`) |
| `vram_peak_mb` | Peak GPU memory of the cell's pid, sampled each second from `nvidia-smi`; `null` off `cuda` |
| `transcript_path` | The cell's `transcript.json`, relative to the results directory |

Statuses:

- `ok` — a transcript was produced and the second call came from the cache.
- `error` — the product raised a typed `QuotableError`. Silence ends here by design: the
  store refuses an empty transcript (`TranscriptionError`).
- `crash` — anything else: an untyped exception, a process that died (`exit N`,
  `signal N`) or a timeout. Always a bug in the product or the harness.
- `fail` — the cell ran but an invariant broke: `cuda_fallback` (a `cuda` cell finished on
  CPU), `not_cached` (the second call was not a cache hit), `guid_moved` (the feed served
  another episode than the pinned guid).
- `skipped` — not run: `no token` (diarize without a Hugging Face token) or `no media` (TTS
  audio not synthesised).

## Scores

`python -m evals.score` writes `<out>/scores.jsonl`, one row per cell: `cell_id`,
`cell_digest` (of the cells row it scored), the metrics, `pass` (`null` on skipped cells) and
`fails`, the names of the rules the cell broke. `--check` replays both files: it fails on an
empty `cells.jsonl`, an unscored or stale row, a crash, a skip for any reason but a missing
token, and any real-path cell that did not pass. Other failing cells are findings, not
check failures.

The reference is the TTS script for TTS entries and the `large-v3` oracle for the rest.
Tokens are lowercased with punctuation, symbols and the fillers `uh`, `um`, `hmm`, `mm`,
`mhm`, `mmm` removed (the set Whisper's own English normaliser ignores; `wer_raw` keeps
them); numbers are not mapped to words, so `1960` and `nineteen sixty` differ.

| Metric | Meaning |
|--------|---------|
| `wer`, `wer_raw` | Word error rate on normalised tokens, and on raw whitespace tokens |
| `cer` | Character error rate on the normalised text |
| `subs`, `dels`, `ins` | Edit counts behind `wer` |
| `ref_tokens`, `hyp_tokens`, `length_ratio` | Normalised token counts and their ratio |
| `drift_level`, `drift_median_s`, `drift_p95_s`, `drift_pairs` | \|Δstart\| at each reference segment's first word, where it matched: `word` level against TTS turn starts (exact), `segment` level against the oracle |
| `drift_bias_s` | Signed median Δstart; negative means the transcript starts words early |
| `drift_shifted` | The 3 h loop only: pairs the alignment matched one repeat away (identical text, equal cost), folded back into ±half a repeat before the median |
| `captions_coverage` | Whisper cells on entries with a captions or transcript-tag reference: transcript words over reference words |
| `asr_drift_*` | The same over every matched word against an auto-captioned video's own word timings (Whisper cells on `asr` entries) |
| `repeat_excess` | Share of 4-grams occurring three or more times, minus the same share in the reference |
| `language` | The transcript's language |
| `speaker_accuracy`, `turns_missed`, `ref_turn_changes`, `hyp_speakers` | Diarize cells on TTS entries: word agreement under the best label mapping, speaker changes not matched within 1 s |

| Rule | Applies to | Fails when |
|------|-----------|------------|
| `status` | every cell | `crash`, `fail` or `error`, except a no-speech error on silence or the 0.6 s clip |
| `silence` | `silence-30s` | any word comes out |
| `language` | all but `tiny` | the transcript names another language than the manifest |
| `length` | Whisper cells but `tiny` | `length_ratio` outside 0.8–1.2 |
| `repeats` | Whisper cells but `tiny` | `repeat_excess` above 0.02 |
| `wer` | Whisper on cached audio | above 0.10 (`turbo`), 0.15 (`small`), 0.20 (`base`) |
| `coverage` | Whisper on cached audio with a captions reference | `captions_coverage` below 0.8 |
| `drift` | Whisper on cached audio | median above 0.2 s (`word`) or 0.5 s (`segment`) |
| `asr_drift` | Whisper on cached audio, `asr` entries | median above 0.2 s |
| `speakers` | diarize cells on TTS entries | `speaker_accuracy` below 0.90 |

`tiny` and captions cells are reported, not judged on quality: captions against the oracle
measure the captions-first promise, which has no threshold. Real-path cells are judged on
status, length, repeats and language only, because a feed can splice different ads into each
download; their text and timing are judged on the cached-audio cells. The 0.6 s clip is
shorter than a VAD window and only has to not crash.

Thresholds were first values, revisited once after the first full run (Decision 10):

- `segment` drift 0.3 → 0.5 s. The oracle starts words early itself: `small`, within 0.1 s
  of the exact TTS turn starts, measures 0.32–0.36 s against it on TED.
- `coverage` added. The oracle runs through the product's VAD, so speech the VAD drops is
  missing from both sides and WER cannot see it; the captions reference is independent.
  It is one-sided because a feed's audio can carry ads its transcript does not.
- `word` drift stays at 0.2 s: `turbo` starts turn-initial words 0.15–0.26 s early, where
  `base` and `small` are within 0.1 s. Citations render whole seconds, so this shows in
  SRT/VTT, not in `[HH:MM:SS]` stamps.
- Fillers dropped from normalised tokens. `turbo` writes them verbatim and the oracle
  mostly does not: 151 of the 728 insertions on `pc20-269`, none in runs, which put that
  cell at 0.1004 against the 0.10 bar for style rather than error (0.088 without).

## Scenarios

`python -m evals.scenarios` (about 30 min on an idle GPU) writes `<out>/scenarios.json`, a
`pass` and the measured numbers per scenario; `--check` replays it. On `lex-497` (cached
audio) and `jfk-loop-3h`, `turbo` · `cuda`, each in its own output directory:

| Scenario | Passes when |
|----------|-------------|
| `straight` | One uninterrupted run yields segments with no CUDA fallback |
| `memory` | RSS and VRAM, sampled every 30 s during `straight`, grow no more than 10 % from the first sample after the first segment to the end |
| `cancel_resume` | Cancelled once 600 s are done: the partial file has segments, the variant lock is free, the next call starts at or after 600 s, and the text past 600 s is within WER 0.05 of `straight` (provisional, Decision 10) |
| `paging` | Offset pages join to the full text exactly; 600 s window pages join to the overlapping segments exactly, no segment is in no window, and only segments straddling a boundary are in two |

`darknet-179` adds `feed_transcript`: live feed and guid, `mode=auto`, with the enclosure
download made to raise; the transcript-tag path must be taken.

`memory` measures growth, not peak. The peak comes before the first segment (the
spectrogram of the whole file is built at once) and is the matrix row's `max_rss_mb`.

## MCP

`python -m evals.mcp_probe` builds the wheel, serves it with `uvx` (`tiny` · `cpu`, a
temporary output directory), transcribes, pages and reads the resource over stdio, and
writes `evals/results/mcp_surface.json`: tools with schemas and annotations, resource templates,
server instructions and the round trip's checks. `--command "<serve command>"` probes
another server. `--check` replays it together with `evals/results/mcp_checklist.md`, the in-client
checklist sat in Claude Code against the repo's `.mcp.json`, which must have no open item.

## Report

`python -m evals.report` writes `<out>/report.md` with the hardware line, the corpus and its
licences, a summary per model and device, one results table per input kind, captions against
the oracle, diarization, the scenarios and the MCP checklist, and pastes the same tables into
the `MERGE_START(TABLES)` block of the local report (`docs/report/quotable-eval.md`, not
tracked). `--readme` rewrites the README's Benchmark section from the manifest's `readme_row`
entries. `--check` requires the report to name every cell with current tables, the README to
have no pending benchmark row, and nothing tracked to mention the deleted benchmark script.

## Results (2026-10-15 and 16, RTX 3070)

The public-safe summary of the last full run, after the fixes it prompted; the verdict and the
bug list are in the local report.

- 211 cells, 204 pass, 7 fail, 0 skipped, 0 crashes. The 9 `error` rows are refusals by design:
  the silence clip (the store refuses an empty transcript) and the TED captions cell (no track
  is guessed when a video names no language). All 11 live-download cells pass.
- Median normalised WER against the `large-v3` oracle on recordings of 4 min and up, `cpu` /
  `cuda`: `tiny` 0.077 / 0.072, `base` 0.059 / 0.059, `small` 0.046 / 0.041, `turbo`
  0.036 / 0.047. Against the exact TTS scripts `turbo` is 0.03–0.04. Worst non-`tiny` cell:
  `base` on the Spanish TEDx talk, 0.154.
- Median realtime on the same recordings, `cpu` / `cuda`: `tiny` 46× / 85×, `base` 26× / 75×,
  `small` 9× / 45×, `turbo` 5× / 36×. Peak VRAM 364 / 460 / 1 004 / 2 412 MB by size, 4 304 MB
  with diarization. Peak RAM grows with audio length, about 1.04 MB per second above 20 min
  (11.4 GB for 3 h).
- The 7 fails are `drift` on `turbo` TTS cells: its word starts land 0.15–0.26 s early.
- Fixed after the first run: the VAD kept 147 s of the 636 s NASA Artemis radio call, so every
  model returned a quarter of the words (captions coverage 0.24, now 0.94–1.01 with the VAD
  guard); the TED captions came back in Albanian (now refused, and `auto` transcribes the
  audio); the untrusted-content notice was missing from the structured results Claude Code
  forwards (now their first field).
- Captions against the oracle: OCW manual 0.10, NASA auto 0.04 and 0.07, TEDx Spanish manual
  0.07, Darknet transcript tags 0.11 and 0.15, Podcasting 2.0 tag 0.10. The Darknet tags sit
  193 s and 121 s ahead of the downloaded audio (dynamic ad insertion).
- Diarization on the dialogues: speaker accuracy 1.000, 0.970, 0.995; the single-speaker lecture
  gets two labels.
- Scenarios 9 of 9: Lex 3 h in 311 s, the loop in 594 s, memory growth within 3 %, resume after
  a cancel with 0 and 1 words lost, paging exact.
- MCP: the wheel round trip passes, the notice included. In Claude Code, `@quotable:transcript://…`
  does not attach, and calls over 120 s move to a background task.
