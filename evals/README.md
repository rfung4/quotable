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
```

In order: generate the local media and validate the manifest against the live sources;
synthesise the TTS dialogues (each turn is cached, so a re-run never re-bills); download
each URL and feed entry's audio once into `evals/media/cache/` and write the `large-v3`
references; run the matrix. Each has `--help`; the first three have a `--check` or
`--validate` that only verifies what exists.

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
