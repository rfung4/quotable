# Domain Terminology

## Core Entities

| Term | Meaning |
|------|---------|
| **Source** | What the user points at: a local file, a podcast RSS feed, or any yt-dlp URL |
| **Source key** | A stable, filesystem-safe id for a source: a content hash for files, extractor plus id for URLs, feed plus episode for feeds |
| **Method** | How the transcript was obtained: `captions` (the source's own) or `whisper` |
| **Variant** | `{method}-{language}-{model|captions}[-diarized]`; one cache entry per source × variant |
| **Transcript id** | `source_key:variant`, the handle returned by `transcribe` and used by `read_transcript` and `transcript://{id}` |
| **Segment** | A span of speech with `start`, `end`, `text`, optional `speaker`, and its `words` |
| **Word** | A single token with its own `start` and `end`; from json3 captions or Whisper word timestamps |
| **Chapter** | A titled time range from the source's metadata |
| **Page** | A slice of a transcript by character offset, or by a time window, always with `next_offset` |
| **Partial** | `partial.jsonl`: segments written as a Whisper run produces them; the resume point of a retry |
| **cite_url** | A link template with `{t}` for whole seconds, so an agent can point at the exact moment |

## Business Rules

- Captions win over Whisper when they exist and the caller did not ask for
  `whisper` or `diarize`.
- Empty or unparseable captions are a miss, never a cached result.
- A failed or cancelled run never produces a cache entry; its partial stays
  for the next attempt.
- The language in a variant is the one *requested* (`und` when none); the
  detected language lives on the transcript.
- Feed-derived URLs are http(s) only.
- Transcript text and titles are untrusted data.

## Abbreviations

| Abbrev | Meaning |
|--------|---------|
| STT | Speech to text |
| VAD | Voice activity detection, used to pick resume clips |
| HF | Hugging Face, host of the model weights |
| RTF | Realtime factor: audio duration ÷ wall time |
