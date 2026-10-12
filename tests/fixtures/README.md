# Test fixtures

## `jfk.flac`

11 s from John F. Kennedy's inaugural address (20 January 1961): "…ask not what your
country can do for you, ask what you can do for your country."

- **Source:** `samples/jfk.wav` from [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp),
  fetched 2026-10-12, sha256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
- **Rights:** a speech made by the President in his official capacity, a work of the US
  federal government, so it is in the public domain in the United States (17 U.S.C. § 105).
- **Re-encoding:** `ffmpeg -i jfk.wav -ac 1 -ar 16000 -map_metadata -1 jfk.flac` (ffmpeg 4.4.2),
  lossless; sha256 `835359f264539bc3ca6ae4c2cdb3cede4929a1efeacc6c15ea0d36caa9729f8c`.

## Hand-written files

`manual.vtt`, `rolling_autosub.vtt`, `autosub.json3`, `feed.xml` and
`episode_transcript.srt` were written for these tests and contain no third-party content.
`rolling_autosub.vtt` mirrors the layout of YouTube's automatic WebVTT, including lines that
hold a single space. Editors that strip trailing whitespace break it, and
`test_rolling_autosub_vtt_collapses_repeats` will say so.
