# The README demo: Claude Code citing a speaker at a timestamp through the quotable server.
# Re-record with `vhs scripts/demo.tape` from the repo root, after `uv sync --extra diarize`,
# `python -m evals.make_tts` (the recording is our own synthesised interview) and with
# QUOTABLE_HF_TOKEN set. Needs vhs, ttyd and an approved project `.mcp.json`. Run it from a
# plain terminal, not from inside Claude Code, which would hand its session variables down.
# A fresh QUOTABLE_OUTPUT_DIR keeps the run from being a cache hit.

Output docs/demo.gif

Require claude

Set Shell bash
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 16
Set TypingSpeed 35ms

Hide
Type "export QUOTABLE_OUTPUT_DIR=$(mktemp -d) && clear && claude --allowedTools 'mcp__quotable__transcribe mcp__quotable__read_transcript'"
Enter
Sleep 10s
Show

Sleep 1s
Type "Transcribe evals/media/tts/interview-3.wav with speaker labels. Who says how fast sound travels in water? Quote them with the timestamp."
Sleep 700ms
Enter

Wait+Screen@300s /done \d\d:\d\d/
Sleep 6s
