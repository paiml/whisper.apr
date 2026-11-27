# Diarization fixtures (VoxConverse dev excerpts)

Three 60 s excerpts for the future DER snapshot (whisper.apr#123, waits on aprender#6523).
No scorer lives here, and no model weights are distributed.

| file | source recording | speakers in excerpt |
| --- | --- | --- |
| `kbkon-60s.wav` / `.rttm` | VoxConverse dev `kbkon` | 6 |
| `wjhgf-60s.wav` / `.rttm` | VoxConverse dev `wjhgf` | 4 |
| `zfkap-60s.wav` / `.rttm` | VoxConverse dev `zfkap` | 3 |

Each `.wav` is the first 60 s of the dev recording, resampled to 16 kHz mono 16-bit PCM
(`ffmpeg -t 60 -ar 16000 -ac 1`). Each `.rttm` is the reference annotation from
`joonson/voxconverse` (master, `dev/<id>.rttm`) with turns cut at 60 s (turns that cross
60 s are truncated to end there).

## Attribution

VoxConverse: J. S. Chung, J. Huh, A. Nagrani, T. Afouras, A. Zisserman,
"Spot the conversation: speaker diarisation in the wild", Interspeech 2020.
Dataset: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/ ,
annotations: https://github.com/joonson/voxconverse .
Licensed under Creative Commons Attribution 4.0 International
(https://creativecommons.org/licenses/by/4.0). The copyright in the underlying audio
remains with the original owners of the video. Modified: trimmed, resampled, downmixed.

## Integrity

`SHA256SUMS` lists the digest of every file here, checked by
`tests/diarization_fixture_manifest.rs`.
