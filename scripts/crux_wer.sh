#!/usr/bin/env bash
# crux_wer.sh — CRUX word-error-rate check: whisper.apr vs a pinned whisper.cpp,
# greedy to greedy, on a sha256-pinned clip set (whisper.apr #89, rows R2/R3/R4).
#
# Usage:
#   scripts/crux_wer.sh --apr <whisper-apr bin> --apr-model <.apr> \
#       --cpp <whisper-cli bin> --cpp-model <ggml .bin> --out <dir> [--clips crux/clips.tsv]
#
# Both engines decode GREEDY with no fallback, no timestamps, language forced
# to English, so the comparison is like for like:
#   whisper.cpp : -bs 1 -bo 1 -nf -nt -l en
#   whisper-apr : --beam-size 1 --best-of 1 --temperature 0 --no-fallback --no-timestamps -l en -o txt
# (whisper-apr's own `parity` subcommand runs whisper.cpp with its DEFAULT
# decoding and keeps timestamped stdout; this harness does neither.)
#
# ORACLE SELF-CHECK: whisper.cpp runs twice per clip; the two transcripts must
# be identical (WER 0). If not, the comparator is not deterministic and no
# apr number from this run means anything, so the run is RED.
#
# Exit: 0 every clip measured and the oracle self-check held (the WER numbers
#       are the receipt; judging them against a bar is the release gate's job)
#       1 oracle self-check failed on some clip
#       2 refused: missing binary/model, bad argument, or a clip sha256 mismatch
set -euo pipefail

clips="crux/clips.tsv"; apr=""; apr_model=""; cpp=""; cpp_model=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --clips) clips="$2"; shift 2 ;;
    --apr) apr="$2"; shift 2 ;;
    --apr-model) apr_model="$2"; shift 2 ;;
    --cpp) cpp="$2"; shift 2 ;;
    --cpp-model) cpp_model="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "crux_wer: unknown argument: $1" >&2; exit 2 ;;
  esac
done
for v in apr apr_model cpp cpp_model out; do
  if [ -z "${!v}" ]; then echo "crux_wer: --${v//_/-} is required" >&2; exit 2; fi
done
for f in "$apr" "$cpp"; do [ -x "$f" ] || { echo "crux_wer: not executable: $f" >&2; exit 2; }; done
for f in "$apr_model" "$cpp_model" "$clips"; do [ -f "$f" ] || { echo "crux_wer: missing: $f" >&2; exit 2; }; done

case "$out" in *..*) echo "crux_wer: --out must not contain '..': $out" >&2; exit 2 ;; esac
here=$(cd "$(dirname "$0")/.." && pwd)
wer_py="$here/crux/wer.py"
mkdir -p "$out/text"
sha() { sha256sum "$1" | cut -d' ' -f1; }

apr_version=$("$apr" --version 2>/dev/null | head -1 || echo unknown)
cpp_commit=$(git -C "$(dirname "$cpp")/../.." rev-parse HEAD 2>/dev/null || echo unknown)
rows="$out/rows.jsonl"; : > "$rows"
oracle_bad=0

while IFS=$'\t' read -r id path want; do
  case "$id" in ''|'#'*) continue ;; esac
  [ -f "$here/$path" ] || { echo "crux_wer: clip missing: $path" >&2; exit 2; }
  got=$(sha "$here/$path")
  if [ "$got" != "$want" ]; then
    echo "crux_wer: REFUSED $id: sha256 $got != manifest $want" >&2; exit 2
  fi
  c1="$out/text/$id.cpp1.txt"; c2="$out/text/$id.cpp2.txt"; a="$out/text/$id.apr.txt"
  "$cpp" -m "$cpp_model" -f "$here/$path" -bs 1 -bo 1 -nf -nt -l en --no-prints > "$c1" 2>/dev/null
  "$cpp" -m "$cpp_model" -f "$here/$path" -bs 1 -bo 1 -nf -nt -l en --no-prints > "$c2" 2>/dev/null
  "$apr" transcribe -f "$here/$path" --model-path "$apr_model" --beam-size 1 --best-of 1 \
    --temperature 0 --no-fallback --no-timestamps -l en -o txt --no-gpu > "$a" 2>/dev/null
  oracle=$(python3 "$wer_py" "$c1" "$c2")
  w=$(python3 "$wer_py" "$c1" "$a")
  if [ "$oracle" != "0.000000" ]; then oracle_bad=1; fi
  jq -cn --arg id "$id" --arg path "$path" --arg sha "$got" --rawfile cpp_text "$c1" \
     --rawfile apr_text "$a" --argjson wer "$w" --argjson oracle_wer "$oracle" \
     '{clip:$id, path:$path, sha256:$sha, wer_apr_vs_cpp:$wer, oracle_cpp_vs_cpp:$oracle_wer,
       whisper_cpp:($cpp_text|gsub("^\\s+|\\s+$";"")), whisper_apr:($apr_text|gsub("^\\s+|\\s+$";""))}' >> "$rows"
  printf '%-18s WER %s   oracle %s\n' "$id" "$w" "$oracle"
done < "$here/$clips"

# measured_at is the receipt's provenance (when this run measured), not a build input.
measured_at=$(date -u +%FT%TZ) # bashrs disable-line=DET002
jq -s --arg apr_version "$apr_version" --arg apr_model_sha "$(sha "$apr_model")" \
   --arg cpp_commit "$cpp_commit" --arg cpp_model_sha "$(sha "$cpp_model")" \
   --arg host "$(hostname)" --arg when "$measured_at" \
   '{schema:"whisper-apr-crux-wer/v1", host:$host, measured_at:$when,
     engines:{whisper_apr:{version:$apr_version, model_sha256:$apr_model_sha,
              decode:"--beam-size 1 --best-of 1 --temperature 0 --no-fallback --no-timestamps -l en --no-gpu"},
              whisper_cpp:{commit:$cpp_commit, model_sha256:$cpp_model_sha,
              decode:"-bs 1 -bo 1 -nf -nt -l en"}},
     clips:.}' "$rows" > "$out/receipt.json"
echo "receipt: $out/receipt.json"
if [ "$oracle_bad" -ne 0 ]; then
  echo "crux_wer: RED: the whisper.cpp self-check was not 0 on some clip; the comparator is not deterministic" >&2
  exit 1
fi
