#!/bin/bash
# Fetch every keepfrac_len tag (Beaker kl<L> from S3; quick reads q32k/q64k from mooney /data via /net, inspection-
# size JSON copies only) into one folder and score it.   bash score_all.sh [dest]   (login node, ~1 min)
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:-/tmp/claude-3018/keepfrac_len_results}
mkdir -p "$DEST"
for t in kl2k kl4k kl8k kl16k kl32k kl64k; do bash "$HERE/../fetch_beaker_results.sh" $t "$DEST" >/dev/null 2>&1 || true; done
for t in q32k q64k; do mkdir -p "$DEST/$t"; cp /net/mooney/data/prasann/early_loss_proxy/results/$t/*.json "$DEST/$t/" 2>/dev/null || true; done
python3 "$HERE/analyze_kl.py" --root "$DEST" --lens 2k,4k,8k,16k,32k,64k --ref dft --out "$DEST/kl_results.json" > "$DEST/analyze_kl.txt"
python3 "$HERE/summarize_kl.py" --scored "$DEST/kl_results.json" --out-json "$DEST/kl_summary.json" --png "$DEST/keepfrac_vs_length.png" > "$DEST/summary.txt"
for L in 32k 64k; do python3 "$HERE/analyze_kl.py" --root "$DEST" --lens $L --tag-prefix q --ref dft --out "$DEST/q${L}_scored.json" > "$DEST/q${L}.txt" || true; done
echo "-> $DEST (kl_results.json, kl_summary.json, summary.txt, keepfrac_vs_length.png)"
