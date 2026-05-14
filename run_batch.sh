#!/bin/bash
# Run pipeline on all 8 parts sequentially, 5 reels each.
# Usage: bash run_batch.sh

cd "/Users/loukiknaik/projects/agentic video editor"
CONFIG="/tmp/ave_batch.yaml"
TOTAL_REELS=0
FAILED=0

for i in $(seq 1 8); do
  INPUT="samples/RiHi4solYHE_part${i}.mp4"
  if [ ! -f "$INPUT" ]; then
    echo "⚠ Part $i not found: $INPUT — skipping"
    ((FAILED++))
    continue
  fi
  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  PART $i / 8 — $(basename $INPUT)"
  echo "════════════════════════════════════════════════════"
  .venv/bin/python main.py "$INPUT" -c "$CONFIG" --output-dir outputs
  REEL_COUNT=$(ls outputs/$(ls -t outputs/ | head -1)/*.mp4 2>/dev/null | wc -l | tr -d ' ')
  TOTAL_REELS=$((TOTAL_REELS + REEL_COUNT))
  echo "  → $REEL_COUNT reels from part $i (total so far: $TOTAL_REELS)"
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  BATCH COMPLETE"
echo "  Total reels: $TOTAL_REELS"
echo "  Failed parts: $FAILED"
echo "  Output dirs:"
ls -d outputs/2026-04-*/ 2>/dev/null | tail -8
echo "════════════════════════════════════════════════════"
