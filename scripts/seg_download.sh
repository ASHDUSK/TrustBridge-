#!/bin/bash
# 分段并行下载器 v3：支持重定向(-L)、逐段校验、跨运行幂等续传、段内断点续传（抗断流）
# 用法: seg_download.sh <URL> <TOTAL_SIZE> <OUT_FILE> [NSEG]
set -u
URL="$1"; TOTAL="$2"; OUT="$3"; NSEG="${4:-16}"
# 成品已存在且大小正确 → 跳过（幂等）
if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -eq "$TOTAL" ]; then
  echo "SKIP $OUT (already complete)"; exit 0
fi
TMP="${OUT}.parts"
mkdir -p "$TMP"
SEG=$(( (TOTAL + NSEG - 1) / NSEG ))

download_part() {
  local i=$1 start=$2 end=$3
  local want=$(( end - start + 1 ))
  local part="$TMP/p$(printf '%04d' $i)"
  for try in $(seq 1 30); do
    local got=0
    [ -f "$part" ] && mv "$part" "$part.tmp"
    [ -f "$part.tmp" ] && got=$(stat -c%s "$part.tmp")
    if [ "$got" -eq "$want" ]; then mv "$part.tmp" "$part"; return 0; fi
    if [ "$got" -gt "$want" ]; then rm -f "$part.tmp"; got=0; fi
    # 段内断点续传：只请求剩余字节并追加（连接中断不丢进度）
    curl -sL --ssl-no-revoke -H "Range: bytes=$(( start + got ))-${end}" --max-time 900 \
      --speed-time 30 --speed-limit 512 -o "$part.tmp.new" "$URL"
    if [ -f "$part.tmp.new" ]; then cat "$part.tmp.new" >> "$part.tmp"; rm -f "$part.tmp.new"; fi
    sleep $(( try > 10 ? 10 : try ))
  done
  echo "  part $i FAILED" >&2
  return 1
}

pids=(); FAIL=0
for i in $(seq 0 $((NSEG-1))); do
  start=$(( i * SEG )); end=$(( start + SEG - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  [ $start -gt $end ] && break
  download_part "$i" "$start" "$end" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || FAIL=1; done

TOTAL_GOT=$(cat "$TMP"/p[0-9][0-9][0-9][0-9] 2>/dev/null | wc -c)
if [ "$FAIL" -eq 0 ] && [ "$TOTAL_GOT" -eq "$TOTAL" ]; then
  cat "$TMP"/p* > "$OUT"
  rm -rf "$TMP"
  echo "OK $OUT $(stat -c%s "$OUT")"
else
  echo "FAIL expected=$TOTAL got=$TOTAL_GOT (parts kept in $TMP)"
  exit 1
fi
