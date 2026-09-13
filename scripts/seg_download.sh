#!/bin/bash
# 分段并行下载器 v2：支持重定向(-L)、逐段校验、跨运行幂等续传
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
  for try in $(seq 1 8); do
    local got=0
    [ -f "$part" ] && got=$(stat -c%s "$part")
    if [ "$got" -eq "$want" ]; then return 0; fi
    if [ "$got" -gt "$want" ]; then rm -f "$part"; got=0; fi
    local code exitc
    code=$(curl -sL -H "Range: bytes=${start}-${end}" --max-time 600 \
      --speed-time 30 --speed-limit 1024 -o "$part.tmp" -w "%{http_code}" "$URL")
    exitc=$?
    if [ $exitc -eq 0 ] && [ -f "$part.tmp" ]; then
      local sz; sz=$(stat -c%s "$part.tmp")
      if [ "$sz" -eq "$want" ]; then mv "$part.tmp" "$part"; return 0; fi
      # 服务器忽略Range返回整文件时 sz > want；重定向页则 sz < want
      rm -f "$part.tmp"
    fi
    sleep $(( try * 2 ))
  done
  echo "  part $i FAILED (curl exit=$exitc http=$code)" >&2
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
