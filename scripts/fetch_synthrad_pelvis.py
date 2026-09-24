"""按需精准下载：只取 SynthRAD2023 Task1.zip 中指定盆腔患者的 mr/ct/mask。

原理：HTTP Range 拉 zip 尾部中央目录 → 解析成员偏移 → 只下载需要的成员字节。
用法: HTTPS_PROXY=http://127.0.0.1:7890 python scripts/fetch_synthrad_pelvis.py [--patients 10]
"""
import argparse
import io
import os
import struct
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/api/records/7260705/files/Task1.zip/content"
TOTAL = 14471900926
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthrad_pelvis")


def fetch_range(start: int, end: int, retries: int = 8) -> bytes:
    """分块(4MB)+断点续传式拉取 [start,end]，连接中断不丢已得字节。"""
    out = bytearray()
    pos = start
    while pos <= end:
        chunk_end = min(pos + 4 * 1024 * 1024 - 1, end)
        req = urllib.request.Request(URL, headers={"Range": f"bytes={pos}-{chunk_end}"})
        data = None
        for i in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:
                print(f"    range {pos}-{chunk_end} 重试{i+1}: {e}", flush=True)
        if not data:
            raise RuntimeError(f"range {pos}-{chunk_end} 下载失败")
        out += data
        pos += len(data)
    return bytes(out)


def parse_central_directory(data: bytes):
    """解析中央目录字节流 → {name: (method, comp_size, uncomp_size, local_off)}"""
    entries = {}
    p = 0
    while p + 46 <= len(data):
        sig = data[p:p + 4]
        if sig != b"PK\x01\x02":
            break
        method = struct.unpack("<H", data[p + 10:p + 12])[0]
        comp = struct.unpack("<I", data[p + 20:p + 24])[0]
        uncomp = struct.unpack("<I", data[p + 24:p + 28])[0]
        fnlen = struct.unpack("<H", data[p + 28:p + 30])[0]
        extralen = struct.unpack("<H", data[p + 30:p + 32])[0]
        cmntlen = struct.unpack("<H", data[p + 32:p + 34])[0]
        loff = struct.unpack("<I", data[p + 42:p + 46])[0]
        extra = data[p + 46 + fnlen: p + 46 + fnlen + extralen]
        # ZIP64：按序补齐 0xFFFFFFFF 的字段（uncomp, comp, loff）
        if 0xFFFFFFFF in (comp, uncomp, loff):
            q = 0
            vals = []
            while q + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[q:q + 4])
                body = extra[q + 4: q + 4 + sz]
                if tag == 0x0001:
                    for k in range(sz // 8):
                        vals.append(struct.unpack("<Q", body[k * 8:(k + 1) * 8])[0])
                    break
                q += 4 + sz
            it = iter(vals)
            if uncomp == 0xFFFFFFFF:
                uncomp = next(it, uncomp)
            if comp == 0xFFFFFFFF:
                comp = next(it, comp)
            if loff == 0xFFFFFFFF:
                loff = next(it, loff)
        name = data[p + 46: p + 46 + fnlen].decode("utf-8")
        entries[name] = (method, comp, uncomp, loff)
        p += 46 + fnlen + extralen + cmntlen
    return entries


def fetch_member(name, meta, out_path):
    method, comp, uncomp, loff = meta
    head = fetch_range(loff, loff + 29)
    assert head[:4] == b"PK\x03\x04", f"{name} 本地头不符"
    fnlen = struct.unpack("<H", head[26:28])[0]
    extralen = struct.unpack("<H", head[28:30])[0]
    dstart = loff + 30 + fnlen + extralen
    raw = fetch_range(dstart, dstart + comp - 1)
    if method == 0:
        data = raw
    elif method == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"{name} 未知压缩方法 {method}")
    assert len(data) == uncomp, f"{name} 解压后大小不符 {len(data)} vs {uncomp}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"  OK {name} -> {out_path} ({uncomp/1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=10)
    args = ap.parse_args()

    # 1) 拉尾部，定位 EOCD 与中央目录
    tail = fetch_range(TOTAL - 4 * 1024 * 1024, TOTAL - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    assert eocd >= 0, "未找到 EOCD"
    cd_size = struct.unpack("<I", tail[eocd + 12:eocd + 16])[0]
    cd_off = struct.unpack("<I", tail[eocd + 16:eocd + 20])[0]
    if cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:   # ZIP64 EOCD
        z64 = tail.rfind(b"PK\x06\x06")
        assert z64 >= 0, "未找到 ZIP64 EOCD"
        cd_size = struct.unpack("<Q", tail[z64 + 40:z64 + 48])[0]
        cd_off = struct.unpack("<Q", tail[z64 + 48:z64 + 56])[0]
    print(f"中央目录: offset={cd_off} size={cd_size/1e6:.2f} MB", flush=True)

    cd_start_in_tail = cd_off - (TOTAL - len(tail))
    if cd_start_in_tail >= 0:
        cd = tail[cd_start_in_tail: cd_start_in_tail + cd_size]
    else:
        cd = fetch_range(cd_off, cd_off + cd_size - 1)
    entries = parse_central_directory(cd)
    print(f"成员数: {len(entries)}", flush=True)

    # 2) 选盆腔患者（跨中心均匀取）
    pids = sorted({n.split("/")[2] for n in entries
                   if n.startswith("Task1/pelvis/1P") and n.count("/") >= 3})
    print(f"盆腔患者总数: {len(pids)}")
    step = max(1, len(pids) // args.patients)
    chosen = pids[::step][: args.patients]
    print("选取:", chosen)

    # 3) 逐成员按需下载（3 线程）
    jobs = []
    for pid in chosen:
        for fn in ("mr.nii.gz", "ct.nii.gz", "mask.nii.gz"):
            name = f"Task1/pelvis/{pid}/{fn}"
            if name in entries:
                jobs.append((name, entries[name],
                             os.path.join(OUT, pid, fn)))
    todo = [(n, m, o) for n, m, o in jobs if not (os.path.exists(o) and os.path.getsize(o) == m[2])]
    print(f"待下载成员: {len(todo)} / {len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda j: fetch_member(*j), todo))
    print("SYNTHRAD_PELVIS_FETCH_DONE")


if __name__ == "__main__":
    main()
