# -*- coding: utf-8 -*-
"""Gate 2 · 全量 maps 重跑（配置指纹 + manifest + --resume）。

每组一次推理存全部中间量（npz，float16 压缩）：
  pred / gt / mask / rcrf_raw / unc_raw / step_sums[K0=10 全步捕获] / ts / meta

组注册表（--group）：
  ixi_test_t1t2 / ixi_test_t2t1      IXI 测试 60 片×2（self ckpt，128²）
  ixi_val_t1t2  / ixi_val_t2t1       IXI 验证抽样 60 片×2（权重扫描专用）
  synthrad_t1ct / synthrad_t2ct      SynthRAD 双模态输入任务（self ckpt，val+test 全量）
  tumor_brats                        BraTS 含瘤 60 片（官方 ckpt，256²）
  crossinst_B / crossinst_C          跨机构两组（官方 ckpt，256²）
  collapse_official                  IXI 测试 20 片（官方 T1→T2 权重，256²）

用法:
  python scripts/eval_maps_full.py --group ixi_test_t1t2 --pilot      # Gate1 预跑
  python scripts/eval_maps_full.py --group ixi_test_t1t2              # 全量（--resume 自动）
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.sampler import TrustSampler  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "results", "maps_full")
SEED = 2026
N_ENS = 4
SKIP = 1
K_CAPTURE = 10          # 捕获全部桥步（K 消融离线做子集）

CKPTS = {
    "self_t1t2": os.path.join(ROOT, "checkpoints", "self_t1t2.ckpt"),
    "self_t2t1": os.path.join(ROOT, "checkpoints", "self_t2t1.ckpt"),
    "self_ct_t12ct": os.path.join(ROOT, "checkpoints", "self_ct_t12ct.ckpt"),
    "self_ct_t22ct": os.path.join(ROOT, "checkpoints", "self_ct_t22ct.ckpt"),
    "official_t1t2": os.path.join(ROOT, "checkpoints", "ixi_t1_t2.ckpt"),
    "official_brats": os.path.join(ROOT, "checkpoints", "brats_flair_t2.ckpt"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 数据加载
def load_ixi(split, direction, n):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == split]
    step = max(1, len(rows) // n)
    rows = rows[::step][:n]
    src_key, gt_key = (("t2_path", "t1_path") if direction == "t2t1"
                       else ("t1_path", "t2_path"))
    items = []
    for r in rows:
        src = np.load(r[src_key]).astype(np.float32)
        gt = np.load(r[gt_key]).astype(np.float32)
        items.append({"id": r["subject"], "z": int(r["z"]), "src": src, "gt": gt,
                      "size": src.shape})
    return items


def load_synthrad(direction):
    mf = ("manifest_ct_a.csv" if direction == "t1ct" else "manifest_ct_c.csv")
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "synthrad_slices_128",
                                                mf), encoding="utf-8")))
    items = []
    for r in rows:                      # val+test 全量（患者数受限如实报告）
        src = np.load(r["mr_path"]).astype(np.float32)
        gt = np.load(r["ct_path"]).astype(np.float32)
        items.append({"id": r["patient"], "z": int(r["z"]), "src": src, "gt": gt,
                      "size": src.shape})
    return items


def load_tumor(n=60, seed=2026):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "brats2018_slices",
                                                "manifest.csv"), encoding="utf-8")))
    man_t = [r for r in man if float(r["tumor_frac"]) > 0.005]
    step = max(1, len(man_t) // n)
    sel = man_t[::step][:n]
    items = []
    for r in sel:
        flair = np.load(r["flair_path"]).astype(np.float32)
        t2 = np.load(r["t2_path"]).astype(np.float32)
        items.append({"id": r["patient"], "z": int(r["z"]), "src": flair, "gt": t2,
                      "size": flair.shape})
    return items


def load_crossinst(set_name, n_subj=6, per_subj=20):
    base = ("MICCAI_BraTS_2018_Data_Training/HGG" if set_name == "B"
            else "MICCAI_BraTS_2018_Data_Validation")
    root = os.path.join(ROOT, "data", "brats2018_raw", base)
    used = {"Brats18_2013_10_1", "Brats18_2013_11_1", "Brats18_2013_12_1",
            "Brats18_2013_13_1", "Brats18_2013_14_1", "Brats18_2013_17_1"}
    import glob as _g
    if set_name == "B":
        subj = [p for p in sorted(os.path.basename(x) for x in _g.glob(os.path.join(root, "Brats18_2013_*")))
                if p not in used][:n_subj]
    else:
        subj = sorted(os.path.basename(x) for x in _g.glob(os.path.join(root, "Brats18_CBICA_*")))[:n_subj]
    items = []
    import nibabel as nib
    SL = range(27, 127)
    for p in subj:
        d = os.path.join(root, p)
        fl = _g.glob(os.path.join(d, "*_flair.nii"))[0]
        t2 = _g.glob(os.path.join(d, "*_t2.nii"))[0]
        flv = np.asarray(nib.load(fl).dataobj, np.float32)
        t2v = np.asarray(nib.load(t2).dataobj, np.float32)
        zs = list(SL)[:: max(1, len(SL) // per_subj)][:per_subj]
        for z in zs:
            if z >= min(flv.shape[2], t2v.shape[2]):
                continue
            a = np.rot90(flv[:, :, z], -1); b = np.rot90(t2v[:, :, z], -1)
            a = np.clip(a / max(float(a.max()), 1e-6), 0, 1)
            b = np.clip(b / max(float(b.max()), 1e-6), 0, 1)
            items.append({"id": p, "z": z, "src": a.astype(np.float32),
                          "gt": b.astype(np.float32), "size": a.shape})
    return items


GROUPS = {
    "ixi_test_t1t2": dict(ckpt="self_t1t2", task="T1->T2*", loader=lambda n: load_ixi("test", "t1t2", n)),
    "ixi_test_t2t1": dict(ckpt="self_t2t1", task="T2->T1*", loader=lambda n: load_ixi("test", "t2t1", n)),
    "ixi_val_t1t2": dict(ckpt="self_t1t2", task="T1->T2*", loader=lambda n: load_ixi("val", "t1t2", n)),
    "ixi_val_t2t1": dict(ckpt="self_t2t1", task="T2->T1*", loader=lambda n: load_ixi("val", "t2t1", n)),
    "synthrad_t1ct": dict(ckpt="self_ct_t12ct", task="CT_T1->CT_S", loader=lambda n: load_synthrad("t1ct")),
    "synthrad_t2ct": dict(ckpt="self_ct_t22ct", task="CT_T2->CT_S", loader=lambda n: load_synthrad("t2ct")),
    "tumor_brats": dict(ckpt="official_brats", task="BRATS_FLAIR->T2", loader=lambda n: load_tumor(n)),
    "crossinst_B": dict(ckpt="official_brats", task="BRATS_FLAIR->T2", loader=lambda n: load_crossinst("B", n_subj=n)),
    "crossinst_C": dict(ckpt="official_brats", task="BRATS_FLAIR->T2", loader=lambda n: load_crossinst("C", n_subj=n)),
    "collapse_official": dict(ckpt="official_t1t2", task="T1->T2", loader=lambda n: load_ixi("test", "t1t2", n)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--n", type=int, default=60, help="切片数（crossinst 为每组被试数）")
    ap.add_argument("--pilot", action="store_true", help="只跑前 2 片并计时")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    g = GROUPS[args.group]
    ckpt_path = CKPTS[g["ckpt"]]
    items = g["loader"](args.n)
    if args.pilot:
        items = items[:2]
    fingerprint_src = json.dumps({
        "ckpt_sha256": sha256_file(ckpt_path),
        "task": g["task"], "n_ens": N_ENS, "skip": SKIP, "K": K_CAPTURE,
        "seed": SEED, "ids": [it["id"] for it in items], "zs": [it["z"] for it in items],
    }, sort_keys=True)
    H_config = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:16]

    run_dir = os.path.join(OUT_ROOT, args.group)
    os.makedirs(run_dir, exist_ok=True)
    eng = BridgeEngine(g["task"], ckpt_path=ckpt_path)
    smp = TrustSampler(eng, store_step_residuals=True, capture_steps_override=K_CAPTURE)

    manifest = {"group": args.group, "H_config": H_config, "ckpt": g["ckpt"],
                "ckpt_sha256": fingerprint_src[:0] or sha256_file(ckpt_path)[:16],
                "planned": len(items), "done": 0, "skipped": 0, "failed": 0,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t00 = time.time()
    for i, it in enumerate(items):
        npz_path = os.path.join(run_dir, f"{it['id']}_z{it['z']:03d}.npz")
        if args.resume and os.path.exists(npz_path):
            try:
                old = np.load(npz_path, allow_pickle=True)
                if str(old["H_config"]) == H_config:
                    manifest["skipped"] += 1
                    continue
            except Exception:
                pass
        t0 = time.time()
        try:
            y = eng.preprocess(it["src"])
            res = smp.sample(y, n_ensemble=N_ENS, step_skip=SKIP, seed=SEED)
            pred = eng.postprocess(res.pred, it["src"].shape)
            tr = res.trust_map
            rcrf_raw, unc_raw = smp.rcrf_raw_map, smp.unc_raw_map
            steps = [(t, S) for t, S in smp.step_residual_maps]
            if tr.shape != it["src"].shape:
                S = tr.shape[0]; H, W = it["src"].shape
                pt_, pl_ = (S - H) // 2, (S - W) // 2
                crop = lambda m: np.ascontiguousarray(m[pt_:pt_ + H, pl_:pl_ + W])
                tr, pred = crop(tr), crop(pred)
                rcrf_raw, unc_raw = crop(rcrf_raw), crop(unc_raw)
                steps = [(t, crop(S)) for t, S in steps]
            mask = (it["src"] > 0.02)
            if mask.sum() < 200:
                manifest["failed"] += 1
                continue
            step_arr = np.zeros((K_CAPTURE,) + tr.shape, np.float16)
            ts_arr = np.zeros(K_CAPTURE, np.int16)
            for k_i, (t_k, S_k) in enumerate(steps):
                step_arr[k_i] = S_k.astype(np.float16); ts_arr[k_i] = t_k
            tmp = npz_path + ".tmp.npz"
            np.savez_compressed(
                tmp,
                pred=pred.astype(np.float16), gt=it["gt"].astype(np.float16),
                mask=mask, rcrf_raw=rcrf_raw.astype(np.float16),
                unc_raw=unc_raw.astype(np.float16),
                step_sums=step_arr, ts=ts_arr,
                H_config=H_config, id=it["id"], z=it["z"], task=g["task"],
                ckpt=g["ckpt"], K0=K_CAPTURE, skip=SKIP, seed=SEED,
                w_sampler=np.array([smp.w_residual, smp.w_uncertainty], np.float32))
            try:
                os.replace(tmp, npz_path)
            except PermissionError:
                npz_path = npz_path.replace(".npz", "_r.npz")
                os.replace(tmp, npz_path)
            manifest["done"] += 1
            if args.pilot or (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(items)}] {it['id']} z{it['z']} "
                      f"{time.time()-t0:.1f}s 片耗时 | 累计 {(time.time()-t00)/60:.1f}min",
                      flush=True)
        except Exception as e:
            import traceback
            manifest["failed"] += 1
            print(f"  FAIL {it['id']} z{it['z']}: {str(e)[:120]}", flush=True)
            traceback.print_exc()

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["gpu_minutes"] = round((time.time() - t00) / 60, 1)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("planned", "done", "skipped", "failed", "gpu_minutes")}, indent=1))
    print("->", run_dir)


if __name__ == "__main__":
    main()
