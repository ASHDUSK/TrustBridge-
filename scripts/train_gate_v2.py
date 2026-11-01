"""门控 v2 训练与验证：6 类模态/解剖分类器 + Mahalanobis OOD 拒判。

类别（6）: brain_T1 / brain_T2 / brain_FLAIR / pelvis_T1w / pelvis_T2w / pelvis_CT
特征: 图像统一重采样 128² → 体区标准化分位数+梯度（trustbridge.gate.gate_features）
分类: 多项 Logistic Regression（标准化空间，导出纯 numpy 可用形式）
OOD:  Mahalanobis 到各类中心的最小距离；τ = eval 集 99.5 分位（0.5% 误拒率），
      并验证合成探针（噪声/平坦/棋盘/平滑斑）全部超阈。

用法:
  python scripts/train_gate_v2.py train
  python scripts/train_gate_v2.py eval
"""
import csv
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.gate import gate_features  # noqa: E402

CLASSES = ["brain_T1", "brain_T2", "brain_FLAIR", "pelvis_T1w", "pelvis_T2w", "pelvis_CT"]
OUT_NPZ = os.path.join(ROOT, "trustbridge", "modality_gate_v2.npz")
RESIZE = (128, 128)


def load_img(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        a = np.load(path).astype(np.float32)
    else:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    hi = max(float(a.max()), 1e-6)
    if hi > 1.5:
        a = a / hi
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize(RESIZE, Image.BILINEAR), dtype=np.float32) / 255.0


def collect():
    d = {c: [] for c in CLASSES}

    def add(cls, path, tag):
        try:
            d[cls].append((load_img(path), tag))
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")

    for split in ("train", "val", "test"):
        tag = "fit" if split in ("train", "val") else "eval"
        for m, cls in (("T1", "brain_T1"), ("T2", "brain_T2")):
            for p in sorted(glob.glob(os.path.join(ROOT, "data", "selfrdb_format_v3",
                                                   m, split, "slice_*.npy"))):
                add(cls, p, tag)

    fl = sorted(glob.glob(os.path.join(ROOT, "data", "brats2018_slices", "FLAIR", "*.npy")))
    for i, p in enumerate(fl):
        add("brain_FLAIR", p, "fit" if (i // 80) % 2 == 0 else "eval")

    fets = sorted(p for p in glob.glob(os.path.join(ROOT, "data", "brats_fets_slices", "*.npy"))
                  if "label" not in os.path.basename(p))
    for i, p in enumerate(fets):
        add("brain_FLAIR", p, "fit" if i % 2 == 0 else "eval")

    mrA = sorted(glob.glob(os.path.join(ROOT, "data", "synthrad_pelvis_slices", "1PA*_mr.npy")))
    mrC = sorted(glob.glob(os.path.join(ROOT, "data", "synthrad_pelvis_slices", "1PC*_mr.npy")))
    ctA = sorted(glob.glob(os.path.join(ROOT, "data", "synthrad_pelvis_slices", "1PA*_ct.npy")))
    for i, p in enumerate(mrA):
        add("pelvis_T1w", p, "fit" if (i // 80) < 6 else "eval")
    for i, p in enumerate(mrC):
        add("pelvis_T2w", p, "fit" if (i // 80) < 2 else "eval")
    for p in ctA:
        add("pelvis_CT", p, "fit")
    return d


def featurize(imgs):
    X, keep = [], []
    for img, tag in imgs:
        f = gate_features(img)
        if f is not None:
            X.append(f)
            keep.append((img, tag))
    return np.array(X), keep


def ood_probes(n_per=40, seed=0):
    rng = np.random.RandomState(seed)
    probes = []
    probes += [("noise", rng.rand(128, 128).astype(np.float32)) for _ in range(n_per)]
    flats = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    probes += [("flat", np.full((128, 128), v, np.float32))
               for v in flats for _ in range(max(1, n_per // len(flats)))]
    cb = np.kron([[((i + j) % 2) for j in range(16)] for i in range(16)],
                 np.ones((8, 8))).astype(np.float32)
    probes += [("checker", cb) for _ in range(max(1, n_per // 4))]
    for freq in (2.0, 4.0, 8.0):
        yy, xx = np.mgrid[0:128, 0:128]
        blob = 0.5 + 0.4 * np.sin(2 * np.pi * xx / (128 / freq)) * np.cos(2 * np.pi * yy / (128 / freq))
        probes += [("smooth_blobs", blob.astype(np.float32)) for _ in range(max(1, n_per // 6))]
    return probes


def train():
    print("1) 收集切片...")
    data = collect()
    for c in CLASSES:
        print(f"  {c}: {len(data[c])} 片")

    print("2) 特征提取（统一 128² 重采样）...")
    split_X = {"fit": [], "eval": []}
    split_y = {"fit": [], "eval": []}
    for ci, c in enumerate(CLASSES):
        X, kept = featurize(data[c])
        for (img, tag), x in zip(kept, X):
            split_X.setdefault(tag, []).append(x)
            split_y.setdefault(tag, []).append(ci)
    Xfit = np.array(split_X["fit"])
    yfit = np.array(split_y["fit"])
    Xev = np.array(split_X["eval"])
    yev = np.array(split_y["eval"])
    print(f"  fit {Xfit.shape} | eval {Xev.shape}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xfit)
    Zfit = scaler.transform(Xfit)
    Zev = scaler.transform(Xev)

    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Zfit, yfit)
    acc_fit = float((lr.predict(Zfit) == yfit).mean())
    acc_ev = float((lr.predict(Zev) == yev).mean())
    print(f"  分类: fit acc {acc_fit:.4f} | eval acc {acc_ev:.4f}")

    # ---- OOD 打分：Mahalanobis 到各类中心的最小距离（标准化空间）----
    mus = np.stack([Zfit[yfit == ci].mean(axis=0) for ci in range(len(CLASSES))])
    covs = []
    for ci in range(len(CLASSES)):
        Xc = Zfit[yfit == ci]
        covs.append(np.cov(Xc, rowvar=False) if Xc.shape[0] > Xc.shape[1]
                    else np.eye(Xfit.shape[1]))
    cov_pooled = np.mean(covs, axis=0) + 0.1 * np.eye(Xfit.shape[1])
    prec = np.linalg.inv(cov_pooled)

    def maha_min(Z):
        d = Z[:, None, :] - mus[None, :, :]
        return np.einsum("ncd,de,nce->n", d, prec, d)

    maha_eval = maha_min(Zev)
    tau_maha = float(np.percentile(maha_eval, 99.5))    # 0.5% 误拒率
    in_cover = float((maha_min(Zfit) <= tau_maha).mean())
    print(f"  τ_maha(eval 99.5pct) = {tau_maha:.2f} | fit 覆盖率 {in_cover:.4f}")

    probes = ood_probes()
    P = np.array([gate_features(img) for _, img in probes
                  if gate_features(img) is not None])
    Pz = scaler.transform(P)
    probe_maha = maha_min(Pz)
    print(f"  合成探针 maha: min {probe_maha.min():.1f} max {probe_maha.max():.1f} "
          f"| >τ 比例 {(probe_maha > tau_maha).mean():.4f}")

    np.savez(OUT_NPZ,
             coef=lr.coef_, intercept=lr.intercept_,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             mus=mus, prec=prec, tau_maha=np.array([tau_maha]),
             classes=np.array(CLASSES),
             acc_fit=np.array([acc_fit]), acc_eval=np.array([acc_ev]),
             date=np.array([datetime.now().isoformat(timespec="seconds")]))
    print("->", OUT_NPZ)


def eval_suites():
    print("加载门控 v2...")
    z = np.load(OUT_NPZ)
    coef, intercept = z["coef"], z["intercept"]
    mus, prec = z["mus"], z["prec"]
    classes = list(z["classes"])
    tau_maha = float(z["tau_maha"][0])
    smean, sscale = z["scaler_mean"], z["scaler_scale"]

    def classify(img, expected):
        body = img[img > 0.02]
        if body.size < 200:
            return "SKIP", None, None, None
        if body.std() < 0.01:
            return "OOD", None, None, None                    # 平坦/常数图（无结构）
        f = gate_features(img)
        if f is None:
            return "SKIP", None, None, None
        zf = (f - smean) / sscale
        d = zf - mus
        maha = float(np.min(np.einsum("nd,de,ne->n", d, prec, d)))
        if maha > tau_maha:
            return "OOD", None, None, maha                    # 距所有类中心过远
        logits = zf @ coef.T + intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        cls = classes[int(p.argmax())]
        if expected == "pelvis_MRI":
            v = "MATCH" if cls in ("pelvis_T1w", "pelvis_T2w") else "MISMATCH"
        else:
            v = "MATCH" if cls == expected else "MISMATCH"
        return v, cls, float(p.max()), maha                   # 判定 / 类别 / 置信 / Mahalanobis

    cases = []
    summary = {"tau_maha": tau_maha, "suites": {}}

    def suite(name, imgs, expected, want):
        rows = []
        for img in imgs:
            v, cls, mp, maha = classify(img, expected)
            rows.append(v)
            cases.append({"suite": name, "verdict": v, "pred": cls,
                          "maxp": None if mp is None else round(mp, 4),
                          "maha": None if maha is None else round(maha, 2)})
        n = len(rows)
        match_rate = sum(1 for v in rows if v == "MATCH") / n
        ood_rate = sum(1 for v in rows if v == "OOD") / n
        if want == "pelvis_MRI":
            ok = sum(1 for v in rows if v in ("MATCH", "MISMATCH")) / n
        else:
            ok = sum(1 for v in rows if v == want) / n
        summary["suites"][name] = {"n": n, "match_rate": round(match_rate, 4),
                                   "ood_rate": round(ood_rate, 4),
                                   "expected_outcome": want, "outcome_acc": round(ok, 4)}
        print(f"  {name}: n={n} 期望={want} 达成率={ok:.1%} "
              f"(MATCH {match_rate:.1%} / OOD {ood_rate:.1%})")

    print("\n===== 验证矩阵 =====")

    def load_glob(pat, lo, hi):
        return [np.load(p).astype(np.float32)
                for p in sorted(glob.glob(os.path.join(ROOT, pat)))[lo:hi]]

    suite("匹配·脑T1→T1源", load_glob("data/selfrdb_format_v3/T1/test/*.npy", 0, 100),
          "brain_T1", "MATCH")
    suite("匹配·脑T2→T2源", load_glob("data/selfrdb_format_v3/T2/test/*.npy", 0, 100),
          "brain_T2", "MATCH")
    suite("匹配·BraTS FLAIR→FLAIR源",
          load_glob("data/brats2018_slices/FLAIR/*.npy", 80, 180),
          "brain_FLAIR", "MATCH")
    def load_pelvis(p):
        a = np.load(p).astype(np.float32)
        im = Image.fromarray((a * 255).astype(np.uint8))
        return np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0

    suite("匹配·盆腔T1w→CT任务(任一盆腔MRI)",
          [load_pelvis(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PA*_mr.npy")))[480:]],
          "pelvis_MRI", "MATCH")
    suite("匹配·盆腔T2w→CT任务(任一盆腔MRI)",
          [load_pelvis(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PC*_mr.npy")))[160:240]],
          "pelvis_MRI", "MATCH")
    suite("匹配·盆腔T2w→CT任务(任一盆腔MRI)",
          [load_pelvis(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PC*_mr.npy")))[160:240]],
          "pelvis_MRI", "MATCH")
    suite("错配·盆腔CT喂MRI源任务(域外模态)",
          [load_pelvis(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PA*_ct.npy")))[:80]],
          "pelvis_MRI", "MISMATCH")

    suite("错配·脑T1喂T2源任务", load_glob("data/selfrdb_format_v3/T1/test/*.npy", 0, 100),
          "brain_T2", "MISMATCH")
    suite("错配·脑T2喂T1源任务", load_glob("data/selfrdb_format_v3/T2/test/*.npy", 0, 100),
          "brain_T1", "MISMATCH")
    suite("错配·脑喂盆腔CT源任务", load_glob("data/selfrdb_format_v3/T2/test/*.npy", 0, 100),
          "pelvis_CT", "MISMATCH")
    def load_pelvis2(p):
        a = np.load(p).astype(np.float32)
        im = Image.fromarray((a * 255).astype(np.uint8))
        return np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0

    suite("错配·盆腔CT喂脑T2源任务",
          [load_pelvis2(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PA*_ct.npy")))[:80]],
          "brain_T2", "MISMATCH")
    suite("错配·盆腔T1w喂脑FLAIR源任务",
          [load_pelvis2(p) for p in sorted(glob.glob(os.path.join(ROOT, "data/synthrad_pelvis_slices/1PA*_mr.npy")))[480:]],
          "brain_FLAIR", "MISMATCH")

    probes = ood_probes(n_per=30)
    suite("OOD·合成探针（噪声/平坦/棋盘/平滑）",
          [img for _, img in probes], "none", "OOD")

    run_dir = os.path.join(ROOT, "results", f"{datetime.now():%Y-%m-%d}_gate_v2")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "verdict", "pred", "maxp", "maha"])
        w.writeheader(); w.writerows(cases)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("->", run_dir)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "train"
    if phase == "train":
        train()
    else:
        eval_suites()
