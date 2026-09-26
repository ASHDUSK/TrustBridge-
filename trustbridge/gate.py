"""输入模态门控：检测"喂错模态/喂错权重"的输入（limitation ① 的系统化修复）。

原理：T1 与 T2 加权像的脑内强度分布形态差异显著（T1 白质亮/CSF 暗，T2 相反）。
对输入图做体区标准化（消除线性强度差异）后提取分位数+梯度特征，
用验证集训练的轻量逻辑回归判别 T1/T2，再与任务声明的源模态比对：
  - 匹配 → 放行
  - 不匹配 → FLAG（应用中警示，报告标注）

模型系数存 npz，推理端零 sklearn 依赖（纯 numpy）。
"""
from __future__ import annotations

import os

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
# 优先包内系数（随代码分发）；兼容旧路径 data/
DEFAULT_GATE = os.path.join(PKG, "modality_gate.npz")
if not os.path.exists(DEFAULT_GATE):
    DEFAULT_GATE = os.path.join(ROOT, "data", "modality_gate.npz")
GATE_PELVIS = os.path.join(ROOT, "data", "modality_gate_pelvis.npz")

# 任务族 → 门控文件（脑任务用脑门控；盆腔 CT 任务用盆腔门控）
GATE_FOR_TASK = {}


def gate_for_task(task_id: str) -> str:
    for k, v in GATE_FOR_TASK.items():
        if task_id.startswith(k):
            return v
    return DEFAULT_GATE

PCTS = [5, 10, 25, 50, 75, 90, 95, 99]


def gate_features(img01: np.ndarray, body_thr: float = 0.02) -> np.ndarray | None:
    """图像 → 门控特征向量（体区标准化，线性强度不变量）。"""
    img = img01.astype(np.float32)
    body = img[img > body_thr]
    if body.size < 200:
        return None
    mu, sd = body.mean(), body.std() + 1e-8
    z = (body - mu) / sd
    feats = [np.percentile(z, p) for p in PCTS]
    # 梯度（标准化图上）
    g = np.hypot(np.gradient(img, axis=0), np.gradient(img, axis=1))
    gb = g[img > body_thr]
    feats += [float(gb.mean()), float(np.percentile(gb, 90))]
    # 高亮占比（T2 的 CSF 高亮特征）
    feats += [float((img > body_thr + 0.5 * sd).mean())]
    return np.array(feats, dtype=np.float64)


def fit_gate(t1_slices: list[np.ndarray], t2_slices: list[np.ndarray],
             out_path: str = DEFAULT_GATE, body_thr: float = 0.02) -> dict:
    """IXI 验证集 T1/T2 切片训练门控，系数存 npz。"""
    X, y = [], []
    for s in t1_slices:
        f = gate_features(s)
        if f is not None:
            X.append(f); y.append(0)          # 0=T1
    for s in t2_slices:
        f = gate_features(s)
        if f is not None:
            X.append(f); y.append(1)          # 1=T2
    X = np.array(X); y = np.array(y)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(X, y)
    scaler, lr = clf[0], clf[1]
    # 等价变换：decision(x) = w·(x-mean_)/scale_ + b = (w/scale_)·x + (b - Σ w·mean_/scale_)
    coef = lr.coef_[0] / scaler.scale_
    intercept = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))
    np.savez(out_path, coef=coef, intercept=np.array([intercept]),
             pcts=np.array(PCTS), body_thr=np.array([body_thr]),
             train_t1=X[y == 0].shape[0], train_t2=X[y == 1].shape[0])
    # 训练自检
    prob = X @ coef + intercept
    acc = ((prob > 0).astype(int) == y).mean()
    return {"train_n": int(len(y)), "train_acc": round(float(acc), 4),
            "coef_path": out_path}


def check(img01: np.ndarray, expected_source: str,
          gate_path: str = DEFAULT_GATE, thr: float = 0.5) -> dict:
    """给定输入图与任务声明的源模态，返回是否匹配。expected_source ∈ {T1, T2}。"""
    z = np.load(gate_path)
    coef, intercept = z["coef"], float(z["intercept"][0])
    f = gate_features(img01)
    if f is None:
        return {"ok": True, "prob_t2": None, "note": "image too small for gate"}
    p_t2 = 1.0 / (1.0 + np.exp(-(f @ coef + intercept)))
    expected_cls = 1 if expected_source.upper() == "T2" else 0
    pred_cls = int(p_t2 > thr)
    ok = (pred_cls == expected_cls)
    return {"ok": bool(ok), "prob_t2": round(float(p_t2), 4),
            "expected": expected_source.upper(), "flagged": not ok}
