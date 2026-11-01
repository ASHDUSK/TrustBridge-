"""输入门控 v2：6 类模态/解剖分类 + Mahalanobis OOD 拒判（纯 numpy 推理）。

v1（gate.py）只能判 T1/T2 二分类；v2 扩展为 6 类
（brain_T1 / brain_T2 / brain_FLAIR / pelvis_T1w / pelvis_T2w / pelvis_CT），
并加两条 OOD 硬规则：
  1) 平坦/常数图（体区标准差近零）→ OOD；
  2) 特征到所有类中心的 Mahalanobis 最小距离 > τ（训练集验证分位）→ OOD。

应用层策略（gate_decision）：
  OOD      → 硬拒绝（不合成、不出评分）——防止对域外输入伪造信任评分；
  MISMATCH → 脑任务硬拒绝（验证矩阵 0% 误拦）；
             CT 任务降级为警示放行（未见患者 T1w 有 52.5% 被误判为脑类，
             硬拒绝会拦掉一半合法输入——见 docs/16 门控 v2 验证矩阵）；
  MATCH    → 放行。

模型系数在 trustbridge/modality_gate_v2.npz，训练/验证脚本 scripts/train_gate_v2.py。
"""
from __future__ import annotations

import os

import numpy as np

from trustbridge.gate import gate_features

PKG = os.path.dirname(os.path.abspath(__file__))
MODEL_V2 = os.path.join(PKG, "modality_gate_v2.npz")

BRAIN_CLASSES = ("brain_T1", "brain_T2", "brain_FLAIR")
PELVIS_MRI_CLASSES = ("pelvis_T1w", "pelvis_T2w")

# 任务声明 → 期望源类别（按源模态映射；CT 任务用盆腔 MRI 组期望，
# 因盆腔 T1w/T2w 跨患者区分不可靠，见验证矩阵）
_EXPECTED_BY_SRC = {"T1": "brain_T1", "T2": "brain_T2",
                    "FLAIR": "brain_FLAIR", "PD": "brain_any"}
BRAIN_ANY_CLASSES = BRAIN_CLASSES  # PD 源无对应类别，接受任意脑类


def expected_for_task(task_id: str, src_mod: str) -> str:
    if task_id.startswith("CT_"):
        return "pelvis_MRI"
    return _EXPECTED_BY_SRC.get(str(src_mod).upper(), "brain_any")


class GateV2:
    """6 类分类 + OOD 拒判。模型缺失时 enabled=False（优雅降级）。"""

    def __init__(self, path: str = MODEL_V2):
        self.enabled = os.path.exists(path)
        if not self.enabled:
            return
        z = np.load(path)
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.mus, self.prec = z["mus"], z["prec"]
        self.tau = float(z["tau_maha"][0])
        self.classes = [str(c) for c in z["classes"]]
        self.smean, self.sscale = z["scaler_mean"], z["scaler_scale"]

    def classify(self, img01: np.ndarray, expected: str) -> dict:
        """img01 ∈ [0,1] → {verdict, pred_class, maxp, maha, note}。

        verdict ∈ MATCH / MISMATCH / OOD / SKIP；SKIP 表示无法评估（图太小等）。
        """
        if not self.enabled:
            return {"verdict": "SKIP", "pred_class": None, "maxp": None,
                    "maha": None, "note": "gate v2 model unavailable"}
        f = gate_features(img01)
        if f is None:
            return {"verdict": "SKIP", "pred_class": None, "maxp": None,
                    "maha": None, "note": "image too small / no body"}
        body = img01[img01 > 0.02]
        if body.std() < 0.01:
            return {"verdict": "OOD", "pred_class": None, "maxp": None,
                    "maha": None, "note": "flat/constant image (no structure)"}
        zf = (f - self.smean) / self.sscale
        d = zf - self.mus
        maha = float(np.min(np.einsum("nd,de,ne->n", d, self.prec, d)))
        if maha > self.tau:
            return {"verdict": "OOD", "pred_class": None, "maxp": None,
                    "maha": maha, "note": "far from all class centers"}
        logits = zf @ self.coef.T + self.intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        cls = self.classes[int(p.argmax())]
        if expected == "pelvis_MRI":
            ok = cls in PELVIS_MRI_CLASSES
        elif expected == "brain_any":
            ok = cls in BRAIN_ANY_CLASSES
        else:
            ok = cls == expected
        return {"verdict": "MATCH" if ok else "MISMATCH", "pred_class": cls,
                "maxp": round(float(p.max()), 4), "maha": round(maha, 2),
                "note": ""}


_GATE: GateV2 | None = None


def get_gate() -> GateV2:
    global _GATE
    if _GATE is None:
        _GATE = GateV2()
    return _GATE


def gate_decision(img01: np.ndarray, task_id: str, src_mod: str) -> dict:
    """应用层决策：verdict → allow / warn / block。

    返回 {action, verdict, pred_class, maxp, maha, message}；
    message 面向用户（中文），block 时不含评分路径。
    """
    expected = expected_for_task(task_id, src_mod)
    r = get_gate().classify(img01, expected)
    v = r["verdict"]
    r["action"] = "allow"
    r["message"] = ""
    r["expected"] = expected
    if v == "OOD":
        r["action"] = "block"
        r["message"] = (f"输入被判定为域外数据（OOD）"
                        f"{'：' + r['note'] if r['note'] else ''}，"
                        f"已拒绝翻译，不生成信任评分。请确认上传的是"
                        f"{'盆腔 MRI' if expected == 'pelvis_MRI' else '脑 MRI'} 图像。")
    elif v == "MISMATCH":
        pred = r["pred_class"]
        if expected == "pelvis_MRI":
            # CT 任务：未见患者 T1w 有 52.5% 被误判为脑类（已知局限），降级为警示
            r["action"] = "warn"
            r["message"] = (f"输入模态存疑：门控判定为「{pred}」，"
                            f"与任务声明的盆腔 MRI 源不符。结果已标注「输入校验未通过」，"
                            f"请核对任务选择与上传内容。")
        else:
            r["action"] = "block"
            r["message"] = (f"输入模态与任务不符：门控判定为「{pred}」，"
                            f"而任务（{task_id}）要求「{expected}」源图像。"
                            f"已拒绝翻译，防止跨模态输入生成不可信评分。")
    return r
