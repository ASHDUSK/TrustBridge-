"""TrustBridge 评估微服务（FastAPI）——把应用能力以 API 形式开放给 HIS/PACS 集成方。

启动: uvicorn api:app --host 127.0.0.1 --port 8000（在 D:\MedTranslateProject 下）
文档: http://127.0.0.1:8000/docs

POST /evaluate   上传一张切片（PNG/JPG/NIfTI/DICOM）→ 翻译 + 质控
  返回: trust_score, qc_flag(分诊), lowtrust_ratio_pct, suspicious_regions[],
        input_gate{verdict,pred_class,maha}, elapsed_ms
  说明: 这是「极速质控」档（单链 CReF，零额外成本）；带 GT 时加 &with_metrics=1

POST /gate       只跑输入门控（毫秒级、不占 GPU），适合前置拦截
GET  /tasks      当前可用任务与权重状态
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="TrustBridge Trust API",
              description="医学图像翻译的信任评估微服务（科研演示，非临床设备）",
              version="0.4")

_ENGINES: dict = {}


def _engine(task_id: str):
    from trustbridge.engine import BridgeEngine
    if task_id not in _ENGINES:
        _ENGINES[task_id] = BridgeEngine(task_id)
    return _ENGINES[task_id]


def _load_arr(data: bytes, filename: str) -> np.ndarray:
    tmp = os.path.join(ROOT, "outputs", "api_tmp")
    os.makedirs(tmp, exist_ok=True)
    p = os.path.join(tmp, f"_{time.time_ns()}_{os.path.basename(filename)}")
    with open(p, "wb") as f:
        f.write(data)
    try:
        from trustbridge.io_utils import load_image
        img, _meta = load_image(p)
        return img
    finally:
        try:
            os.remove(p)
        except OSError:
            pass


@app.get("/tasks")
def tasks():
    from trustbridge.engine import TASKS
    return {"tasks": [{"id": k, "source": v[1], "target": v[2]} for k, v in TASKS.items()]}


@app.post("/gate")
def gate(file: UploadFile = File(...), task_id: str = Form("T1->T2")):
    """仅输入门控：错模态/域外/退化拦截，毫秒级。"""
    from trustbridge.engine import TASKS
    from trustbridge.gate_v2 import gate_decision
    if task_id not in TASKS:
        return JSONResponse(status_code=400, content={"error": f"未知任务 {task_id}"})
    img = _load_arr(file.file.read(), file.filename)
    t0 = time.time()
    gd = gate_decision(img, task_id, TASKS[task_id][1])
    return {"action": gd["action"], "verdict": gd["verdict"],
            "pred_class": gd.get("pred_class"), "expected": gd.get("expected"),
            "maha": gd.get("maha"), "message": gd["message"],
            "latency_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/evaluate")
def evaluate(file: UploadFile = File(...), task_id: str = Form("T1->T2"),
             with_metrics: bool = Form(False)):
    """完整评估：翻译 + 信任图 + 分诊（极速质控档：单链 CReF）。"""
    from trustbridge.engine import TASKS
    from trustbridge.pipeline import translate_single
    from trustbridge import report as rpt
    if task_id not in TASKS:
        return JSONResponse(status_code=400, content={"error": f"未知任务 {task_id}"})
    img = _load_arr(file.file.read(), file.filename)
    eng = _engine(task_id)
    display, src_mod, tgt_mod, _ = TASKS[task_id]
    t0 = time.time()
    out = translate_single(eng, img, n_ensemble=1, step_skip=2, seed=0)
    res, pred = out["result"], out["pred01"]
    regions = rpt.lowtrust_regions(res.trust_map, float(res.lowtrust_thresh))
    payload = {
        "task": task_id,
        "trust_score": round(float(res.trust_score), 1),
        "qc_flag": res.qc_flag,
        "triage": {"PASS": "可进入下一环节", "REVIEW": "需人工复核",
                   "REJECT": "不建议使用"}.get(res.qc_flag, res.qc_flag),
        "lowtrust_ratio_pct": round(float(res.lowtrust_ratio), 2),
        "suspicious_regions": [
            {"centroid_rc": r["质心(行,列)"], "area_px": r["面积(px)"],
             "pct": r["占比(%)"], "mean_trust": r["区域平均信任"]} for r in regions],
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "disclaimer": ("科研演示，不得用于临床诊断；本系统拦截错模态/域外输入，"
                       "但无法察觉自信的幻觉与基座模型结构性失效（见论文 §4.1.1）。"),
    }
    if with_metrics:
        payload["note"] = "带 GT 指标请走 scripts/evaluate_your_model.py 的离线批量协议"
    return payload
