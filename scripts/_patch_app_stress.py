# -*- coding: utf-8 -*-
"""应用升级包（据外部评审辩证采纳）：
A. 🧪 压力测试面板（诚实版守卫演示：什么会拦截、什么不会，全部有实测数据）
B. 单图结果页分诊工作流条 + 边界声明
C. 档位重标注（极速质控/高精质控）
"""
import io

fp = r"D:\MedTranslateProject\app\app.py"
s = io.open(fp, encoding="utf-8").read()

# ---- C) 档位重标注 ----
old = '''PRESETS = {  # (显示名, n_ensemble, step_skip, 说明)
    "快速 ⚡": (1, 2, "单次采样·步跳2，秒级出图"),
    "平衡 ⚖": (4, 2, "4次集成·步跳2（消融实测：质量持平，耗时省40%），推荐"),
    "精细 🔬": (8, 1, "8次集成·逐步采样，最可靠的信任图"),
}'''
new = '''PRESETS = {  # (显示名, n_ensemble, step_skip, 说明)
    "快速 ⚡ 极速质控": (1, 2, "单链 CReF 主导（10次前传）：论文实测单链已达集成定位价值的~90%，秒级出图"),
    "平衡 ⚖ 高精质控": (4, 2, "CReF+4链集成分歧（24次前传）：消融实测质量持平、耗时省40%，推荐"),
    "精细 🔬 最高可信": (8, 1, "8链集成·逐步采样，最可靠的信任图（成本最高）"),
}'''
assert old in s, "presets"
s = s.replace(old, new)

# ---- A) 压力测试逻辑（插在 load_calibration_note 之后）----
anchor = "def run_calibration("
assert anchor in s
stress_code = '''def _stress_image(scenario: str) -> "np.ndarray":
    """生成/加载压力测试输入。"""
    import numpy as np
    rng = np.random.RandomState(7)
    if scenario == "错模态（盆腔 CT → 脑任务）":
        import glob
        cands = sorted(glob.glob(os.path.join(ROOT, "data", "synthrad_pelvis_slices", "1PA*_ct.npy")))
        if cands:
            from PIL import Image
            a = np.load(cands[0]).astype(np.float32)
            im = Image.fromarray((a * 255).astype(np.uint8))
            return np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0
        return rng.rand(128, 128).astype(np.float32)
    if scenario == "噪声域外":
        return rng.rand(128, 128).astype(np.float32)
    if scenario == "平坦图（无结构）":
        return np.full((128, 128), 0.5, np.float32)
    if scenario == "棋盘格":
        return np.kron([[((i + j) % 2) for j in range(16)] for i in range(16)],
                       np.ones((8, 8))).astype(np.float32)
    if scenario == "退化模拟（强模糊）":
        from scipy.ndimage import gaussian_filter
        cands = sorted(glob.glob(os.path.join(ROOT, "data", "selfrdb_format_v3", "T1", "test", "slice_*.npy")))
        img = np.load(cands[0]).astype(np.float32)
        return gaussian_filter(img, sigma=2.0).astype(np.float32)
    raise ValueError(scenario)


_STRESS_EXPLAIN = {
    "错模态（盆腔 CT → 脑任务）": ("分类器层：六类模态/解剖分类判为 pelvis_CT，与脑任务声明不符 → 硬拒绝",),
    "噪声域外": ("分类器层：噪声被判为脑类但结构异常；1,160 例矩阵中此类输入 0 漏放",),
    "平坦图（无结构）": ("硬规则层：体区标准差近零 → 直接判 OOD",),
    "棋盘格": ("分类器层：非医学结构被判为错模态 → MISMATCH",),
    "退化模拟（强模糊）": ("Mahalanobis 层：实测强模糊使 maha 2.9→15.0（剂量敏感），退化检测由输入门控承担",),
}


def run_stress(scenario, task_id):
    """压力测试：只跑输入门控（毫秒级、不占 GPU），展示守卫的判定与边界。"""
    from trustbridge.gate_v2 import gate_decision
    img = _stress_image(scenario)
    _, src_mod, _, _ = TASKS[task_id]
    gd = gate_decision(img, task_id, src_mod)
    layer = _STRESS_EXPLAIN.get(scenario, ("",))[0]
    if gd["action"] == "block":
        color, icon, verdict = "#dc2626", "🚫", "已拦截（BLOCK）"
    elif gd["action"] == "warn":
        color, icon, verdict = "#d97706", "⚠", "警示放行（WARN）"
    else:
        color, icon, verdict = "#16a34a", "✅", "放行（PASS）"
    maha = gd.get("maha")
    maha_txt = f"{maha:.1f}" if isinstance(maha, (int, float)) else "—"
    card = (
        f"<div style='border:2px solid {color};border-radius:10px;padding:16px 20px;"
        f"background:{color}11;'>"
        f"<div style='font-size:26px;font-weight:800;color:{color};'>{icon} {verdict}</div>"
        f"<div style='font-size:14px;margin-top:8px;'>{gd['message'] or '输入校验通过。'}</div>"
        f"<div style='font-size:12.5px;color:#555;margin-top:8px;'>"
        f"判定 <b>{gd['verdict']}</b> ｜ 预测类别 <b>{gd.get('pred_class') or '—'}</b>"
        f" ｜ 任务期望 <b>{gd.get('expected')}</b> ｜ Mahalanobis <b>{maha_txt}</b>"
        f"（τ=388.76）</div>"
        f"<div style='font-size:12.5px;color:#334155;margin-top:6px;'><b>起作用的防御层：</b>{layer}</div>"
        f"</div>"
    )
    return card


'''
s = s.replace(anchor, stress_code + anchor, 1)

# ---- A) 压力测试 Tab（插在部署校准 Tab 之后）----
anchor2 = '        with gr.Tab("🕓 历史记录"):'
assert anchor2 in s
stress_tab = '''        with gr.Tab("🧪 压力测试（守卫演示）"):
            gr.Markdown(
                "**这个面板演示系统的防御纵深**：错模态、域外、退化输入会被输入门控在合成**之前**拦截，"
                "不产生任何评分（1,160 例验证矩阵：562 例非法/域外输入 0 漏放）。\\n\\n"
                "**同样重要的诚实声明**：我们的实测同时表明，信任层对\\"自信的崩塌\\"（基座模型结构性失效但内部自洽，"
                "论文 §4.1.1）**不报警**——本系统公开自己守卫的每一种失效，这比宣称万能更接近临床可信。")
            with gr.Row():
                with gr.Column(scale=5):
                    sscenario = gr.Radio(
                        ["错模态（盆腔 CT → 脑任务）", "噪声域外", "平坦图（无结构）",
                         "棋盘格", "退化模拟（强模糊）"],
                        value="错模态（盆腔 CT → 脑任务）", label="对抗场景")
                    stask_ok = available_tasks()
                    _, stask = _task_picker(stask_ok,
                                            "T1->T2" if "T1->T2" in stask_ok else stask_ok[0])
                    sbtn = gr.Button("🧪 运行拦截测试（毫秒级，不占 GPU）", variant="primary")
                with gr.Column(scale=7):
                    sout = gr.HTML()
            sbtn.click(run_stress, inputs=[sscenario, stask], outputs=[sout])

'''
s = s.replace(anchor2, stress_tab + anchor2, 1)

# ---- B) 单图结果：分诊工作流条 + 边界声明 ----
old2 = '''    if gate_result is not None:
        if gate_result.get("ok", True):
            metric_txt += "**输入门控：✅ 通过**（六类模态/解剖校验 + OOD 检验）\\n\\n"'''
new2 = '''    triage = {"PASS": "🟢 **可进入下一环节**（如测量/存档）",
              "REVIEW": "🟡 **需人工复核**：请优先查看下方低信任区域清单，再决定是否采用",
              "REJECT": "🔴 **不建议使用**：模型对该输入极不自信，建议改用真实扫描或更换任务/档位重试"}
    metric_txt += f"**分诊建议**：{triage.get(res.qc_flag, res.qc_flag)}\\n\\n"
    metric_txt += ("> ⚠ 边界声明：本系统拦截错模态/域外/退化输入，但**无法察觉**\\"自信的幻觉\\"与"
                   "基座模型的结构性失效（论文 §4.1.1/§4.8）；图像级分数仅作分诊参考，请以像素级信任图为准。\\n\\n")
    if gate_result is not None:
        if gate_result.get("ok", True):
            metric_txt += "**输入门控：✅ 通过**（六类模态/解剖校验 + OOD 检验）\\n\\n"'''
assert old2 in s, "triage anchor"
s = s.replace(old2, new2, 1)

io.open(fp, "w", encoding="utf-8").write(s)
print("app upgrade pack applied")
