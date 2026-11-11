# -*- coding: utf-8 -*-
"""v0.8.3 重算链 2（GPU 空闲后运行）：uq_baselines 双重归一化修复后的 4 组重跑
（自训×2 + 官方×2），随后 CPU 端 patient_stats 重算。"""
import subprocess, sys, time

CMDS = [
    # 自训退化重跑（链1中官方运行覆盖了同目录，目录已改名 _official）
    ["scripts/eval_degradation.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "30"],
    # 自训（显式 ckpt）
    ["scripts/eval_uq_baselines.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "60"],
    ["scripts/eval_uq_baselines.py", "--task", "T2->T1", "--ckpt", "checkpoints/self_t2t1.ckpt", "--n", "60"],
    # 官方权重（不传 --ckpt → TASKS 默认官方权重）
    ["scripts/eval_uq_baselines.py", "--task", "T2->T1", "--n", "60"],
    ["scripts/eval_uq_baselines.py", "--task", "T1->T2", "--n", "60"],
    # 崩塌甄别重跑（论文口径融合 + 体区 AUROC）
    ["scripts/eval_paper_extras.py", "collapse", "--n", "20"],
    # 肿瘤分割分析重跑（论文口径融合；tumor_trust_corrected）
    ["scripts/eval_tumor_corrected.py", "--n", "60"],
    # CPU：患者级配对 bootstrap（读上面四组 + 链1的 tumor/crossinst 产物）
    ["scripts/eval_patient_stats.py"],
    # CPU：QC 阈值工作点（PASS 75 / REVIEW 55 的拦截表现）
    ["scripts/eval_qc_thresholds.py"],
]

if __name__ == "__main__":
    t00 = time.time()
    for cmd in CMDS:
        print(f"\n==== V083 RECALC2: {' '.join(cmd[1:5])} ==== [{time.strftime('%H:%M:%S')}]",
              flush=True)
        subprocess.run([sys.executable] + cmd, cwd=r"D:\MedTranslateProject",
                       check=False)
    print(f"\nV083_RECALC2_DONE total {(time.time()-t00)/60:.1f} min", flush=True)
