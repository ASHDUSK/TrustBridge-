# -*- coding: utf-8 -*-
"""Sprint 重算链：AUROC mask + 权重 0.6/0.4 修复后的全部受影响实验。"""
import subprocess, sys

CHAIN = [
    ["scripts/eval_uq_baselines.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "60"],
    ["scripts/eval_uq_baselines.py", "--task", "T2->T1", "--ckpt", "checkpoints/self_t2t1.ckpt", "--n", "60"],
    ["scripts/eval_degradation.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "30"],
    ["scripts/eval_tumor_sparsify.py", "--n", "60"],
    ["scripts/eval_crossinst_brats.py", "--n", "60"],
    ["scripts/eval_body_decile.py"],
]

for cmd in CHAIN:
    tag = " ".join(cmd[1:4])
    print(f"==== SPRINT: {tag} ====", flush=True)
    subprocess.run([sys.executable] + cmd, cwd=r"D:\MedTranslateProject", check=False)

print("SPRINT_RECALC_DONE", flush=True)
