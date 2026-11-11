# -*- coding: utf-8 -*-
"""链2c：slice_scores ×2（QC/Pearson 数据源，论文口径融合）+ qc_thresholds 重跑。"""
import subprocess, sys, time

CMDS = [
    ["scripts/eval_slice_scores.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "60"],
    ["scripts/eval_slice_scores.py", "--task", "T2->T1", "--ckpt", "checkpoints/self_t2t1.ckpt", "--n", "60"],
    ["scripts/eval_qc_thresholds.py"],
]

if __name__ == "__main__":
    t00 = time.time()
    for cmd in CMDS:
        print(f"\n==== V083 RECALC2c: {' '.join(cmd[1:4])} ==== [{time.strftime('%H:%M:%S')}]",
              flush=True)
        subprocess.run([sys.executable] + cmd, cwd=r"D:\MedTranslateProject", check=False)
    print(f"\nV083_RECALC2C_DONE total {(time.time()-t00)/60:.1f} min", flush=True)
