# -*- coding: utf-8 -*-
"""链2b：官方 uq ×2（align_map 修复后）+ patient_stats + qc_thresholds 补跑。"""
import subprocess, sys, time

CMDS = [
    ["scripts/eval_uq_baselines.py", "--task", "T2->T1", "--n", "60"],
    ["scripts/eval_uq_baselines.py", "--task", "T1->T2", "--n", "60"],
    ["scripts/eval_patient_stats.py"],
    ["scripts/eval_qc_thresholds.py"],
]

if __name__ == "__main__":
    t00 = time.time()
    for cmd in CMDS:
        print(f"\n==== V083 RECALC2b: {' '.join(cmd[1:4])} ==== [{time.strftime('%H:%M:%S')}]",
              flush=True)
        subprocess.run([sys.executable] + cmd, cwd=r"D:\MedTranslateProject", check=False)
    print(f"\nV083_RECALC2B_DONE total {(time.time()-t00)/60:.1f} min", flush=True)
