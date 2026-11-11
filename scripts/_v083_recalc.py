# -*- coding: utf-8 -*-
"""v0.8.3 重算链：raw-map 0.6/0.4 融合 + 体区 AUROC 掩码 + align_map 全部修复后的受影响实验。

修复内容（相对 2026-11-10 重算链）：
  1. eval_degradation / eval_tumor_sparsify / eval_crossinst_brats / eval_body_decile
     的信任图统一改为 sampler 原始图手动融合（CReF 0.6 / 集成 0.4，论文口径），
     不再依赖 sampler 默认 (0.4/0.6) 的 res.trust_map。
  2. 上述脚本切片级 AUROC 全部改为体区掩码内计算（tr[mask], err[mask]）。
  3. 修复 res.rcrf_raw_map 属性错误（原始图在 sampler 实例上）与 >128 输入的对齐崩溃。
"""
import subprocess, sys, time

CMDS = [
    ["scripts/eval_degradation.py", "--task", "T1->T2", "--ckpt", "checkpoints/self_t1t2.ckpt", "--n", "30"],
    ["scripts/eval_degradation.py", "--task", "T1->T2", "--ckpt", "checkpoints/ixi_t1_t2.ckpt", "--n", "30"],
    ["scripts/eval_tumor_sparsify.py", "--n", "60"],
    ["scripts/eval_crossinst_brats.py"],
    ["scripts/eval_body_decile.py"],
]

if __name__ == "__main__":
    t00 = time.time()
    for cmd in CMDS:
        tag = " ".join(cmd[1:4])
        print(f"\n==== V083 RECALC: {tag} ==== [{time.strftime('%H:%M:%S')}]",
              flush=True)
        subprocess.run([sys.executable] + cmd, cwd=r"D:\MedTranslateProject",
                       check=False)
    print(f"\nV083_RECALC_DONE total {(time.time()-t00)/60:.1f} min", flush=True)
