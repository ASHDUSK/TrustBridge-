# AutoDL 256² 训练包（T1→T2 方向）

## 内容
- data/selfrdb_format_256/   训练数据（train 2445 / val 545 / test 756 × T1,T2）
- third_party/SelfRDB/       官方代码（含 TrustBridge 最小补丁）
- configs/train_ixi_t1t2_256_cloud.yaml   云版配置（路径已 Linux 化）
- requirements_cloud.txt     补充依赖（镜像自带 torch/cuda）

## 启动（在 AutoDL 实例终端，数据已放到 /root/autodl-tmp/ 后）
```bash
cd /root/autodl-tmp
pip install -r requirements_cloud.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
nohup python third_party/SelfRDB/main.py fit \
  --config configs/train_ixi_t1t2_256_cloud.yaml \
  > train_t1t2_256.log 2>&1 &
tail -f train_t1t2_256.log     # 观察前几十步确认在跑（~3-4 s/step）
```
## 产出
- /root/autodl-tmp/runs/t1t2_256_cloud/**/checkpoints/*.ckpt（每 5 epoch 保存）
- 训练完打包：`tar czf ckpt_out.tar.gz runs/ &&` 用 AutoDL 文件传输下载
## 预算参考
4090 约 3-4 s/step × 18,400 步 ≈ 18-20 h ≈ 30-40 元；可随时关机（无中断续训需求时建议一次跑完）
