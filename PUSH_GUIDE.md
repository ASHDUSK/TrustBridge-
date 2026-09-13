# PUSH_GUIDE · 推送到 GitHub 与发布 Release（3 步 + 网络预案）

> 本仓库已在本地完成 init / commit / tag。以下操作需要你的 GitHub 账号，
> 全程约 5 分钟（不含上传 Release 附件的时间）。

## 第 1 步：在 GitHub 网页创建空仓库

1. 登录 https://github.com → 右上角 **+** → **New repository**；
2. Repository name 建议 `trustbridge`（或 medbridge-studio）；
3. 选择 **Public**；**不要**勾选任何初始化选项（README/.gitignore/license 都不要——本地已就绪）；
4. Create repository。

## 第 2 步：关联远端并推送

在本目录打开终端：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
git push origin v0.1.0
```

推送后，把 `scripts/download_checkpoints.sh` 里 `SELF_REPO` 的
`https://github.com/YOUR_USERNAME/YOUR_REPO` 替换为实际地址，并提交一次小修改。

### 国内网络三种解法（按顺序尝试）

1. **代理**：如果你有代理（如 7890 端口）：
   ```bash
   git config --global http.proxy http://127.0.0.1:7890
   git config --global https.proxy http://127.0.0.1:7890
   # 推完取消: git config --global --unset http.proxy; git config --global --unset https.proxy
   ```
2. **GitHub Desktop**：网页登录后 File → Add local repository → 选本目录 → Push origin，全程走其内置通道，无需命令行；
3. **兜底——网页上传**：网页建仓库时勾选"Add README"生成初始分支，然后在仓库页 **Add file → Upload files** 把本目录全部文件拖拽上传（本仓库无超大文件，网页可承载），commit 即可。

## 第 3 步：创建 Release v0.1.0（放自训练权重）

1. 仓库页 → **Releases** → **Create a new release** → Choose a tag: 输入 `v0.1.0`；
2. Release 标题与正文：直接粘贴 `release_assets/RELEASE_TEMPLATE.md` 的内容，
   并把其中 `<网盘链接占位>` 替换为你的实际网盘地址；
3. **Attach binaries**：上传 `release_assets/self_t2t1.ckpt`（537MB < 2GB 单文件限制）；
4. Publish release。

发布后回填 `scripts/download_checkpoints.sh` 中 `SELF_REPO`（见第 2 步），别人即可
`bash scripts/download_checkpoints.sh` 一键拉齐全部权重。

## 推送后自查清单

- [ ] 无痕浏览器打开仓库页，README 图片（assets/）正常显示；
- [ ] `git clone` 到另一目录，`python app/app.py` 能启动；
- [ ] Release 页 self_t2t1.ckpt 可下载；
- [ ] `bash scripts/download_checkpoints.sh` 官方权重下载成功（必要时带 MIRROR= 前缀）。

## 推送前可选：把提交作者改成你的 GitHub 身份

本仓库的提交作者目前是占位身份（未检测到全局 git 配置）。若希望提交计入你的
GitHub 贡献统计，推送前在仓库目录执行（邮箱用 GitHub 账号绑定邮箱即可，无需真实可用）：

```bash
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub绑定邮箱"
git commit --amend --reset-author --no-edit
git tag -f v0.1.0
```

不改也不影响任何功能。

## 常见问题

- **push 提示 403**：仓库属主不对或无写权限——确认 remote 地址的用户名正确；
- **push 超时/连接重置**：见上文网络预案；仓库总大小约 2MB，任何网络下重试几次都能过；
- **想改仓库名**：GitHub 仓库 Settings 里改名即可，本地只需更新 remote URL。
