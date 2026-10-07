# 上传到 GitHub

以下步骤将项目发布为代码仓库，并将其展示在个人主页上。这里只说明操作步骤，尚未创建或上传远程仓库。

## 1. 准备独立目录

使用 `pv-semantic-edge-github.zip`，解压到桌面等独立位置，例如：

```text
C:\Users\ZHU\Desktop\pv-semantic-edge
```

该目录打开后应直接看到 `README.md`、`pyproject.toml`、`src/` 和 `configs/`。上传解压后的项目文件，而不是仅上传 ZIP。

当前开发目录位于另一个 Git 工作目录之下。使用独立的发布目录，可避免把其他项目一起提交。发布包已排除 `.venv/`、`runs/`、`data/` 和模型权重。

## 2. 在 GitHub 创建空仓库

登录 GitHub，点击右上角 **+ → New repository**：

- Repository name：可用 `pv-semantic-edge`。
- Description：`Sketch-conditioned augmentation and edge-aware recognition for photovoltaic EL defects.`
- 需要公开展示时选择 **Public**。
- 不勾选初始化 README、.gitignore 或 License；项目中已有这些文件。

点击 **Create repository**，复制仓库的 HTTPS 地址，例如：

```text
https://github.com/YOUR_USERNAME/pv-semantic-edge.git
```

`YOUR_USERNAME` 替换为自己的 GitHub 用户名。

## 3. 提交并上传

安装 [Git for Windows](https://git-scm.com/downloads/win) 后，在解压目录打开 PowerShell，执行：

```powershell
cd "C:\Users\ZHU\Desktop\pv-semantic-edge"
git init -b main
git add .
git status
```

确认待提交内容是项目源码、配置、文档和测试，然后执行：

```powershell
git commit -m "Add EL augmentation and defect recognition code"
git remote add origin https://github.com/YOUR_USERNAME/pv-semantic-edge.git
git push -u origin main
```

第一次提交若提示没有姓名或邮箱，可在该仓库设置提交信息，再重新执行 `git commit`：

```powershell
git config user.name "你的名字"
git config user.email "你的 GitHub 邮箱或 noreply 邮箱"
```

推送时按登录提示完成 GitHub 认证。若现有认证方式要求令牌，请按 GitHub 的认证流程操作，不要把令牌写入远程地址或项目文件。

以上过程与 [GitHub 官方本地代码上传说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows) 对应。

## 4. 展示在个人主页

上传完成后，进入自己的 GitHub 个人主页，在仓库展示区点击 **Customize your pins**，选中本仓库并保存。操作见 [GitHub 官方置顶说明](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile)。

在仓库首页的 **About** 中，可以填写论文 DOI 链接，并添加 `photovoltaics`、`electroluminescence`、`defect-detection`、`data-augmentation`、`latent-diffusion` 等主题。

代码仓库置顶后即可出现在个人主页，不需要额外启用 GitHub Pages。

## 5. 后续更新

在同一个发布目录修改文件后执行：

```powershell
git add .
git commit -m "Update experiment settings and documentation"
git push
```

数据公开后，在 README 中补充下载地址、类别顺序和数据划分说明。模型权重如单独发布，再补充对应配置和下载链接。
