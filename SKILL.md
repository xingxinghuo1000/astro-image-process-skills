---
name: astro-image-process-skills
description: >-
  Use when aligning, registering and stacking astrophotography / star-field
  frames shot on a fixed tripod, e.g. a Canon CR2/RAW night-sky sequence.
  Registers the moving sky, rejects the static foreground and hot-pixel noise,
  then outputs a brightness-stretched stacked image. Trigger keywords: 星野,
  星空, 银河, 对齐, 叠加, star field alignment, astro stacking, CR2 alignment.
---

# Astro Image Process Skills

工具集：把固定三脚架拍摄的连续星野照片（Canon CR2/RAW）**对齐 + 叠加 + 自动拉伸**。

## 为什么难，以及本工具怎么解决

- 星空在移动，地面景物静止 → 全图相关会被前景/噪声锁死。工具**只对齐星空**（相似变换：旋转 + 平移），前景会有轻微拖影。
- CMOS 热噪点在每帧同一像素位置，会被误当作特征匹配。工具先构建**静态图案图**（逐像素高分位数）并扣除，只留下会移动的真实星点。
- 用 **ORB + RANSAC** 求相似变换，再用**星点质心**迭代精修到亚像素；自动剔除配准失败的帧。
- 用**中值叠加**去掉热噪点拖出的彩色条纹；再用自适应 MTF **自动拉伸**，让银河显现。

## 环境

始终使用仓库根目录的虚拟环境 `.venv/`（Windows）：

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 运行

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" \
    --start 5994 --end 6012 \
    --out out --method median
```

常用参数：

- `--method mean|median` — 均值信噪比更高；**中值**可去除热噪点条纹、卫星轨迹（默认 mean，星野推荐 median）。
- `--ref N` — 参考帧编号，默认取序列中间帧。
- `--stretch-bg 0.20` — 拉伸后背景目标亮度，越大越亮。
- `--save-frames` — 另存每帧对齐后的全分辨率 TIFF（较占空间）。
- `--exclude 6007,6012` — 剔除个别帧。

## 输出（在 `--out` 目录）

| 文件 | 说明 |
| --- | --- |
| `stack_<method>.tif` | 对齐叠加后的**线性** 16-bit 原图（供后期） |
| `stack_<method>_stretched.tif` | 自适应拉伸后的 16-bit 成品 |
| `stretched_preview.jpg` / `stack_preview.jpg` | 预览图 |
| `transforms.json` | 每帧的相似变换、内点数、评分 |
| `aligned/IMG_xxxx.tif` | 每帧对齐结果（需 `--save-frames`） |

## 注意

- 输入 CR2 放在仓库外（如 `D:\pic\...`），不要拷进仓库。
- `out/`、`*.tif`、`.venv/` 已 git 忽略，勿提交。
- OpenCV `cv2.imwrite` 需要 **BGR**；写 TIFF 用脚本内 `write_tiff16`，否则红蓝通道会互换。
