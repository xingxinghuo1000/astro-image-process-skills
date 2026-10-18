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

> **注意**：本技能所在的目录（存放 `SKILL.md`、`scripts/` 的安装位置）**不一定等于**
> 当前工作目录。不要默认 `.venv` 就在技能目录旁边——它更可能位于**工作目录**。

选择 Python 解释器的顺序：

1. **优先在工作目录**查找已存在的虚拟环境，依次尝试
   `.venv/`、`venv/`（Windows：`Scripts\python.exe`；POSIX：`bin/python`）。
   用 `pyvenv.cfg` 是否存在，或
   `python -c "import sys; print(sys.prefix != sys.base_prefix)"` 返回 `True`
   来确认它**确实是一个 Python 虚拟环境**；是则**直接用**，不要再建新的。
2. 若工作目录里没有可用的虚拟环境，就在**工作目录**创建 `.venv/` 并安装依赖：

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

（`requirements.txt` 用技能目录里的那份；Aliyun 镜像在国内快很多。）

## 运行

下面示例以工作目录下已存在 `.venv`（Windows）为前提，脚本路径指向本技能目录：

```bash
./.venv/Scripts/python.exe <skill-dir>/scripts/align_stack.py \
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
- `--neutralize` — **背景中性化**（类 PixInsight `BackgroundNeutralization`）：按通道测量天空背景并缩放到中性灰，先在线性主图上做、拉伸后再校正一次，消除偏色（如光害导致的紫/绿底）。
- `--star-reduce PX` — **缩星**（类 PixInsight `StarReduction`）：高通星点掩膜下的形态学腐蚀，把星点半径缩小约 PX 像素（推荐 1~3），背景与银河不受影响，让星云/银河更突出。0 为关闭。

## 输出（在 `--out` 目录）

| 文件 | 说明 |
| --- | --- |
| `stack_<method>.tif` | 对齐叠加后的**线性** 16-bit 原图（供后期） |
| `stack_<method>_stretched.tif` | 自适应拉伸后的 16-bit 成品 |
| `stretched_preview.jpg` / `stack_preview.jpg` | 预览图 |
| `transforms.json` | 每帧的相似变换、内点数、评分 |
| `star_mask.png` | 星点掩膜（使用 `--star-reduce` 时输出，便于检查） |
| `aligned/IMG_xxxx.tif` | 每帧对齐结果（需 `--save-frames`） |

## 注意

- 输入 CR2 放在仓库外（如 `D:\pic\...`），不要拷进仓库。
- `out/`、`*.tif`、`.venv/` 已 git 忽略，勿提交。
- OpenCV `cv2.imwrite` 需要 **BGR**；写 TIFF 用脚本内 `write_tiff16`，否则红蓝通道会互换。
