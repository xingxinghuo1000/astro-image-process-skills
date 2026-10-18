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
- `--neutralize` — **背景中性化**（类 PixInsight `BackgroundNeutralization`）：按通道测量天空背景并缩放到中性灰，消除偏色（如光害导致的紫/绿底）。作用于拉伸后的图像。
- `--flat` — **自动平场 / 背景梯度去除**（类 PixInsight `AutomaticBackgroundExtractor` ABE）：把图像分成网格、每格取低分位数估计天空背景，拟合一张低阶二维多项式背景面并按通道扣除（只扣除梯度的"形状"，保留暗天空基座，避免把背景压到 0）。**作用于拉伸前的线性图**：梯度在线性域是加性的，拟合/扣除才有物理意义；放到拉伸后会失真并干扰自动拉伸的背景定位。可调 `--flat-mode subtract|divide`、`--flat-bins`、`--flat-degree`、`--flat-pct`、`--flat-strength`。
  - **平场是否采用由用户决定**：只要用了 `--flat`，就一定会输出对比图 `flat_compare.jpg`（左：不平场，"成品"观感；右：平场后），供用户判断要不要保留。想一次比较多个强度，用 `--flat-sweep 0.4,0.5,0.6,0.7`，对比图里会平铺"不平场 + 各强度"，选好后再用 `--flat-strength` 定值正式出图。
  - ⚠️ **外圈亮环**：若画面被银河/大范围辉光铺满，低阶背景模型分不清"天空背景"和"目标本身的辉光"，满强度扣除会把中心压暗、边缘相对变亮，出现一圈亮环。此时把 `--flat-strength` 调到 ~0.3~0.6（默认 1.0）即可在保留梯度去除的同时消除亮环。
- `--star-reduce PX` — **缩星**（类 PixInsight `StarReduction`）：高通星点掩膜下的形态学腐蚀，把星点半径缩小约 PX 像素，背景与银河不受影响，让星云/银河更突出。**默认 1（轻微）**，`0` 关闭；想要更狠再用 2~3（2 已经比较激进，会明显抹掉细小星点）。
- `--composite` — **天地合成**（可选）：把「对齐到星空的天景」和「直接把多张照片叠加（不配准）得到的地景背景」在识别出的**天地分界线**处拼接。天景里星空锐利、前景被拖影；地景里前景锐利、星空拖出星轨——据此检测分界线。适合固定三脚架拍星空、想要"前景清晰 + 星空清晰"的合成。附加参数：
  - `--composite-horizon mask|line` — `mask`（默认）沿锯齿状树线/地形走；`line` 用一条稳健直线近似地平线。
  - `--composite-thresh T` — 天空证据阈值（占 99.5 分位的比例，默认 0.05）；调高把分界线上移（天空变少），调低纳入更暗的天空。
  - `--composite-feather PX` — 分界线两侧羽化融合带宽度（全分辨率像素，默认 25）。
  - `--ground-dir` / `--ground-start` / `--ground-end` — 用另一组专门的地景照片当地景层（默认用输入帧本身、不配准叠加）。

## 输出（在 `--out` 目录）

> **每个阶段都会输出中间结果**：`stageN_<name>.tif`（16-bit TIFF）+ 同名 `_preview.jpg`（半尺寸预览），`Num`（即 `N`）就是该阶段的序号，按实际启用的步骤递增；并把**所有阶段拼接成一张对比图** `stages_montage.jpg`（线性阶段会自动拉伸后展示，便于和后续阶段比较）。

| 文件 | 说明 |
| --- | --- |
| `stageN_<name>.tif` + `_preview.jpg` | **每个处理阶段的中间结果**（TIFF + 预览），`N` 为该阶段序号。如 `stage1_stacked` / `stage2_composite` / `stage3_flat` / `stage4_stretched` / `stage5_neutralized` / `stage6_star_reduced` |
| `stages_montage.jpg` | **各阶段成果多图拼接对比图**（全部阶段并排，带标签） |
| `merge_final_<start>_<end>.tif` | **最终成品**：16-bit TIFF（完整分辨率）。`<start>`/`<end>` 为**合并序列的起始/结束帧号**，如 `merge_final_5994_6012.tif` |
| `merge_final_<start>_<end>.jpg` | **最终成品**：JPEG（完整分辨率） |
| `stack_<method>.tif` | 对齐叠加后的**线性** 16-bit 主图（供后期） |
| `stack_preview.jpg` | 线性主图的半尺寸预览 |
| `transforms.json` | 每帧的相似变换、内点数、评分 |
| `star_mask.png` | 星点掩膜（使用 `--star-reduce` 时输出，便于检查） |
| `flat_compare.jpg` | 平场对比图（使用 `--flat` 时输出）：不平场 vs 平场（可含 `--flat-sweep` 强度扫描），**交给用户目视决定是否采用/选用哪个强度** |
| `ground_stack.tif` + `_preview.jpg` | 地景层：不配准直接叠加的帧（使用 `--composite` 时输出），前景锐利、星空拖轨 |
| `sky_mask.png` / `sky_ground_overlay.jpg` | 天地分界线掩膜 / 叠加了分界线的天景预览（使用 `--composite` 时输出，便于检查分割） |
| `aligned/IMG_xxxx.tif` | 每帧对齐结果（需 `--save-frames`） |

处理阶段顺序：**叠加 → 天地合成 → 自动平场 → 拉伸 → 中性化 → 缩星**（合成/平场/中性化/缩星均可关闭）。

**最终出图**：成品同时输出**完整分辨率的 TIFF 和 JPEG**，命名为 `merge_final_<start>_<end>.tif` / `.jpg`（`<start>`/`<end>` 是合并序列的起始/结束帧号），可直接交付；其余中间结果一律以 `stageN_` 开头。

## 平场工作流（对比 → 用户决策）

平场是"判断题"，不要替用户拍板。推荐流程：

1. 先跑一次 **带强度扫描**：`--flat --flat-sweep 0.4,0.5,0.6,0.7`；
2. 把生成的 `flat_compare.jpg` 给用户看（平铺了"不平场 + 各强度"的成品观感），让用户决定**是否要平场、用哪个强度**；
3. 用户定好后正式出图：`--flat --flat-strength <S>`（不加 `--flat` 即不平场）。

若画面被银河铺满、外圈出现亮环，就是强度偏大，调低 `--flat-strength`。

## 天地合成（`--composite`）

固定三脚架拍摄时星空在转、前景静止：

- **天景**：帧对齐到星空后叠加 → 星空锐利，前景被拖影。
- **地景**：帧**不配准**直接叠加 → 前景锐利，星空拖成星轨。
- 用**时间方差**得到"天空"证据图（未配准的帧在天空区域变化大，在地面无变化），取每列的**下包络**作为天地分界线：能跨过天极附近的低运动"空洞"，也能沿锯齿状树线走；最后在分界线羽化带内把两层融合。该步骤作用于线性图，之后的平场/拉伸/中性化/缩星都作用在合成图上。

分界线是否符合预期，看 `sky_ground_overlay.jpg`（青色 = 掩膜边界，绿线 = 直线拟合）。若不满意：调 `--composite-thresh`、改 `--composite-horizon line`、或多帧叠加让星空运动更明显、或用 `--ground-dir` 指定专门拍摄的地景照片。

## 代码结构

`scripts/align_stack.py` 只是命令行入口；实现按流水线步骤拆分在 `scripts/astro/`（`registration` / `stacking` / `flat` / `stretch` / `neutralize` / `star_reduce` / `composite` / `horizon` / `render` / `pipeline` 等模块）。

## 注意

- 输入 CR2 放在仓库外（如 `D:\pic\...`），不要拷进仓库。
- `out/`、`*.tif`、`.venv/` 已 git 忽略，勿提交。
- OpenCV `cv2.imwrite` 需要 **BGR**；写 TIFF 用脚本内 `write_tiff16`，否则红蓝通道会互换。
