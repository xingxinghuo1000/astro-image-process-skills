---
name: astro-image-process-skills
description: >-
  Use when aligning, registering and stacking astrophotography / star-field
  frames shot on a fixed tripod or a sky-tracked rig, e.g. a Canon CR2/RAW
  night-sky sequence.
  Registers the moving sky, rejects the static foreground and hot-pixel noise,
  then outputs a brightness-stretched stacked image. Trigger keywords: 星野,
  星空, 银河, 对齐, 叠加, star field alignment, astro stacking, CR2 alignment.
---

# Astro Image Process Skills

工具集：把固定三脚架或星空跟踪拍摄的连续星野照片（Canon CR2/RAW）**对齐 + 叠加 + 自动拉伸**。

## 为什么难，以及本工具怎么解决

- 星空在移动，地面景物静止 → 全图相关会被前景/噪声锁死。工具**只对齐星空**（单应变换：旋转 + 平移 + 轻微投影），前景会有轻微拖影。
- CMOS 热噪点在每帧同一像素位置，会被误当作特征匹配。工具先构建**静态图案图**（逐像素高分位数）并扣除，只留下会移动的真实星点。**星空跟踪**时星点本身在各帧几乎不动，分位图会把星点一起删掉：流程会检测这种情况并自动跳过静态扣除、改开 `--hot-post`。
- 用 **ORB + RANSAC** 求单应变换，再用**星点质心**迭代精修到亚像素；自动剔除配准失败的帧。逐帧配准默认多线程并行（`--workers`）。
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

- `--method median` — **强约束：星野叠加必须用中值（median），不要用 mean。** 相机在跟踪/赤道仪上时画幅会旋转，而热噪点固定在传感器像素上，对齐后它相对天空移动；mean 会把每个热噪点摊成 `热值/N` 的彩色**短条纹**，低于任何阈值、无法消除；median 直接把它当离群值剔除，从根上消掉这些红/绿/蓝短线。默认即 `median`，`mean` 仅用于特殊场景且会打印警告。
- `--hot-pre` / `--hot-post` — **热噪点去除**（单帧、逐通道）：把拉伸后仍可见的**纯红/绿/蓝点**（CMOS 热噪点）检测并修复。
  - 判据（逐通道，在"通道 − 局部中值"残差上）：过 MAD/绝对阈值、是**孤立的局部极大**（8 邻域无更亮者，排除星点边缘）、且**被单一通道主导**（排除白色星点）。
  - 修复：热噪点经去马赛克后其实是一个 ~3–5px 的彩色斑（不是单像素），所以以峰值点为中心、半径 `--hot-grow`（默认 **3**）内的**仍彩色且偏高**的像素一起替换；替换值取"双 5×5 中值"背景（能忽略整块斑，又不像形态学开运算那样把附近的星点也抹掉）。
  - `--hot-pre` 在**叠加前逐帧**修（从源头去除，和跟踪无关，最彻底但慢）；`--hot-post` 在**叠加后的线性主图**上修（快）。**先保证 `--method median`，多数彩色短线纹就没了；旋转中心附近几乎不动的坏点再开 `--hot-post`。** 推荐：`--method median --hot-post`（需要逐帧清理再用 `--hot-pre`）。
  - 其余可调：`--hot-sigma`（MAD 倍数，默认 6）、`--hot-abs`（绝对阈值，默认 0.02）、`--hot-ksize`（背景中值窗口，默认 5）、`--hot-color`（单通道主导倍数，默认 2）、`--hot-neigh`（局部极大倍数，默认 1.0）。输出 `hot_compare_pre.jpg` / `hot_compare_post.jpg`（前/后对比图，**交给用户目视判断**）和 `hot_metrics.json`。
- `--ref N` — 参考帧编号，默认取序列中间帧。
- `--workers N` — 逐帧配准的并行线程数（默认 `0` = 每个 CPU 核一个，最多到帧数）。配准阶段每个线程独立建 ORB，并行期间 OpenCV 内部线程数降到 1，避免 72 核机器上过度订阅。
- `--io-workers N` — CR2 解码并行度（默认 `0` = 8）：灰度预览和全分辨率叠加两段都并行解码，队列只保留约 `2N` 帧，内存约 `N×120 MB`。
- `--stack-workers N` — 中值叠加的行带并行度（默认 `0` = 8）。每带按约 1 GB 自适应切分；中值排序本身也吃临时内存，RAM 预算约 `2~3×N GB`。先做普通 `median`，只对边缘有无效像素的点补算 `nanmedian`，比整幅 `nanmedian` 快约 5 倍。
- `--render-workers N` — 各阶段 TIFF/JPEG 并行写出（默认 `0` = 4），同时生成多张半尺寸预览。
- `--no-static` — 关闭静态图样（热噪点/固定图案）扣除。若相机在**跟踪星空**（赤道仪/星野赤道仪），星点在各帧几乎不动，按分位数生成的静态图会**把星点本身当成静态图案删掉**，导致 ORB 只剩噪声；流程会自动检测这种情况并跳过静态扣除、同时启用 `--hot-post`，一般不需要手动加。
- `--stretch-bg 0.20` — 拉伸后背景目标亮度，越大越亮。
- `--save-frames` — 另存每帧对齐后的全分辨率 TIFF（较占空间）。
- `--exclude 6007,6012` — 剔除个别帧。
- `--neutralize` — **背景中性化**（类 PixInsight `BackgroundNeutralization`）：按通道测量天空背景并缩放到中性灰，消除偏色（如光害导致的紫/绿底）。作用于拉伸后的图像。
- `--wb` — **自动白平衡**（中性化之后）：用天空最亮像素（whitepatch，默认）、全天空均值（grayworld）或两者的几何平均（mixed）测量通道增益，再按 `--wb-strength`（默认 0.6，对数域混合，0 关闭 / 1 全量）施加**保持亮度**的增益。可选 `--wb-mode whitepatch|grayworld|mixed`、`--wb-gains "R,G,B"`（手动增益，覆盖自动测量）。会输出 `wb_compare.jpg`（前/后并排，标出增益），**必须用视觉能力目视判断**：白平衡过强会把绿色气辉洗成蓝色、星点发紫，过弱则保留黄绿底；不够/过头就调整 `--wb-strength` 或改用 `--wb-mode mixed` 重新出图。
- `--flat` — **自动平场 / 背景梯度去除**（类 PixInsight `AutomaticBackgroundExtractor` ABE）：把图像分成网格、每格取低分位数估计天空背景，拟合一张低阶二维多项式背景面并按通道扣除（只扣除梯度的"形状"，保留暗天空基座，避免把背景压到 0）。**作用于拉伸前的线性图**：梯度在线性域是加性的，拟合/扣除才有物理意义；放到拉伸后会失真并干扰自动拉伸的背景定位。可调 `--flat-mode subtract|divide`、`--flat-bins`、`--flat-degree`、`--flat-pct`、`--flat-strength`。
  - **平场是否采用由用户决定**：只要用了 `--flat`，就一定会输出对比图 `flat_compare.jpg`（左：不平场，"成品"观感；右：平场后），供用户判断要不要保留。想一次比较多个强度，用 `--flat-sweep 0.4,0.5,0.6,0.7`，对比图里会平铺"不平场 + 各强度"，选好后再用 `--flat-strength` 定值正式出图。
  - ⚠️ **外圈亮环**：若画面被银河/大范围辉光铺满，低阶背景模型分不清"天空背景"和"目标本身的辉光"，满强度扣除会把中心压暗、边缘相对变亮，出现一圈亮环。此时把 `--flat-strength` 调到 ~0.3~0.6（默认 1.0）即可在保留梯度去除的同时消除亮环。
- `--star-reduce PX` — **缩星**（类 PixInsight `StarReduction`）：高通星点掩膜下的形态学腐蚀，把星点半径缩小约 PX 像素，背景与银河不受影响，让星云/银河更突出。**默认 1（轻微）**，`0` 关闭；想要更狠再用 2~3（2 已经比较激进，会明显抹掉细小星点）。
- `--composite` — **天地合成**（可选）：把「对齐到星空的天景」和「直接把多张照片叠加（不配准）得到的地景背景」在**天地分界线**处拼接。天景里星空锐利、前景被拖影；地景里前景锐利、星空拖出星轨。
  - **地景识别：用大模型识图（主路径）** — 让具备视觉能力的大模型（本 agent 本身即可）看一张预览（如 `ground_stack_preview.jpg`，前景锐利）后，直接判断"哪里是地景/前景"（树、建筑、地形、地平线），产出一张**地景掩膜 PNG**（**白色 = 地景/前景**，黑色 = 天空），再用 `--ground-mask PATH` 传入。流程会按 `--composite-feather` 羽化后拼接，掩膜不需要像素级完美——覆盖住前景主体、允许树冠镂空/交界粗糙即可。
    - 实践：可用亮度（前景比天空暗）+ 时间方差（前景静止）+ 空间先验（地景在下部/下部边缘）组合出候选，再由大模型目视核对/修整；0812 序列产出的掩膜约覆盖画面 25% 地景（底部山脊），明显优于纯方差自动检测。
  - `--ground-mask PATH` — 大模型识别的地景掩膜（灰度 PNG，白 = 地景），**优先级最高**，覆盖下面的自动检测。
  - **自动检测（未给 `--ground-mask` 时的回退）** — 用未配准帧的**时间方差**（天空在动、地景静止）取每列下包络估计分界线。可选微调：`--composite-horizon mask|line`（`mask` 沿树线/地形走；`line` 用稳健直线）、`--composite-thresh T`（天空证据阈值，默认 0.05）、`--composite-feather PX`（羽化带宽度，全分辨率像素，默认 25，也用于羽化模型掩膜）。
  - `--ground-dir` / `--ground-start` / `--ground-end` — 用另一组专门的地景照片当地景层（默认用输入帧本身、不配准叠加）。

## 输出（在 `--out` 目录）

> **每个阶段都会输出中间结果**：`stageN_<name>.tif`（16-bit TIFF）+ 同名 `_preview.jpg`（半尺寸预览），`Num`（即 `N`）就是该阶段的序号，按实际启用的步骤递增；并把**所有阶段拼接成一张对比图** `stages_montage.jpg`（线性阶段会自动拉伸后展示，便于和后续阶段比较）。

| 文件 | 说明 |
| --- | --- |
| `stageN_<name>.tif` + `_preview.jpg` | **每个处理阶段的中间结果**（TIFF + 预览），`N` 为该阶段序号。如 `stage1_stacked` / `stage2_hotfix` / `stage3_composite` / `stage4_flat` / `stage5_stretched` / `stage6_neutralized` / `stage7_whitebalanced` / `stage8_star_reduced` |
| `stages_montage.jpg` | **各阶段成果多图拼接对比图**（全部阶段并排，带标签） |
| `merge_final_<start>_<end>.tif` | **最终成品**：16-bit TIFF（完整分辨率）。`<start>`/`<end>` 为**合并序列的起始/结束帧号**，如 `merge_final_5994_6012.tif` |
| `merge_final_<start>_<end>.jpg` | **最终成品**：JPEG（完整分辨率） |
| `stack_<method>.tif` | 对齐叠加后的**线性** 16-bit 主图（供后期） |
| `stack_preview.jpg` | 线性主图的半尺寸预览 |
| `transforms.json` | 每帧的单应变换、内点数、评分 |
| `timing.json` | 本次运行的分阶段耗时、并行线程数，用于定位下一轮性能瓶颈 |
| `star_mask.png` | 星点掩膜（使用 `--star-reduce` 时输出，便于检查） |
| `flat_compare.jpg` | 平场对比图（使用 `--flat` 时输出）：不平场 vs 平场（可含 `--flat-sweep` 强度扫描），**交给用户目视决定是否采用/选用哪个强度** |
| `wb_compare.jpg` | 白平衡对比图（使用 `--wb` 时输出）：中性化后 vs 白平衡后，标注实际增益，**交给用户目视判断强弱** |
| `hot_compare_pre.jpg` / `hot_compare_post.jpg` | 热噪点去除前/后对比图（使用 `--hot-pre` / `--hot-post` 时输出）：整幅缩略 + 最强彩点处的放大裁切，**交给用户目视判断效果** |
| `hot_metrics.json` | 热噪点审计：叠加主图上检出的坏点数量、叠加前替换的通道像素数 |
| `ground_stack.tif` + `_preview.jpg` | 地景层（使用 `--composite` 时输出）：固定三脚架为不配准叠加的帧（前景锐利、星空拖轨），星空跟踪为参考帧 |
| `sky_mask.png` / `sky_ground_overlay.jpg` | 天地分界线掩膜 / 叠加了分界线的天景预览（使用 `--composite` 时输出，便于检查分割） |
| `aligned/IMG_xxxx.tif` | 每帧对齐结果（需 `--save-frames`） |

处理阶段顺序：**叠加 → 热噪点修复 → 天地合成 → 自动平场 → 拉伸 → 中性化 → 白平衡 → 缩星**（热噪点修复/合成/平场/中性化/白平衡/缩星均可关闭；`--hot-pre` 则作用于叠加前的每一帧）。

**最终出图**：成品同时输出**完整分辨率的 TIFF 和 JPEG**，命名为 `merge_final_<start>_<end>.tif` / `.jpg`（`<start>`/`<end>` 是合并序列的起始/结束帧号），可直接交付；其余中间结果一律以 `stageN_` 开头。

### 性能（实测：124 帧 4744–4869，72 核 / 137 GB）

完整流程（含平场、中性化、白平衡、缩星、天地合成）**363 s**，主要环节：

| 环节 | 时间 | 并行方式 |
| --- | ---: | --- |
| 中值叠加 | 116 s | 8 个行带；普通 median + 边缘补算 nanmedian |
| 逐帧解码/对齐/写盘 | 96 s | 8 个解码线程，预取队列 |
| 灰度预览解码 | 27 s | 8 个解码线程 |
| 逐帧配准 | 29 s | 72 线程 ORB |
| 热噪点检测/修复 | 3.8 s* | 3 通道并行；C++ 膨胀求 8 邻域 |
| 自动平场 | 3.3 s* | 3 通道并行 |
| 各阶段写盘 | 17 s | 4 线程 |

\* 微基准单图耗时；流水线里的 `hotfix`/`flat_compare` 还包含对比图生成。实测热噪点检测+修复由 19.6 s 降到 3.8 s，平场由 8.8 s 降到 3.3 s。单线程中值按 256 行基准外推约 19 min，优化后 116 s。`timing.json` 会记录每次运行的实际分项耗时。

## 平场工作流（对比 → 用户决策）

平场是"判断题"，不要替用户拍板。推荐流程：

1. 先跑一次 **带强度扫描**：`--flat --flat-sweep 0.4,0.5,0.6,0.7`；
2. 把生成的 `flat_compare.jpg` 给用户看（平铺了"不平场 + 各强度"的成品观感），让用户决定**是否要平场、用哪个强度**；
3. 用户定好后正式出图：`--flat --flat-strength <S>`（不加 `--flat` 即不平场）。

若画面被银河铺满、外圈出现亮环，就是强度偏大，调低 `--flat-strength`。

## 白平衡工作流（目视判断 → 调整）

白平衡是"判断题"，自动增益只是起点，必须看 `wb_compare.jpg`：

1. 先跑 `--wb --wb-mode whitepatch --wb-strength 0.6`（默认）；
2. 看 `wb_compare.jpg`：星点应接近白/淡蓝/淡黄，背景不应整体发紫或发蓝；若背景仍明显偏绿/黄（气辉、光害），可改 `--wb-mode mixed` 或 `grayworld`，或把 `--wb-strength` 调到 0.7~0.8；若星点发紫、天空过蓝，说明过头，降到 0.4~0.5 或改回 `whitepatch`；
3. 用选定的 `--wb-strength`/`--wb-mode`（或直接用 `--wb-gains "R,G,B"`）重新出图，再核对一次 `wb_compare.jpg`。

## 天地合成（`--composite`）

固定三脚架拍摄时星空在转、前景静止；星空跟踪时星点不动、地景随跟踪移动：

- **天景**：帧对齐到星空后叠加 → 星空锐利，前景被拖影。
- **地景**：固定三脚架时帧**不配准**直接叠加（前景锐利、星空拖成星轨）；星空跟踪时地景也在动，只取**参考帧**的地景，避免叠加糊掉。
- 用**时间方差**得到"天空"证据图（未配准的帧在天空区域变化大，在地面无变化），取每列的**下包络**作为天地分界线：能跨过天极附近的低运动"空洞"，也能沿锯齿状树线走；最后在分界线羽化带内把两层融合。该步骤作用于线性图，之后的平场/拉伸/中性化/白平衡/缩星都作用在合成图上。

分界线是否符合预期，看 `sky_ground_overlay.jpg`（青色 = 掩膜边界，绿线 = 直线拟合）。若不满意：调 `--composite-thresh`、改 `--composite-horizon line`、或多帧叠加让星空运动更明显、或用 `--ground-dir` 指定专门拍摄的地景照片。

## 代码结构

`scripts/align_stack.py` 只是命令行入口；实现按流水线步骤拆分在 `scripts/astro/`（`registration` / `stacking` / `hotpixels` / `flat` / `stretch` / `neutralize` / `whitebalance` / `star_reduce` / `composite` / `horizon` / `render` / `pipeline` 等模块）。

## 注意

- **叠加方式强制用 `median`，禁止用 `mean`**（见上）。若成品里出现红/绿/蓝的短线/短弧，几乎都是误用了 `mean`：热噪点被摊成 `热值/N` 的彩色条纹。改用 `median` 即可消除，通常无需其他后处理；个别坏点（旋转中心附近）再用 `--hot-post`。
- 输入 CR2 放在仓库外（如 `D:\pic\...`），不要拷进仓库。
- `out/`、`*.tif`、`.venv/` 已 git 忽略，勿提交。
- OpenCV `cv2.imwrite` 需要 **BGR**；写 TIFF 用脚本内 `write_tiff16`，否则红蓝通道会互换。
