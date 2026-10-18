# Python vs C++ 星野对齐/叠加实现对比报告

> 结论先行：**两者功能等价、画质几乎一致，性能基本打平（互有胜负，差异在 ±15% 的
> 运行间波动范围内）。但 C++ 版本在代码量、依赖、构建与维护成本上明显更重。**
> 因此后续**以 Python (`scripts/align_stack.py`) 为唯一主要开发路径**；`cpp/` 仅作为
> 参考实现/备份保留，不再作为默认方案投入维护。

- 对比对象
  - Python：`scripts/align_stack.py`
  - C++：`cpp/`（自研算法 + 内嵌 LibRaw，无 OpenCV）
- 测试数据：`D:\pic\DCIM-2026\08-shangduhu-2`，`IMG_5994..6012`（缺 6001，共 18 帧，全分辨率 5496×3670）
- 参考帧：`6003`

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| OS | Windows |
| CPU | 72 逻辑核心（`multiprocessing.cpu_count()==72`） |
| 内存 | 137 GB |
| 输入 | 18 张 CR2（约 450 MB，位于 D:） |
| Python | 仓库根 `.venv`（numpy 2.5.4 / opencv-python 5.0.0.93 / rawpy 0.27.1） |
| C++ | MSVC 14.44 (VS2022 BuildTools) + CMake + Ninja；Release `/O2 /fp:fast /openmp` |
| 计时 | 独立监视进程，`wall = 从启动到退出`；每个配置连跑 2 次 |

说明：本机核数很多（72），Python(OpenCV) 与 C++(LibRaw) 都会大量起线程；加上 D: 磁盘
缓存命中差异，导致运行间波动较大。下面给出两次实测值与区间，而非单点。

---

## 2. 功能与正确性

两者算法一致：半分辨率灰度 → 跨帧高分位静态图减除（去热噪/固定图案）→ 高通 →
星点连通域检测 → RANSAC 相似变换（旋转+平移）→ 星点最近邻迭代精修到亚像素 →
全分辨率线性 RGB 双线性 warp → mean/median 叠加 → 自适应 MTF 拉伸 → 写 16-bit TIFF。

- 注册结果：**两者 18/18 全部成功注册**，旋转 −0.44°…+0.44°，平移最大约 17 px（全分辨率），与理论一致。
- 逐帧变换一致性（C++ vs Python，18 帧）：

| 指标 | 数值 |
|---|---|
| 旋转差 最大 | 0.0104° |
| 平移差 最大 / 平均 | 1.80 px / 0.44 px |
| 匹配得分差 | < 0.02 |

- 叠加图一致性（整幅 5496×3670×3，uint16）：

| 方法 | 相关系数 | 平均绝对差 |
|---|---|---|
| mean | 0.99718 | 9.95 / 65535（0.0152%） |
| median | 0.99305 | 21.38 / 65535（0.0326%） |

> 差异主要来自边界外插、双线性/最近邻权重、LibRaw 与 rawpy 解码细节的微小不同，
> 属正常范围，肉眼不可辨。

**正确性结论：等价。**

---

## 3. 性能

### 3.1 运行时间（18 帧，秒；越小越好）

| 方法 | Python 第1/2次 | Python 均值 | C++ 第1/2次 | C++ 均值 | 结论 |
|---|---|---|---|---|---|
| mean | 174.9 / 169.8 | **172.4** | 195.9 / 185.2 | 190.6 | Python 略快（约 −10%） |
| median | 212.1 / 232.5 | 222.3 | 191.0 / 217.2 | **204.1** | C++ 略快（约 −8%） |

- 补充观测：C++ 首轮曾出现 141 s(mean)/153 s(median)，说明**波动很大**，单点成绩不可靠。
- 结论：**没有量级差异**。median 因 C++ 走内存中位数、Python 走磁盘 memmap，C++ 稍占优；
  mean 因 OpenCV 的 `warpAffine` 是 SIMD 优化、Python 端 numpy 向量化，C++ 标量 warp 反而略慢。

### 3.2 峰值内存（整进程树 RSS / Working Set，MB；越小越好）

| 方法 | Python | C++ | 结论 |
|---|---|---|---|
| mean | ~3125 | ~1913 | C++ 约省 39% |
| median | ~5121 | ~3374 | C++ 约省 34% |

- C++ 优势来自：全内存运算、无解释器/中间副本。
- Python median 用 `out/_aligned_uint16.dat`（约 2.18 GB 磁盘 memmap）+ 128 行分块，
  仍会触碰较多页，故 RSS 更高；mean 为流式累加。
- 注意：本机内存充足（137 GB），该差异**不影响实际使用**。

---

## 4. 代码量与复杂度

| 项 | Python | C++ |
|---|---|---|
| 手写代码 | **413 行**（单文件） | 1125 行（`cpp/src`，6 源 + 5 头） |
| 构建脚本 | 无 | 78 行（CMakeLists + build.bat） |
| 第三方依赖 | numpy / opencv-python / rawpy（pip 一行装） | 内嵌 LibRaw：**114 文件 / 约 62300 行**（2.4 MB） |
| 关键算法来源 | OpenCV 现成（ORB、RANSAC、warpAffine） | **全部手写** |
| 产物 | 脚本 + venv | `output/astro_align.exe`（1.19 MB） |

C++ 版必须自己实现：高斯高通、连通域星点检测、相似变换 RANSAC、亚像素精修、
双线性 warp、16-bit TIFF 写入、以及**从不压缩的 zlib/PNG 写入器**（连 PNG 都要手写
CRC32/Adler32/分块）。这些都是 OpenCV 一行调用的功能——**是风险与维护点的来源**，
尽管本次已用数值对比验证等价。

---

## 5. 构建与部署

| 项 | Python | C++ |
|---|---|---|
| 环境准备 | `python -m venv .venv` + `pip install -r requirements.txt` | 装 CMake + Ninja + MSVC BuildTools；内嵌并配置 LibRaw |
| 干净构建耗时 | 无（直接运行） | 约 **28 s**（本机 72 核；单机内核少会更久） |
| 首次落地成本 | 低 | 高（本次耗大量时间用于定位工具链、拉取/裁剪 LibRaw、修构建问题） |
| 运行 | `./.venv/Scripts/python.exe scripts/align_stack.py ...` | `output\astro_align.exe ...`（参数完全一致） |
| 可移植性 | 任何有 Python 的机器 | 需按平台重建（CMake 可移植，但无预编译产物） |
| 分发 | 带 venv 或现场 pip | 单 exe（但绑定平台/运行时） |

---

## 6. 开发与维护

- **迭代速度**：Python 直接改一行即跑；C++ 每次改算法需重编（增量约数秒~数十秒），
  且要小心指针/内存/OpenMP 竞态（本次已踩到一次 median 的 `colbuf` 共享竞态）。
- **调参/试验**：检测阈值、RANSAC 参数、拉伸参数在 Python 里改起来快得多。
- **可读性/可审计性**：413 行 vs 1125 行 + 62k 行第三方，Python 明显更易 review。
- **功能扩展**：加新方法（如 kappa-sigma、坏帧剔除策略、drizzle）在 Python 中成本低。

---

## 7. 结论与决策

| 维度 | 胜出方 |
|---|---|
| 画质/正确性 | 平 |
| 运行速度 | 平（±15%，mean Python 略快，median C++ 略快） |
| 内存 | C++ |
| 代码量/依赖 | **Python** |
| 构建/部署 | **Python** |
| 开发/维护/扩展 | **Python** |

**决策：后续以 Python 为唯一主要开发路径**（见 `AGENTS.md`）。`cpp/` 保留为参考实现，
不主动维护；如需极致单文件分发或免 Python 环境部署时再考虑。

---

## 附录 A：复现命令

```bat
:: Python
.venv\Scripts\python.exe scripts\align_stack.py --dir "D:\pic\DCIM-2026\08-shangduhu-2" ^
    --start 5994 --end 6012 --ref 6003 --method median --out out_py_median

:: C++
cpp\build.bat Release
output\astro_align.exe --dir "D:\pic\DCIM-2026\08-shangduhu-2" ^
    --start 5994 --end 6012 --ref 6003 --method median --out out_cpp_median
```

峰值内存/计时通过轮询整个进程树的 RSS 获得（Windows venv 的 `python.exe` 是重定向
启动器，真正的解释器是其子进程，需按进程树统计）。

## 附录 B：原始测量数据

| 配置 | 轮次 | wall (s) | 峰值内存 (MB) |
|---|---|---|---|
| Python mean | 1 | 174.9 | 3121 |
| Python mean | 2 | 169.8 | 3131 |
| Python median | 1 | 212.1 | 5136 |
| Python median | 2 | 232.5 | 5106 |
| C++ mean | 1 | 195.9 | 1912 |
| C++ mean | 2 | 185.2 | 1913 |
| C++ median | 1 | 191.0 | 3374 |
| C++ median | 2 | 217.2 | 3374 |
| C++ mean（更早一次） | — | 141.2 | — |
| C++ median（更早一次） | — | 153.3 | — |
