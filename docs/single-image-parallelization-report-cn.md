# 单图并行化改造报告

**语言：** 简体中文 | [English](single-image-parallelization-report.md)

> 结论先行：这次改造分两轮完成。第一轮把 **124 帧、5496×3670** 的流程从“逐帧/
> 逐通道串行”推进到 **帧间流水线 + 单图内并行**，总耗时 **362.7 s**，约 **2.9 s/帧**；
> 旧流程在 160 帧基线上约 **17.4 s/帧**，按每帧归一化约 **6×**。第二轮针对中值叠加
> 做“小 band + 高并发”：band 目标从约 1 GB 降到 16 MB，worker 从 16 提到 CPU 核数，
> 中值 **115.8 s → 55.3 s**，stack 峰值工作集约 **63 GB → 18 GB**；全流程 **343.9 s**
> （总时间受磁盘波动影响，仅作参考）。两轮最终 TIFF 的 md5 完全一致。

![并行化总览](images/parallelization-map.png)

*图 1：并行任务分布与内存边界。*

---

## 1. 改造范围与测试数据

| 项 | 值 |
| --- | --- |
| 输入 | `D:\pic\DCIM-2023\0812-liuxingyu\02\serial`，`IMG_4744..4869` |
| 帧数 | 124（`4746/4747` 缺失） |
| 分辨率 | 5496×3670，全分辨率线性 RGB；配准在 2748×1835 灰度上做 |
| 参考帧 | `IMG_4780` |
| 场景 | 星空跟踪序列；高通相关 0.89，自动跳过静态模式扣除 |
| 天地合成 | `--ground-mask`，地面占半分辨率画面 24.93% |
| 叠加 | `--method median`，124/124 帧全部通过配准校验 |
| 硬件 | 72 逻辑核心、137 GB 内存、Windows |
| 输出 | `out_4744_4869/` |
| 最终 TIFF | `merge_final_4744_4869.tif` |

> 这里把“并行化”分成两层：
>
> - **帧间并行 / 流水线**：不同 CR2 的解码、不同帧的配准可以同时做。
> - **单图内并行**：在同一张图内部，按通道、行带或输出面板拆分任务。
>
> 前者决定吞吐，后者解决“单张大图本身就算得慢”的问题。两者不能混为一谈。

---

## 2. 改造原则

1. **只并行独立工作单元。** 中值的行带之间、RGB 通道之间、不同输出文件之间没有写冲突；
   这是线程并行的前提。
2. **优先线程，不优先进程。** NumPy 的排序/分区内核和 OpenCV 的滤波会释放 GIL；线程
   共享 `numpy` 数组和 memmap，不需要把 15 GB 的中间结果复制到每个进程。
3. **限制在途数据量。** 预取队列只保留约 `2N` 帧；中值每个行带默认目标 16 MB
   （最少 1 行），避免 72 个全图任务同时把内存打满。
4. **保持输出确定性。** 行带写入互不重叠；通道写入互不重叠；阶段文件各自独立。
   线程完成顺序不影响最终像素。
5. **先保证数值正确，再谈并行。** 全有效像素仍走 `np.median`；只有被 warp 暴露出来的
   无效边界才走 `np.nanmedian`。

---

## 3. 总览：并行发生在哪里

见开头的总览图。下表按位置拆开：

| 位置 | 并行单元 | 默认并行度 | 共享/同步方式 |
| --- | --- | --- | --- |
| 灰度预览解码 | 每帧一个任务 | 8 | 线程池，结果按帧序收集 |
| 配准 | 每帧一个任务 | 最多 72（到帧数） | `ThreadPoolExecutor`，每任务独立 ORB |
| 全分辨率解码 | 每帧一个预取任务 | 8 | 有界 future 队列，约 `2N` 帧 |
| 中值叠加 | 输出行带 | 8 | 每带读 memmap，写独立行段 |
| 热噪点修复 | R/G/B 通道 | 3 | 每通道写独立通道切片 |
| 自动平场 | R/G/B 通道 | 3 | 每通道独立拟合/回写 |
| 阶段 TIFF/JPEG | 每个阶段 | 4 | 每任务写自己的文件 |
| 阶段对比图面板 | 每个面板 | 4 | 每个面板独立生成，最后拼接 |

**关键点：** 单图内并行不是把整幅图复制给多个进程，而是让每个线程只处理一个通道或
一个行带。这样既利用多核，又把峰值内存压在可控范围内。

---

## 4. 单图内并行改造（核心）

### 4.1 中值叠加：从串行 128 行块到自适应小行带

改造前，`stacking.py` 是固定 `step=128` 的串行循环：

```python
# 改造前：scripts/astro/stacking.py
def median_stack(store, H, W, step=128):
    stack = np.empty((H, W, 3), np.float32)
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        stack[r0:r1] = np.median(block, axis=0)
    return stack
```

改造后，行带大小按“约 16 MB/带”自适应（最少 1 行），行带之间用线程池并行；同时加入
快速中值路径：

```python
# 改造后：scripts/astro/stacking.py
def median_stack(store, H, W, step=None, workers=1, band_mb=16):
    n = store.shape[0]
    if step is None:
        target = max(1.0, float(band_mb)) * 1e6
        step = int(target / max(1, n * W * 3 * 4))
        step = max(1, min(256, step))
    bands = [(r0, min(H, r0 + step)) for r0 in range(0, H, step)]
    stack = np.empty((H, W, 3), np.float32)

    def one(band):
        r0, r1 = band
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        med = np.median(block, axis=0)
        bad = block <= 0
        invalid = bad.any(axis=0)
        if invalid.any():
            flat = block.reshape(n, -1)
            badflat = bad.reshape(n, -1)
            idx = np.flatnonzero(invalid.reshape(-1))
            sub = flat[:, idx]
            sub[badflat[:, idx]] = np.nan
            med.flat[idx] = np.nanmedian(sub, axis=0)
        stack[r0:r1] = med

    workers = max(1, min(int(workers), len(bands)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, bands))
    return np.nan_to_num(stack, nan=0.0)
```

#### 核心算法差异

| 方面 | 改造前 | 改造后 |
| --- | --- | --- |
| 并行单元 | 无，固定 128 行串行 | 自适应行带，CPU 核数个线程 |
| 行带大小 | 固定 128 行 | 按 `band_mb` 推到约 16 MB/带（最少 1 行） |
| 边界无效值 | `0` 当作真实像素参与中值 | 先快速中值，仅对含 `0` 的像素用 `nanmedian` |
| 有效像素结果 | `np.median` | 仍然 `np.median`，逐像素一致 |
| 边界结果 | 可能被 0 拉黑 | 只用有效帧，不产生黑边 |
| 内存 | 一个 128 行 float32 块 | workers 个约 8–16 MB 块 + 少量边缘临时数组 |

> 为什么不是整幅 `nanmedian`？`nanmedian` 比为每个像素做 NaN 检查、掩码和分区慢得多。
> 在这类序列里，绝大多数像素在 124 帧中都有覆盖；只有 warp 后的窄边界会缺帧。
> 因此“先算全量普通中值，再只补算无效边界”是性价比最高的写法。

`np.median` 对全有效像素和旧实现完全一致；对边界像素，新实现不再把 0 当数据，而是
仅用真正覆盖该像素的帧。最终成品 TIFF 的 md5 为
`aebb14498eb2845c940e71e67cd53687`，与“整幅 `nanmedian` 参考路径”的结果一致。

![中值行带并行](images/median-bands.png)

*图 2：中值叠加的行带并行，以及“快速中值 + 边界 nanmedian”路径。*

### 4.2 热噪点：逐通道并行 + 用 dilate 取代 8 次 roll

改造前，`hotpixels.py` 的顺序是：先串行求 3 个通道的 residual，再串行检测/修复；
邻域最大值用 8 次 `np.roll`，边界会环绕。

```python
# 改造前：scripts/astro/hotpixels.py
def _neighbor_max(a):
    m = np.full_like(a, -1.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            m = np.maximum(m, np.roll(np.roll(a, dy, 0), dx, 1))
    return m

for c in range(3):
    resid[..., c] = ...
for c in range(3):
    ...
```

改造后，三个通道互相独立，直接并行；邻域最大值用“中心为 0 的 3×3 核 + `cv2.dilate`”：

```python
# 改造后：scripts/astro/hotpixels.py
_NEIGH_KERNEL = np.ones((3, 3), np.uint8)
_NEIGH_KERNEL[1, 1] = 0

def _neighbor_max(a):
    return cv2.dilate(a.astype(np.float32), _NEIGH_KERNEL)

def one(c):
    ...
    cand &= r > neigh_ratio * np.maximum(_neighbor_max(r), 0.0)
    ...
```

#### 核心算法差异

- **并行层**：R/G/B 三个通道是独立任务，默认 3 线程。
- **邻域实现**：`np.roll` 的 8 次整图搬移变成一次 `cv2.dilate`；内部像素的“8 邻域最大值”
  语义不变。
- **边界语义**：`np.roll` 会环绕，`dilate` 默认不环绕。图像最外一圈的行为因此更合理，
  不会把对侧边缘的亮像素当成邻居。
- **内存**：显式索引 `(c+1)%3, (c+2)%3` 代替 `np.delete`，少一次整图拷贝。
- **检测准则不变**：仍是 MAD 阈值 + 绝对阈值 + 8 邻域极大 + 通道支配；测试帧检测到
  8233 个热像素（`hot_metrics.json`）。

### 4.3 自动平场：每个颜色通道独立拟合

平场的多项式拟合本来就是逐通道独立的。改造前是 `for c in range(3)` 串行；改造后把
`one(c)` 放进线程池，三个通道同时算：

```python
# 改造后：scripts/astro/flat.py（结构）
def one(c):
    ...  # tile 低分位 -> 最小二乘 -> 2D 多项式 -> 回写 out[..., c]
    return c, (model.min(), model.max(), model.mean())

workers = max(1, min(int(workers), 3))
if workers == 1:
    results = (one(c) for c in range(3))
else:
    ex = ThreadPoolExecutor(max_workers=3)
    results = ex.map(one, range(3))
```

这里是**执行方式变化，不是算法变化**：网格、分位数、低阶多项式、pedestal、strength
混合全部保持不变。实测输出与逐通道串行版本一致。

### 4.4 阶段图片：按阶段/面板并行写出

改造前每个 stage 串行 `write_stage`，montage 和 flat 对比图的面板也串行生成。
改造后：

- `pipeline.py` 用 `render_workers` 并行写各阶段的 TIFF + JPEG；
- `render.write_montage` 并行生成每个阶段面板；
- `render.write_flat_compare` 并行生成 no-flat / 各 strength 预览面板。

每张阶段图写自己的文件，互不加锁；montage 的面板最后按固定顺序 `hstack`，所以输出
顺序仍然确定。

---

## 5. 帧间与 IO 并行（配套改造）

### 5.1 配准：每帧一个任务

`_register_task` 把单帧的“高通 → ORB → 匹配 → RANSAC → 星点精修 → 质量校验”封装成
纯函数任务，主线程用线程池按帧分发：

```python
# 改造后：scripts/astro/pipeline.py
tasks = [(k, idx, cleaned[k], ...) for k, (idx, _) in enumerate(files)]
prev_threads = cv2.getNumThreads()
if workers > 1:
    cv2.setNumThreads(1)
try:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        reg_results = list(ex.map(_register_task, tasks))
finally:
    cv2.setNumThreads(prev_threads)
```

这里刻意把 OpenCV 内部线程数降到 1：外层已经有 72 个帧任务，如果 ORB/滤波再各自开
一套 OpenCV 线程，会发生严重的线程过度订阅。任务结束后恢复原值。

### 5.2 全分辨率解码：有界预取

改造前每帧在循环里同步 `load_linear`；解码 CR2 时 CPU 在等磁盘。改造后，主循环只
消费 future，后台线程提前解码后面的帧：

```python
# 改造后：scripts/astro/pipeline.py（结构）
pending = {}
io_limit = max(1, io_workers * 2)

def submit_frames():
    while next_submit < len(files) and len(pending) < io_limit:
        pending[next_submit] = io_pool.submit(imgio.load_linear, files[next_submit][1])
        next_submit += 1

def take_frame(k):
    return pending.pop(k).result()
```

队列上限约为 `2×io_workers` 帧。按全分辨率 RGB float32 约 240 MB/帧估算，8 个 worker
的在途数据约 4 GB，不会随着 124 帧线性增长。预取只改变“何时解码”，不改变“解码成
什么”，所以不引入数值差异。

---

## 6. 性能结果

### 6.1 端到端

| 运行 | 帧数 | 总耗时 | 每帧 |
| --- | ---: | ---: | ---: |
| 旧流程基线 | 160 | 2785 s | 17.4 s |
| 第一轮：帧间 + 单图内并行 | 124 | 362.7 s | 2.9 s |
| 第二轮：小 band + 高并发 | 124 | 343.9 s | 2.8 s |

> 帧数不同，不能直接拿总时间对比；上表用**每帧归一化**说明量级。绝对值会受磁盘
> 缓存、CPU 频率、是否首次解码 CR2 影响。第二轮总时间只降约 5%，因为同一次运行的
> frame loop 从 95.6 s 变成 134.3 s（磁盘/IO 波动）；可复现的收益主要看中值叠加和
> 内存峰值，见 6.5。

第一轮 `timing.json` 中，wall time 427.6 s，和 `total` 362.7 s 的差约 65 s，主要是
Windows 上 memmap 文件删除/落盘的收尾时间。

### 6.2 阶段耗时

![阶段耗时](images/stage-timings.png)

*图 3：第一轮 124 帧运行的阶段耗时拆解。*

| 阶段 | 秒 | 占 362.7 s（第一轮） |
| --- | ---: | ---: |
| 中值叠加 | 115.8 | 31.9% |
| 帧循环（解码等待 + warp + 累加） | 95.6 | 26.4% |
| 配准 | 29.4 | 8.1% |
| 灰度预览解码 | 27.0 | 7.4% |
| 阶段 TIFF/JPEG 写出 | 16.7 | 4.6% |
| hotfix（含对比图） | 14.5 | 4.0% |
| flat 对比面板 | 14.1 | 3.9% |
| 天地合成 | 12.7 | 3.5% |
| 最终 TIFF/JPEG 写出 | 12.1 | 3.3% |
| 其余（拉伸/中性化/WB/缩星/montage/审计） | 24.8 | 6.8% |

注意：这些阶段是**顺序发生的阶段计时**，不是互相重叠的 wall-clock 分解；`frame_loop`
里还包含等待预取 future 的时间。第二轮阶段对比见 6.5。

### 6.3 代表阶段的前后对比

| 项目 | 改造前（剖析/基线） | 改造后（第一轮/第二轮） | 说明 |
| --- | ---: | ---: | --- |
| 中值叠加（124 帧，第一轮） | 228.6 s | 115.8 s | 8 行带 + 快速中值 |
| 中值叠加（124 帧，第二轮） | 115.8 s | 55.3 s | 16 MB 小 band + 72 workers |
| 热噪点检测/修复核心 | 19.6 s | 3.8 s | 3 通道并行 + dilate |
| 自动平场 | 8.8 s | 3.3 s | 3 通道并行 |
| 阶段图片写出 | ~44 s | 16.7 s | 4 个 render workers |
| 灰度预览解码 | ~171 s | 27.0 s | 8 个 IO workers |
| 端到端 | 17.4 s/帧 | 2.9 s/帧 | 不同帧数，按每帧归一化 |

> “改造前”的数字来自本轮改造过程中的剖析/微基准，不是在同一 wall-clock 下与旧版
> 并排跑两套代码；用于判断瓶颈和量级，不应当当作严格的 benchmark 论文数据。第二轮全流程见 6.5。

### 6.4 线程数不是唯一旋钮

- **中值行带**：默认 band 目标 16 MB，workers 按 CPU 核数和 RAM 选；本机 72 核/
  69 GB 可用内存选 72。小 band 让每个 worker 的工作集从 ~1 GB 降到 ~8 MB，同时暴露更多
  并行任务。同一份 124 帧真实对齐数据上：16×1 GB 为 115.8 s，16×16 MB 为 47.9 s，
  72×16 MB 为 43.1 s；峰值工作集约 63 GB → 18 GB，输出逐位一致。band 再小会增加
  任务/排序开销，16 MB（本序列实际为 1 行/带）附近是折中。
- **CR2 解码**：自动模式按 CPU 和 RAM 选，上限 16；本机 16 worker 解码 124 帧约
  16 s，8 worker 约 27–32 s。继续加只会增加在途内存。
- **阶段写出**：自动模式按 CPU 选，上限 8；受磁盘写入和 JPEG 编码限制，8 个 worker
  足够覆盖大部分等待。
- **配准**：自动模式每个 CPU 核一个线程、最多到帧数；配准期间 OpenCV 内部线程降为
  1，避免 72 核机器上过度订阅。

---

### 6.5 第二轮性能优化：小 band + 高并发

第一轮的中值叠加用约 1 GB 的行带，worker 上限 16。这个组合的问题是：

- 每个 worker 的 `block` 是 `N × band_rows × W × 3 × 4`，1 GB 带意味着每个 worker
  要同时持有 124 帧的一整条；16 个 worker 就是十几 GB 的 float32 临时块。
- 只加 worker 不缩小 band，内存会线性上涨；实测 24/32 个 worker 在 1 GB 带下没有
  继续变快，24 个并发在连续多轮测试中还曾触发内存分配失败。
- 行带数量少，任务粒度粗，最后的尾部任务也会拖长墙钟时间。

第二轮把两个参数一起改：

1. **缩小 band**：新增 `--stack-band-mb`，默认目标 16 MB；`step` 下限从 16 行改为
   1 行。124 帧时 16 MB 目标实际落到 **1 行/带**（约 8.2 MB）。
2. **提高并发**：`--stack-workers` 默认按 CPU 核数选；本机 72 核选 72。`median_stack`
   内部再按可用内存限制 workers（预算约为可用 RAM 的 50%，每个 worker 按 3 倍 band
   估算临时空间），避免大帧数时 OOM。
3. **保持数值路径不变**：仍然先做普通 `np.median`，只对 warp 边界暴露的无效像素补
   `np.nanmedian`；小 band 只改变任务切分，不改变像素语义。

#### 测试方法

- 同一份 124 帧真实对齐数据：5496×3670，124 帧 uint16 memmap（约 15 GB）。
- 每个配置单独起进程，避免上一轮的 allocator 缓存影响内存统计。
- 用 Windows `GetProcessMemoryInfo` 记录进程峰值工作集；它包含 memmap 页缓存，所以
  绝对值高于纯 Python 堆，但不同配置之间的相对趋势有效。
- 所有配置输出 checksum 一致；`16 × 16 MB` 额外与 `stage1_stacked.tif` 做了逐位比较。

#### 结果

| 配置 | 中值耗时 | 进程峰值工作集* |
| --- | ---: | ---: |
| 16 workers × 1 GB band | 115.8 s | ~62.6 GB |
| 8 × 16 MB | 80.6 s | ~15.8 GB |
| 16 × 16 MB | 47.9 s | ~16.0 GB |
| 24 × 16 MB | 45.0 s | ~16.4 GB |
| 48 × 16 MB | 43.7 s | ~17.6 GB |
| **72 × 16 MB** | **43.1 s** | **~18.1 GB** |
| 64 × 32 MB | 44.3 s | ~24.7 GB |
| 64 × 64 MB | 52.1 s | ~36.1 GB |
| 32 × 128 MB | 56.2 s | ~36.6 GB |

* 含 memmap 页缓存，不是纯 Python 堆内存。

观察：

- **16 MB/1 行是甜点**：比 1 GB 带快约 **2.7×**，峰值工作集约降 **71%**。
- **不是 band 越小越好**：32/64 MB 反而比 16 MB 慢或更占内存；1 行已经是最小粒度，
  再小没有空间。
- **72 workers 最好，48 很接近**：72×16 MB 为 43.1 s，48×16 MB 为 43.7 s；8 核机器
  自动选 8×16 MB 也有 80.6 s，明显好于 8×1 GB 的 108.3 s。
- 小 band 的收益来自更好的缓存局部性、更小的 `bad`/`nanmedian` 临时数组，以及更多
  可并行的任务；代价是任务数从 31 个增加到 3670 个，Python/线程调度开销上升，但
  在这个分辨率下仍被并行收益盖过。

#### 全流程验证

用第二轮默认参数重跑同一套 124 帧流程：

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2023\0812-liuxingyu\02\serial" \
    --start 4744 --end 4869 --ref 4780 --method median \
    --hot-post --flat --flat-strength 0.5 --flat-sweep 0.4,0.5,0.6,0.7 \
    --neutralize --wb --star-reduce 1 \
    --composite --ground-mask out_4744_4869/ground_mask.png \
    --out out_4744_4869
```

| 项目 | 第一轮 | 第二轮 |
| --- | ---: | ---: |
| 中值叠加 | 115.8 s | 55.3 s |
| 灰度预览解码 | 27.0 s | 16.8 s |
| 配准 | 29.4 s | 35.2 s |
| frame loop | 95.6 s | 134.3 s |
| 阶段写出 | 16.7 s | 16.5 s |
| 总耗时 | 362.7 s | 343.9 s |

最终产物：

- `stack_median.tif` md5：`cdf95e2881aef92c34e87d10d31b0b5b`（两轮一致）
- `merge_final_4744_4869.tif` md5：`aebb14498eb2845c940e71e67cd53687`（两轮一致）

全流程总时间只降约 5%，主要因为 frame loop 受磁盘/IO 波动影响变大；中值叠加本身
从 115.8 s 降到 55.3 s，这是第二轮最稳定、可复现的收益。

#### 结论与边界

- 最终默认：`--stack-band-mb 0` → 16 MB；`--stack-workers 0` → CPU 核数（RAM 限制）。
- 这仍然是**精确中值**，不是近似分位数。
- 对极大帧数，`step` 的 1 行下限会让单带重新随 N 增长；此时 `median_stack` 的 RAM
  上限会自动减少 workers。若要在任意 N 下都保持 O(1) 于 N 的内存，需要再上多趟
  radix/histogram select；那是下一步，不是本轮默认路径。

---

## 7. 正确性验证

| 检查 | 结果 |
| --- | --- |
| 配准通过 | 124/124 |
| 最终 TIFF md5 | `aebb14498eb2845c940e71e67cd53687` |
| stack md5 | `cdf95e2881aef92c34e87d10d31b0b5b` |
| 中值快路径 vs 整幅 `nanmedian` 参考 | 最终 TIFF 一致 |
| 热噪点测试帧 | 8233 个热像素（`hot_metrics.json`） |
| 平场 | 与逐通道串行版本一致 |
| 冒烟测试 | 4744–4760 小段完整跑通（含合成/平场/WB/缩星） |

`out_4744_4869/` 中的成品和阶段图已人工查看：天地合成、平场对比、白平衡对比、阶段
拼接图均正常。

---

## 8. 调参与内存估算

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2023\0812-liuxingyu\02\serial" \
    --start 4744 --end 4869 --ref 4780 --method median \
    --hot-post --flat --flat-strength 0.5 --flat-sweep 0.4,0.5,0.6,0.7 \
    --neutralize --wb --star-reduce 1 \
    --composite --ground-mask out_4744_4869/ground_mask.png \
    --io-workers 8 --stack-workers 8 --render-workers 4 --workers 72 \
    --out out_4744_4869
```

> 上面是历史基准命令，显式固定了 8/8/4；当前默认已改为按 CPU 核数和可用 RAM 自动选择。

| 参数 | 默认/建议 | 作用 |
| --- | --- | --- |
| `--workers` | 0 = 到 CPU 数/帧数 | 逐帧配准线程数 |
| `--io-workers` | 0 = 自动（上限 16） | 预览/全分辨率解码线程数、预取队列 |
| `--stack-workers` | 0 = 自动（CPU 核数，RAM 限制） | 中值行带线程数 |
| `--stack-band-mb` | 0 = 16 MB | 每个中值行带的目标工作集 |
| `--render-workers` | 0 = 自动（上限 8） | 阶段 TIFF/JPEG 并行写出 |

粗略内存：

```text
全分辨率对齐栈（uint16 memmap） ≈ N × H × W × 3 × 2 B
                                  = 124 × 3670 × 5496 × 3 × 2
                                  ≈ 15.0 GB（磁盘，不要求全在 RAM）

中值每个行带（目标，float32）    ≈ 16 MB
中值总预算                       ≈ 3 × stack_workers × 16 MB
解码预取队列                     ≈ io_workers × 2 × 240 MB
```

如果内存紧张，先降 `--stack-band-mb`，再降 `--stack-workers` 和 `--io-workers`；
不要通过降低 `--workers` 来省内存，配准任务本身占用的主要是半分辨率灰度图。

---

## 9. 局限与后续

- **中值仍是最大头**：调优后中值叠加 55.3 s，微基准 43.1 s；已经接近“读 memmap +
  排序”的成本。继续提速需要减少数据搬运（例如改变存储布局/分块方式），而不是
  单纯加线程。
- **帧循环仍是单线程组装**：warp 和累加按帧顺序进行，预取只隐藏了部分等待；未来可以
  把 warp 也做成有界 future，但要注意内存和输出顺序。
- **Python/NumPy 的边界**：当前方案依赖 NumPy/OpenCV 释放 GIL；不是所有第三方函数都
  能安全放进线程池，新增步骤要先确认这一点。
- **没有进程池/GPU**：进程池会增加 15 GB 级数组的复制成本；GPU 需要重写中值和通道
  滤波，收益/维护成本另算。
- **不做“无限线程”**：72 核不等于 72 个全图中值任务；这里把行带切到约 16 MB
  （本序列实际 1 行/带），72 个 worker 各只持有一小条，峰值远低于 72 个全图任务。
- **严格 O(1) 于 N 的内存**：当前仍是“每带读取 N 帧”的精确中值；如果帧数继续增大，
  可再实现 4-bit radix select（4 趟、内存与 N 无关），但会增加 I/O。

---

## 10. 相关文件

| 文件 | 改动 |
| --- | --- |
| `scripts/astro/stacking.py` | 自适应行带（`band_mb`）、线程池、RAM 限制、快速中值 + 边界 `nanmedian` |
| `scripts/astro/hotpixels.py` | 逐通道并行、`cv2.dilate` 邻域、减少临时拷贝 |
| `scripts/astro/flat.py` | 逐通道并行 |
| `scripts/astro/pipeline.py` | 配准线程池、CR2 预取、阶段并行写出、`timing.json` |
| `scripts/astro/render.py` | montage/flat 对比面板并行 |
| `scripts/astro/registration.py` | 每帧任务化；单应变换与质量校验 |
| `scripts/astro/util.py` | CPU/可用内存探测、自动 worker 选择 |
| `scripts/align_stack.py` | `--io-workers`、`--stack-workers`、`--stack-band-mb`、`--render-workers` |

更完整的命令行说明见 [`README-cn.md`](README-cn.md)；图像对齐的几何原理见
[`wide-field-registration-cn.md`](wide-field-registration-cn.md)。
