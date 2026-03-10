# photo_maker

本项目是一个面向本地照片目录的处理工具集，当前已经实现了四条完整能力链路：

1. RAW 照片批量转换为 JPG，并带缓存跳过机制。
2. 单张 RAW 照片转 JPG/Base64，再调用 OpenAI 兼容的 VLM API 生成中文图片描述。
3. 整个照片目录做 JPG 准备、图片 embedding、相似照片聚类，并通过本地 Web UI 展示结果。
4. 在聚类结果之上调用 VLM 做二次挑图，并把选择结果实时显示和写回 cluster JSON。

项目最初以 Canon R5 的 `.CR3` 和 Sony A7C2 的 `.ARW` 为目标格式开发，后续在聚类流程中也支持直接读取常见 JPG/JPEG，以及可转成 JPG 的 PNG/WebP。

## 功能概览

### 1. RAW 转 JPG

入口脚本：[convert_raw_to_jpg.py](/Users/xiyang/code/photo_maker/convert_raw_to_jpg.py)

能力：

1. 递归扫描指定目录下的 `.cr3` 和 `.arw` 文件。
2. 使用 `rawpy` 真正解码 RAW 并输出 JPG，不是提取 embedded preview。
3. 输出文件默认缓存到 `/Volumes/mac_ext/temp/raw_converted/`。
4. 输出命名规则为 `最后一级目录名__原文件名.jpg`。
5. 如果目标 JPG 已存在，则直接跳过，不重复转换。
6. 默认使用 8 线程并发转换。
7. 默认输出单行进度，支持 `--verbose` 查看逐文件日志。

当前 RAW 渲染参数：

```python
raw.postprocess(
		use_camera_wb=True,
		no_auto_bright=True,
		output_bps=8,
)
```

这意味着项目当前偏向：

1. 使用相机白平衡。
2. 不启用自动提亮。
3. 输出 8-bit RGB JPG。

### 2. 单张 RAW 转 Base64 + VLM 描述

入口脚本：[describe_raw_with_api.py](/Users/xiyang/code/photo_maker/describe_raw_with_api.py)

相关配置文件：[photo_description_config.py](/Users/xiyang/code/photo_maker/photo_description_config.py)

能力：

1. 校验一张 RAW 文件是否合法。
2. 先复用 JPG 缓存；若不存在则先做 RAW 到 JPG 转换。
3. 将 JPG 转成 Base64。
4. 按 OpenAI Chat Completions 兼容格式，发送 `text + image_url(data:image/jpeg;base64,...)` 请求。
5. 输出中文图片描述结果。

当前 VLM 配置：

1. API Base URL: `http://192.168.2.5:1234/v1`
2. Model: `qwen3.5-2b`
3. System Prompt: 让模型以专业摄影助手身份，用中文描述画面。
4. 默认用户 Prompt: 描述主体、场景、构图、光线、色彩、情绪和拍摄时机。

这条链路可以理解为本项目里的 VLM 能力入口。

### 3. 照片聚类

入口脚本：[cluster_photos.py](/Users/xiyang/code/photo_maker/cluster_photos.py)

能力：

1. 扫描目录中的 RAW/JPG/JPEG/PNG/WebP。
2. 对 RAW 和可转换格式统一准备 JPG 缓存。
3. 使用 ModelScope 下载和加载图片 embedding 模型。
4. 为每张图片计算 embedding。
5. 使用基于 cosine distance 的层次聚类，把高相似照片聚成 cluster。
6. 将结果写入 JSON 文件，默认放在 JPG 缓存目录下。
7. 对 cluster 内部图片做稳定排序，优先按源文件修改时间，再按文件名和路径兜底。

当前 embedding 模型：

1. ModelScope 模型 ID: `damo/multi-modal_clip-vit-base-patch16_zh`
2. 不使用 Hugging Face。
3. embedding 维度当前为 `512`。

当前聚类策略：

1. 使用 `AgglomerativeClustering`。
2. `metric="cosine"`
3. `linkage="average"`
4. 通过 `cluster_similarity_threshold` 控制聚类严格程度。
5. 实际 distance threshold 计算方式为 `1.0 - similarity_threshold`。

### 4. 本地 Web UI

后端入口：[server/app.py](/Users/xiyang/code/photo_maker/server/app.py)

前端页面：[server/index.html](/Users/xiyang/code/photo_maker/server/index.html)

能力：

1. 启动本地 FastAPI 服务。
2. 首页自动加载控制面板和结果展示区域。
3. 调用 macOS 原生目录选择器选择待处理照片目录。
4. 创建后台聚类任务并轮询进度。
5. 支持取消正在运行的任务。
6. 展示 embedding 缓存统计信息。
7. 支持清空 embedding 缓存。
8. 结果区使用响应式平铺照片墙展示 cluster。
9. 点击任意图片可在新标签页打开大图。
10. 支持 VLM 挑图对话框，可配置 prompt、endpoint 和模型。
11. 模型列表可从 endpoint 自动拉取。
12. 支持实时显示 VLM token 输出，包括 thinking token。
13. 支持只处理未挑过的 cluster，或重新覆盖已有挑图结果。

### 5. 基于 VLM 的挑图

核心实现文件：[vlm_pick.py](/Users/xiyang/code/photo_maker/vlm_pick.py)

能力：

1. 读取已有的 cluster JSON。
2. 确保每个 cluster 内的图片都有从 0 开始的 `image_id`。
3. 把同一 cluster 的多张图片和 prompt 一起发送给 OpenAI 兼容多模态接口。
4. 支持从 endpoint 动态获取模型列表。
5. 支持逐 token 流式读取模型输出。
6. 支持记录普通输出 token 和可能存在的 thinking token。
7. 要求模型最后输出一行 JSON，格式为 `{"select": 1}`。
8. 从这个 JSON 中读取选择结果，并写回 cluster JSON。

当前默认挑图 prompt 会要求模型：

1. 比较同组相似照片。
2. 重点看主体状态、姿态、清晰度、构图、画面完整性、光线和整体观感。
3. 最后一行严格输出 `{"select": <整数>}`。

## 当前设计

### 设计目标

本项目的设计不是做一个通用图像平台，而是围绕本地个人照片工作流做三件事：

1. 尽量少重复计算。
2. 对大目录可持续运行。
3. 输出结果适合人工快速浏览和筛选。

### 设计决策

#### 1. 统一先落 JPG 缓存

无论是 VLM 描述还是聚类，都先统一为 JPG 缓存路径，这样可以避免：

1. 每次都重复解码 RAW。
2. 同一张图在不同流程里多次生成临时文件。
3. Web UI 无法直接展示 RAW 的问题。

#### 2. embedding 结果持久化缓存

embedding 缓存在：

```text
/Volumes/mac_ext/temp/raw_converted/.embedding_cache/
```

缓存文件使用 `.npz` 保存，校验条件包括：

1. `model_id`
2. `image_path`
3. 图片文件 `mtime_ns`
4. 图片文件大小 `st_size`

只要图片内容没变、模型没变，重复聚类时就不需要重新算 embedding。

#### 3. 进程内复用 embedding 模型

在同一个 Python 进程里，多次运行聚类时会复用已经加载好的 ModelScope CLIP 模型，避免每次都重新初始化模型对象。

#### 4. Web UI 侧重“浏览”而不是“编辑”

页面设计偏向 Google Photos 式的平铺查看：

1. 图片无圆角。
2. 自适应列数和浏览器缩放。
3. 平铺自动排列，优先让用户快速扫图。
4. 图片点击直接开大图，不引入复杂的内嵌 viewer 状态。

#### 5. VLM 挑图作为“聚类后的第二阶段”

VLM 不直接对整个目录的全部照片做选择，而是在聚类结果基础上二次处理。这样做的好处是：

1. 每次请求发给模型的图片数量更可控。
2. 更符合“从一组相似图里挑一张”的真实使用场景。
3. 更便于把结果落回对应 cluster。
4. 可以反复调整 prompt，而不需要重新做 embedding 和聚类。

### Web UI 设计要点

当前页面包含：

1. 左侧控制区：目录、缓存目录、线程数、处理上限、聚类相似度滑杆。
2. 状态区：任务阶段、进度条、运行状态。
3. 缓存区：embedding 缓存文件数、大小、目录。
4. 结果区：每个 cluster 一组平铺照片墙。
5. VLM 挑图对话框：endpoint、API key、模型选择、prompt 调试、处理选项和实时输出面板。

页面是响应式的：

1. 宽屏下为侧边栏 + 主内容双栏。
2. 窄屏下自动堆叠为单栏。
3. 照片墙列宽会随视口变化自动调整。

## 项目结构

```text
photo_maker/
├── convert_raw_to_jpg.py         # RAW -> JPG / Base64 基础能力
├── describe_raw_with_api.py      # 单张 RAW -> VLM 描述
├── photo_description_config.py   # VLM API 与 prompt 配置
├── cluster_photos.py             # 图片聚类主逻辑
├── vlm_pick.py                   # 基于 VLM 的 cluster 挑图逻辑
├── server/
│   ├── app.py                    # FastAPI 服务
│   └── index.html                # Web UI
├── requirements.txt              # Python 依赖
└── README.md
```

## 依赖

当前依赖列表来自 [requirements.txt](/Users/xiyang/code/photo_maker/requirements.txt)：

```text
rawpy
Pillow
openai
modelscope
torch
torchvision
scikit-learn
oss2
fastapi
uvicorn
```

## 安装

本项目当前按“直接使用系统 Python / 全局安装依赖”的方式使用。

安装依赖：

```bash
/opt/homebrew/bin/python3 -m pip install --break-system-packages -r requirements.txt
```

说明：

1. 当前环境是 macOS + Homebrew Python。
2. 如果遇到 PEP 668 限制，需要 `--break-system-packages`。
3. 项目开发时明确没有采用 venv。

## 用法

### 1. 批量 RAW 转 JPG

```bash
/opt/homebrew/bin/python3 convert_raw_to_jpg.py \
	'/Volumes/p4510_8t_static/photos/20250929国庆/'
```

带参数示例：

```bash
/opt/homebrew/bin/python3 convert_raw_to_jpg.py \
	'/Volumes/p4510_8t_static/photos/20250929国庆/' \
	--output-dir '/Volumes/mac_ext/temp/raw_converted/' \
	--workers 8 \
	--verbose
```

### 2. 单张 RAW 走 VLM 描述

```bash
/opt/homebrew/bin/python3 describe_raw_with_api.py \
	'/Volumes/p4510_8t_static/photos/20250929国庆/_MG_7068.CR3'
```

自定义 prompt：

```bash
/opt/homebrew/bin/python3 describe_raw_with_api.py \
	'/path/to/photo.CR3' \
	--prompt '请重点描述人物姿态、表情、构图和氛围。'
```

### 3. 命令行聚类

```bash
/opt/homebrew/bin/python3 cluster_photos.py \
	'/Volumes/p4510_8t_static/photos/20250929国庆/'
```

带参数示例：

```bash
/opt/homebrew/bin/python3 cluster_photos.py \
	'/Volumes/p4510_8t_static/photos/20250929国庆/' \
	--output-dir '/Volumes/mac_ext/temp/raw_converted/' \
	--workers 8 \
	--batch-size 16 \
	--cluster-similarity-threshold 0.88
```

说明：

1. 结果 JSON 默认输出到：

```text
/Volumes/mac_ext/temp/raw_converted/<目录名>__clusters.json
```

2. 例如测试目录 `20250929国庆` 的结果默认是：

```text
/Volumes/mac_ext/temp/raw_converted/20250929国庆__clusters.json
```

### 4. 启动 Web UI

```bash
/opt/homebrew/bin/python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

然后浏览器打开：

```text
http://127.0.0.1:8765/
```

### 5. Web UI 中执行 VLM 挑图

在页面中：

1. 先完成一次聚类，确保已经有 result JSON。
2. 点击 `VLM 挑图` 打开对话框。
3. 输入或确认 endpoint。
4. 点击 `拉取模型`，从 endpoint 自动获取模型列表。
5. 选择模型。
6. 使用默认 prompt 或继续修改 prompt。
7. 选择处理策略：
	`只处理还没有 VLM 结果的 cluster`
	`重新挑图并覆盖已有结果`
8. 点击 `开始 VLM 挑图`。
9. 在右侧实时查看 token 输出。
10. 完成后页面会高亮被选中的图片，并把结果写回 cluster JSON。

## Web API 概览

当前主要接口：

1. `GET /`：返回前端页面。
2. `GET /api/defaults`：返回默认输出目录。
3. `POST /api/pick-directory`：调用 macOS 原生目录选择器。
4. `POST /api/jobs`：创建聚类任务。
5. `GET /api/jobs/{job_id}`：查询任务进度和结果。
6. `POST /api/jobs/{job_id}/cancel`：取消任务。
7. `GET /api/cache/embedding/status`：获取 embedding 缓存状态。
8. `POST /api/cache/embedding/clear`：清空 embedding 缓存。
9. `POST /api/vlm/models`：从指定 endpoint 拉取模型列表。
10. `POST /api/vlm/pick/stream`：流式执行 VLM 挑图并实时推送 SSE 事件。
11. `GET /api/image?path=...`：读取并返回缓存图片。

## 输出格式

聚类结果 JSON 主要字段：

1. `input_dir`
2. `cache_output_dir`
3. `result_path`
4. `embedding_model_id`
5. `total_photos`
6. `cluster_count`
7. `singleton_count`
8. `cluster_similarity_threshold`
9. `cluster_distance_threshold`
10. `clusters`
11. `singletons`
12. `embedding_shape`

其中 `clusters` 中每个元素包含：

1. `cluster_id`
2. `size`
3. `average_similarity`
4. `items`

每个 `items` 元素包含：

1. `image_id`
2. `source_path`
3. `jpg_path`

如果执行过 VLM 挑图，cluster 中还会新增 `vlm_pick` 字段，典型内容包括：

1. `select`
2. `selected_image_id`
3. `selection_json`
4. `content_text`
5. `thinking_text`
6. `transcript_text`
7. `endpoint`
8. `model`
9. `prompt`
10. `updated_at`

其中当前约定的最终选择 JSON 为：

```json
{"select": 1}
```

系统会优先从这个 JSON 中读取 `select`，并把结果写回 cluster JSON。

## 已实现但刻意不做的事情

当前版本没有做：

1. 相似图片对单独列表展示。这个功能已经被移除，只保留 cluster 结果。
2. Hugging Face 模型下载。embedding 模型固定走 ModelScope。
3. 浏览器内复杂大图灯箱。当前策略是直接新标签页打开大图。
4. 数据库或长期任务队列。当前使用内存任务状态即可满足本地使用。

## 当前已验证的能力

开发过程中已经验证过：

1. `/Volumes/p4510_8t_static/photos/20250929国庆/` 目录上的 RAW 转 JPG。
2. 二次运行会正确跳过已有 JPG。
3. `qwen3.5-2b` 的 OpenAI 兼容 VLM 描述接口调用。
4. ModelScope embedding 模型下载、加载和 embedding 计算。
5. 聚类 JSON 结果生成。
6. embedding 缓存命中与清空逻辑。
7. Web UI 的目录选择、进度轮询、取消任务、缓存状态和结果展示。
8. OpenAI 兼容 endpoint 的模型自动拉取。
9. VLM 挑图流式 token 输出。
10. `{"select": N}` 结果解析与 cluster JSON 回写。

## 后续可扩展方向

如果继续演进，这个项目比较自然的下一步包括：

1. 给结果页加大图 viewer，支持同 cluster 左右切换。
2. 读取 EXIF 拍摄时间，替代目前基于文件修改时间的排序。
3. 增加更多 VLM prompt 模板，例如婚礼、人像、风景、旅行纪实。
4. 增加更细粒度的聚类后筛选，例如按日期、机身、镜头过滤。
