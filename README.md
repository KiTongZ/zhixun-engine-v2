# 智训引擎

智训引擎是赛题九的工业模型续训方案。一套 Harness 支持三项任务：识别气体类别、估计轴承剩余寿命、估计涡扇发动机剩余寿命。方案使用 PyTRiO 基础模型 `Qwen/Qwen3.8-27B` 决定每轮训练的学习率和轮数，实际更新的是各任务的官方 CPU 多层感知机（MLP）。

## 一轮训练如何运行

1. **加载父模型。** Harness 读取当轮公开训练集和验证集，通过 SDK 校验平台声明的父检查点摘要。加载 `checkpoint.npz` 后，检查任务类型、特征顺序和分类标签，并继承权重、Adam 优化器状态、标准化参数及训练计数。
2. **确定训练预算。** Harness 将样本数、父模型已训练轮数和当前验证指标发给平台模型。每轮允许续训 1 至 3 个 epoch，学习率限制在任务默认值的 0.5 至 2 倍。
3. **选出候选模型。** 每个 epoch 只用训练集更新权重，再用验证集评估。分类任务选择 Macro-F1 最高的已训练检查点，回归任务选择 RMSE 最低的已训练检查点。
4. **交付和预测。** Harness 用官方格式保存候选模型，提交训练历史与模型调用回执。独立预测入口加载候选检查点，输出六个气体类别的分数，或一个非负的剩余寿命估计值。

续训、候选选择和预测的实现见 [contest_submission.py](deploy/contest/contest_submission.py)。

## 公开验证集结果

三项实验均从各自的官方初始检查点出发，连续训练 3 个 epoch。A、B 的学习率为 0.003，C 为 0.002。表中“选中轮次”从本次续训的第一个 epoch 开始计数。

| 任务 | 公开验证指标 | 官方父模型 | 续训候选 | 选中轮次 | SwanLab 记录 |
| --- | --- | ---: | ---: | ---: | --- |
| A · 气体六分类 | Macro-F1 ↑ | 0.714030 | 0.725695 | 1 | [查看实验 A](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/ty222agv) |
| B · 轴承剩余寿命 | RMSE（分钟）↓ | 681.159833 | 656.351785 | 2 | [查看实验 B](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/qa3rfg8n) |
| C · 发动机剩余寿命 | RMSE（循环）↓ | 52.868573 | 52.350178 | 1 | [查看实验 C](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/luzfdx9p) |

这是本地 Harness 在官方公开验证集上的测量结果。为复现相同预算，本地 SDK 宿主替身返回固定训练参数。SwanLab 记录了任务模型的逐轮损失、验证指标和候选指标。真实 27B 决策调用由赛事平台在试运行时执行，隐藏集成绩以平台评测为准。

## 构建比赛镜像

在仓库根目录执行：

```sh
docker build --platform linux/amd64 -f deploy/contest/Dockerfile -t zhixun-contest:artifact-batch-v1 .
docker run --rm zhixun-contest:artifact-batch-v1 describe
docker save -o zhixun-contest-artifact-batch-v1.tar zhixun-contest:artifact-batch-v1
```

`describe` 应返回协议 `contest.harness.workspace.v2`、SDK 版本 `1.0.4` 和任务类型 `classification`、`regression`。将导出的 tar 上传到赛事平台，选择“基础模型（可正式参赛）”和 `Qwen/Qwen3.8-27B`，在平台密钥栏配置 PyTRiO API Key。训练由镜像内的 `/app/harness.py` 启动。每个候选提交权重、配置、特征展开代码和独立预测入口。正式预测命令为 `python3 -B /model/predict.py`。

## 复现本地实验

使用 Python 3.12。分别解压赛事提供的三个基线包，使以下目录各自直接包含 `checkpoint.npz`、`baseline-info.json`、`public/train.jsonl` 和 `public/validation.jsonl`。

```text
vendor/baselines/cpu-a-guide-20261110-v1/
vendor/baselines/cpu-b-rul-20261110-v1/
vendor/baselines/cpu-c-guide-20261110-v1/
```

联调脚本会将每次训练在线记录到 `KiTongZ/zhixun-engine-v2`。使用有该项目写入权限的 SwanLab 账号，在仓库根目录执行：

```sh
python -m pip install -r deploy/contest/requirements.txt swanlab==0.10.5
swanlab login
python deploy/contest/tests/local_round.py --dataset A --epochs 3
python deploy/contest/tests/local_round.py --dataset B --epochs 3
python deploy/contest/tests/local_round.py --dataset C --epochs 3
```

每次成功运行会输出 `LOCAL ROUND OK` 和本次实验目录。在该目录中，`finish.json` 保存本地宿主回执，`workspace/state-r1/round.json` 保存逐轮指标和选中轮次，`workspace/candidate-r1/checkpoint.npz` 是通过独立预测入口验证的候选模型。复现脚本见 [local_round.py](deploy/contest/tests/local_round.py)。

## 轴承等权采样与后续实验


长寿命轴承贡献更多快照。为避免其在每轮训练中占据过大比重，B 的每个 epoch 按轴承等权、有放回地抽取与原训练集等量的快照，编号仅用于分组。A、C 沿用第二版策略。B 提交第十轮的轮内最优候选。

本地从公开训练集轮流留出一整根轴承，剩余设备独立拟合标准化和模型；同折内从同一预训练状态比较原采样、部分均衡和完全均衡。九折的逐轴承平均 RMSE 分别为 749.79、730.08、698.18 分钟，完全均衡在七折改善。使用官方父模型在公开验证集做 20 个 epoch 的固定预算对照，原采样和完全均衡分别为 689.16、648.09 分钟。随后按第二版预算经实际训练入口本地回放十轮，最终公开验证 RMSE 为 643.18 分钟，MAE 为 538.08 分钟。等权版已完成平台试运行，B 隐藏 RMSE 为 217.03 分钟，MAE 为 141.51 分钟。

[九折采样对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/jwwhspl9) · [十轮回放](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/h2d2cfb6)

## 发动机后五轮学习率候选

本次候选保留轴承等权采样，在发动机任务第 6 至 10 轮使用 0.004 学习率。公开验证回放第十轮 RMSE 为 51.602 循环，默认分支为 52.232；MAE 分别为 37.209 和 37.839。四个工况子集的末轮 RMSE 都有改善。0.004 分支在第七轮出现较大的非对称惩罚，隐藏集效果待测。单线程和四线程配置共用同一数据与种子。

[单线程实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/2zzmhqx3) · [四线程配置复核](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/r6gyrg6u)

## 提交信息

- **模型类型**　基础模型，`Qwen/Qwen3.8-27B`
- **Harness 镜像**　`zhixun-contest-artifact-batch-v1.tar`，Linux/amd64
- **实验归档**　[SwanLab · zhixun-engine-v2](https://swanlab.cn/@KiTongZ/zhixun-engine-v2)
- **项目源码**　[GitHub · KiTongZ/zhixun-engine-v2](https://github.com/KiTongZ/zhixun-engine-v2)







## 发动机末轮输出层候选

C 的第十轮在轮内最佳检查点上冻结隐藏层，用公开训练行拟合线性输出的残差，岭惩罚系数为 0.001。公开主指标严格改善时才提交修正后的检查点。完整公开验证 RMSE 从 51.602 降到 51.407 循环，MAE 从 37.209 升至 37.302；27 台整机留出对照 RMSE 从 54.289 降至 53.773。公开集的高估误差 P95 略升，隐藏成绩待测。

[训练与评估记录](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/ukyppnnd) · 候选镜像 `zhixun-contest-c-head-ridge.tar`

## 预测产物隔离预检

以 A、B、C 实际训练产物分别只读挂载到 /model，在仅安装 Python 和 NumPy 的 Linux 容器中关闭网络、使用非特权用户执行预测。逐一检查批量 JSON、带 samples 字段的批次、重复样本、空批次和逐行请求，验证默认路径以及显式目录和检查点参数。对照同一检查点的参考输出核对顺序和数值。

在仓库根目录执行以下命令，参数依次为官方基线包根目录和三个已经生成的候选模型目录。

```sh
docker build --platform linux/amd64 -t zhixun-predict-check:bare -f deploy/contest/tests/Dockerfile.predict-check .
python deploy/contest/tests/isolated_predict.py PACKAGES A_CANDIDATE B_CANDIDATE C_CANDIDATE
```

## 正式训练与评测报告

平台已完成 A、B、C 各 10 轮训练与评测，客观成绩为 **12.0001 / 35**。平均性能提升 3.346373 / 15，最差阶段表现 0.280711 / 5，恢复速度 3.373016 / 5，稳定性 5 / 5，资源效率 0 / 5。

| 任务 | 第 10 轮公开验证指标 | 平台报告的任务运行用时 |
| --- | --- | ---: |
| A · 气体识别 | 准确率 0.980362，Macro F1 0.978181 | 1,347.15 秒 |
| B · 轴承寿命 | RMSE 633.94311 分钟，MAE 543.743545 分钟 | 678.48 秒 |
| C · 发动机寿命 | RMSE 51.469978 循环，MAE 37.333099 循环 | 1,911.49 秒 |

C 第 10 轮记录确认平台执行了输出层修正（岭惩罚 0.001）。本节的模型指标来自逐轮公开验证栏；截图未披露分任务隐藏集指标、各阶段对总分的贡献或资源效率计分明细。

## 本地候选：轴承后九轮小预算续训

已正式评测的镜像为 zhixun-contest-artifact-batch-v1（12.0001 / 35）。新候选 zhixun-contest-b-small-budget 沿用其预测协议与 A/C 策略。B 在第 2–10 轮以 0.0015 学习率做一次设备等权抽样更新，每次抽取原训练集行数的 1/32。本地实际入口回放首轮使用 3 个 epoch、学习率 0.006。

四批筛选共运行 90 条十轮路径（含对照复跑），另用真实训练入口回放两次。最终九折分别留出一根训练轴承，在其余设备上初始化并预训练，公开验证集用于轮内选模。相对各折初始模型，十轮平均 RMSE 改善由 3.80% 增至 7.36%；逐折最差轮改善的均值由 −3.81% 升至 +3.08%。最长连续未追平历史最佳的轮数平均由 7.78 降至 6.78；相邻轮误差变化绝对值占初始 RMSE 的平均比例由 5.21% 降至 2.30%。这些指标用于本地比较，官方阶段与评分权重未在本地重现。

完整公开集三种抽样种子的十轮末 RMSE 均值从 647.11 降至 642.07 分钟，其中一个种子退步。三种子十轮平均改善为 5.31% → 4.94%，最差轮改善均值为 2.86% → 2.54%。标准种子真实训练入口回放末轮 RMSE 为 624.40 分钟。十轮处理的训练行次从 147,672 降到 23,076；线上 Token 与资源效率分待评测。此候选尚未送正式评测。

[初筛](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/o1dwy931) · [半量复核](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/7mkog8b9) · [三种子对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/h2xla2wu) · [小预算九折复核](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/bd9idmxv) · [最终入口回放](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/9gipavjh)

## 最新本地候选 · 特征路径缓存

本版沿用轴承小预算方案，将重复的特征路径编码及字段名检查结果缓存在进程内。缓存容量分别为 4,096 和 1,024 项，每条输入的数值仍逐项校验。相同检查点读取两版数据所得的特征、标签、样本顺序和预测在六个公开数据分片上逐项一致。

以下为同一台电脑上各三次读取的 CPU 时间中位数，单位为秒。

| 任务 | 训练集（原版 → 新版） | 验证集（原版 → 新版） |
| --- | ---: | ---: |
| A 气体识别 | 0.219 → 0.047 | 0.859 → 0.250 |
| B 轴承寿命 | 1.719 → 0.797 | 0.266 → 0.062 |
| C 发动机寿命 | 9.797 → 3.109 | 2.328 → 0.922 |

以上只计数据读取与校验。平台总用时、资源效率得分和各任务隐藏集表现仍待正式评测。镜像归档经重新加载并通过 describe 检查。使用新版代码重新打包三个任务的模型产物，三种启动参数共九次隔离预测测试均通过。[效率与预测一致性实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/1ifa06g4)

气体识别另跑了四组共 52 个十轮筛选配置（含各组对照）。从同一父检查点比较分类输出层融合、特征幅度扰动与类别平衡、传感器或特征随机遮挡、类别偏置修正。完整公开验证回放未显示可复现的优势，候选沿用原有分类模型。[输出层](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/edvm3r5c) · [幅度与类别平衡](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/4z4solhs) · [随机遮挡](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/xyjl6gqg) · [偏置修正](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/lk9c3z8o)

## 基于 feature-cache 的资源优化候选

归档为 `zhixun-contest-feature-cache-efficient.tar`，基线源码为 34134d5。A、B、C 每轮保留被选中检查点的公开验证指标，候选声明、状态文件和最终回执共用同一次计算结果。相较该基线，每轮省去四次重复的全验证集前向计算。学习率策略、模型决策 Token 上限和 C 末轮输出层惩罚均沿用 feature-cache。B 小预算轮遇到匿名设备编号时使用固定种子的均匀抽样，可识别设备时继续等权抽样。

本地验证使用公开数据和模拟模型网关。平台 v9 报告为入口预检的只读检查操作次数耗尽，平台正式预检和评分仍需由平台验证。

## 基于回滚版的资源优化

交付归档为 `zhixun-contest-feature-cache-resource.tar`，采用 Linux/amd64。以 `feature-cache-efficient` 的 59fff6c 为对照，回滚版的 B 匿名编号回退、验证指标复用和 C 末轮选模策略一并继承。Harness按轮读取平台提供的父检查点，使用当轮公开数据更新权重，并根据真实模型调用回执提交该轮结果。

固定长度的数值列表复用特征路径，同时逐值检查有限性。Adam 在每批梯度更新前计算一次裁剪倍率和两项偏差校正。模型决策缩短为预算 JSON，输出上限取 64 与平台上限的较小值，并通过 SDK 关闭思考输出。短回复不完整或数值不可用时，按该任务和轮次的推荐预算继续。

| 任务 | 回滚版十轮 CPU 秒 | 新版十轮 CPU 秒 | 新版末轮公开验证指标 |
| --- | ---: | ---: | --- |
| A 气体识别 | 1.94 | 1.28 | Macro F1 0.718114 |
| B 轴承寿命 | 4.64 | 4.28 | RMSE 624.397524 分钟 |
| C 发动机寿命 | 40.06 | 31.20 | RMSE 51.407247 循环 |

每项任务以官方父检查点开始，在两版中使用相同预算、公开数据与随机种子，各回放十轮，共 30 对逐轮指标和检查点文件完全一致。计时包含数据读取、计算和产物保存，模型网关由本地模拟。短提示的字符数中位数在 A、B、C 中分别为 593 → 316、598 → 343、731 → 314。正式平台的计费 Token、完整预检和资源效率得分待验证。[SwanLab 成对回放](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/goz3c2tc)
从最终归档重新加载 Linux 镜像后，在只读容器中挂载公开数据和父检查点，A、B、C 各完成一轮训练。随后仅挂载各自的模型产物，共通过九次断网批量预测测试。[A](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/e5tg8f7l) · [B](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/bd6mwiim) · [C](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/705s4sdm)

## 12.961812 分报告后的候选

报告显示客观成绩 12.961812 / 35，平均性能提升 3.65939 / 15、最差阶段表现 0.096073 / 5、恢复速度 4.206349 / 5、稳定性 5 / 5、资源效率 0 / 5。截图中 A 的十轮平台运行时间为 1,031.28 秒。现有本地运行和报告无法拆分该时间的来源，平台计分和隐藏集指标仍以平台为准。

这版从资源优化版 ba319e8 继续。A 的公开验证 Macro F1 下降、或 C 的公开验证 RMSE 比父模型高出超过 1% 时，只额外验证一次半步参数更新。半步优于本轮原候选时才采用，随产物保留本轮续训的 Adam 状态。B 继续使用既有采样和预算。阈值来自公开数据和模拟工况切换的三种子实验。

| 本地场景 | 十轮平均 | 最差轮 | 末轮 |
| --- | ---: | ---: | ---: |
| A 留出样本 Macro F1，越高越好 | 0.72647 → 0.72758 | 0.71886 → 0.72472 | 0.72177 → 0.72743 |
| C 整机留出 RMSE，循环 | 55.10 → 55.08 | 56.16 → 56.07 | 53.98 → 54.04 |
| C 模拟工况正序切换 RMSE，循环 | 59.41 → 57.06 | 71.95 → 61.58 | 58.38 → 57.35 |
| C 模拟工况逆序切换 RMSE，循环 | 57.93 → 56.59 | 64.56 → 58.93 | 55.64 → 55.69 |

数据加载复用严格 JSON 解码器，并继续拒绝重复字段和非有限常量。C 训练文件的三次读取 CPU 时间中位数从 1.828 秒降到 1.781 秒。A、B 的小文件计时波动较大。运行记录现在包含准备、网关、训练选模、打包、候选提交和状态保存时间，以及网关响应提供的 Token 用量。这里的本地回放使用模拟网关。

[SwanLab 半步实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/kh1puxfe) · [C 触发阈值对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/9qrbyp5g) · [数据一致性和计时](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/w00c0wdc) · [实际入口十轮对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/plz6bxl6)
最终归档为 zhixun-contest-feature-cache-stable.tar，已重新加载并通过 A、B、C 容器内首轮训练，以及九次断网、只读的独立预测检查。60 轮成对回放中 B 的十个检查点逐字节一致。C 在相同预算下的完整十轮本地 CPU 时间为 32.188 → 31.078 秒，该计时不包含真实平台等待。

## 轴承分轮预算与分块读取候选

本版从 `feature-cache-stable`（a8a19d1）继续。归档为 `zhixun-contest-feature-cache-taper.tar`。B 第 2 至 7 轮抽取公开训练样本的 1/64，第 8 至 10 轮恢复 1/32。以官方公开数据的 7,032 条训练样本为例，后九轮训练行次从 1,980 降至 1,320。按 128 的批量大小，后九轮的梯度更新从 18 次降到 12 次。能识别设备时按设备等权抽样，匿名编号时均匀抽样。

| B 本地场景 | 十轮平均 RMSE | 最差轮 RMSE | 末轮 RMSE |
| --- | ---: | ---: | ---: |
| 九个整轴承留出折 | 656.21 → 656.30 | 677.84 → 670.99 | 651.06 → 651.30 |
| 全公开验证集，三个种子 | 639.36 → 634.32 | 653.02 → 651.44 | 638.49 → 640.27 |
| 匿名训练编号，三个种子 | 662.29 → 657.31 | 672.29 → 667.23 | 666.75 → 659.10 |
| 实际训练入口，固定种子 42 | 638.00 → 638.89 | 649.22 → 654.97 | 624.40 → 621.47 |

单位为分钟，越低越好。整轴承留出实验在预训练和续训中均排除了该轴承。加权输出层和目标分层抽样的收益未能在对照中稳定复现。正式评测得分与资源计量由平台确认。

加载器按约 1 MiB 分批读取并计算原始文件 SHA-256，避免同时持有整份原始字节、解码文本和拆分后的行。A、C 的十轮成对回放各有十个逐字节相同的检查点。归档重载后，三项任务均完成 Docker 内的首轮训练和九次断网、只读批量预测检查。

[SwanLab 分轮预算](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/l6pijpsl) · [真实训练入口十轮对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/4njeymj1) · [输出层实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/v01k0ps9) · [分层抽样实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/y1aezixg)
在独立 Windows 进程各三次加载 C 公开训练文件，峰值工作集的中位数从 250.81 MiB 降至 192.45 MiB（约 23%）。C 的十轮成对回放 CPU 时间为 31.09 → 31.59 秒。公开数据的样本顺序、特征、标签、SHA-256 和检查点预测均通过一致性复核，另验证了 UTF-8 跨块和混合换行。[读取对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/crzfih8d)

## C 工况校正单变量对照

提交归档为 `zhixun-contest-c-condition-calibrated.tar`，从已完成线上评测的 `feature-cache-taper`（26c90eb，客观成绩 12.915568 / 35）派生。A、B 保持原训练和评估策略。C 每轮沿原路线更新父检查点，在本轮最终候选的基础上，用公开训练样本拟合工况残差校正。传感器按工况单独标准化。剩余寿命至少 150 循环的训练样本在校正拟合中加倍权重。验证数据只用于决定是否接受校正。

只有当公开验证集整体 RMSE 改善，且存在样本的近失效段（小于 25 循环）和长寿命段（至少 250 循环）各自的 RMSE 均未上升，才将校正系数写入候选配置。预测入口只读取当次传感器观测和产物内的校正系数。对拟合时没见过的工况，输出基础模型预测。下轮继承原模型检查点，并在当轮重新拟合校正层。

在固定公开训练与验证投影的十轮成对回放中，C 的逐轮平均 RMSE 从 52.0065 降至 51.6498，最差轮由 52.3842 降至 52.3589，末轮均为 51.4160。A、B 共二十个检查点逐字节一致。三项任务均通过镜像重载、容器内首轮训练和只读模型目录中的独立批量预测检查。线上正式评测用于判断整套提交的综合收益。

模型类型为基础模型，`Qwen/Qwen3.8-27B`。评测时沿用对照版本的绑定模型和其他设置。[源码分支](https://github.com/KiTongZ/zhixun-engine-v2/tree/c-condition-calibrated) · [公开数据工况校正实验](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/eycedo6d) · [训练入口十轮成对回放](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/apfchd3u)
另将实际触发校正的第 2 轮 C 产物放入仅挂载模型目录的预测容器，验证了单条 JSONL、乱序批次、重复样本、空批和三种路径参数。陌生工况回退到基础模型，预测数值逐项一致。该轮公开 RMSE 由 52.0587 降至 51.3775，MAE 由 38.5471 升到 39.1595。[带校正产物的隔离预检](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/7cuxvrd0)

## C 离线策略库候选

`zhixun-contest-c-offline-strategies.tar` 是从 `c-condition-calibrated`（dc2e88f）派生的后续对照。镜像归档 65,144,320 字节（62.126 MiB）。策略库是随镜像分发的 JSON，含三条改写后的公开资料摘要。C 的训练入口按多工况首轮、前期长寿命误差关注、后期续训三个情形选择一条，把短建议加入同一次模型请求。回执和本轮状态同时记下策略编号和来源。外部资料以公开事实和策略摘要的形式提供给决策模型，数值拟合使用平台挂载的当轮公开训练集。

本地使用模拟网关验证三个检索分支、同预算下的 A/B/C 十轮回放和候选检查点一致性。新镜像在断网容器完成 C 首轮训练，回执中的检索项为 `operating_regimes`。真实网关的预算选择和线上得分仍待验证。先用工况校正版本与 12.915568 的原版做 C 校正对照，再用本版与工况校正版对照检索增量。

[NASA 工况资料](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) · [按整机分组验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) · [早停研究](https://pubmed.ncbi.nlm.nih.gov/12662814/) · [SwanLab 成对回放](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/nzs2iky6) · [断网容器训练](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/38loiqtg)
同预算对照的 30 对检查点逐字节一致。检索让 C 每轮提示的字符数中位数从 314 增至 418，实际新增 Token 以真实网关计量为准。与工况校正版相比，归档增加 7.5 KiB。

## 4 核、4 GiB 本地测试

训练和独立预测的 Docker 测试命令使用 `--cpus 4 --memory 4g --memory-swap 4g`。训练入口和独立预测脚本在导入 NumPy 之前初始化 BLAS 线程配置，默认每进程使用 1 个 BLAS 线程，可用 `BASELINE_CPU_THREADS` 做显式对照。

在旧镜像上挂载同一份 C 公开数据，测量两轮单机训练、校正拟合与验证前向。每种设置在 4 核、4 GiB 与断网条件下重复三次，取中位数。

| OpenBLAS 实际线程数 | 墙钟秒 | 进程 CPU 秒 |
| ---: | ---: | ---: |
| 24，旧默认设置 | 0.5940 | 1.9731 |
| 1 | 0.3271 | 0.3256 |
| 2 | 0.3089 | 0.4612 |
| 4 | 0.3197 | 0.7552 |

四种设置的公开 RMSE 在记录到的九位小数一致。单线程相比旧默认的墙钟时间缩短约 45%，CPU 时间减少约 84%。本地干净进程已确认源码修复后两个入口默认都初始化为 1 个线程，显式配置 2 时训练入口初始化为 2。新版归档为 zhixun-contest-c-offline-4c4g.tar。重载后在 4 核、4 GiB 容器中实测训练入口为 1 个 OpenBLAS 线程。A、B、C 各通过一轮训练及三种路径参数的断网只读预测检查。[BLAS 基准计时](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/omlc3nz6)

### 读取和批量预测剖析

在相同容器资源限制下，对比 `c-offline-4c4g` 和 `c-cpu-profiled`。C 的 JSONL 解析针对固定传感器结构直接展开特征，按块转换为 NumPy 数组。校正按工况构造矩阵，验证阶段复用同一份基线预测和校正残差。C 两版交替执行，各取三次中位数。

| C 公开数据本地测量 | `c-offline-4c4g` | `c-cpu-profiled` |
| --- | ---: | ---: |
| 训练与验证数据读取 CPU 秒 | 1.6172 | 1.1165 |
| 训练与验证数据读取墙钟秒 | 1.8546 | 1.3445 |
| 截至读取结束的进程峰值 RSS（MiB） | 200.61 | 102.67 |
| 完整剖析脚本进程峰值 RSS（MiB） | 345.47 | 246.83 |
| 23,049 条批量预测 CPU 秒 | 0.1213 | 0.0853 |

A、B、C 对照的解析产物和预测哈希一致，C 校正配置一致。重载归档 `zhixun-contest-c-cpu-profiled.tar` 后，三套数据各通过一轮本地训练和只挂载模型目录的断网只读预测。表中时间仅覆盖对应的本地路径，线上阶段耗时与官方资源效率分未实测。[SwanLab 基准](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/p0eyqu2u)

### A 题乙醛跨批次误判与离线检索

按照赛题标签，公开训练批次有 445 条样本，其中 30 条乙醛。公开验证批次有 1,244 条样本，其中 109 条乙醛。官方起始检查点漏掉了全部 109 条，108 条判为乙醇，1 条判为甲苯。按类别做交错留出及按原行序留出，基准、更新门控与小幅阈值调整在三个随机种子下均未恢复乙醛召回。另用公开训练集拟合乙醛与乙醇的补充分支，最佳特征方案救回 10 条，全部来自验证样本的前半组，后半组仍为零。该分支留作实验记录。[SwanLab 对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/aga56mm5)

`a-acetaldehyde-retrieval` 镜像附带短策略卡。每轮 A 至少有 10 条公开验证乙醛样本，且父模型的乙醛召回低于 50%、至少一半乙醛都误判为同一类时，才把类级诊断和策略摘要加入同一次模型调用。来源及触发证据写入实验记录，教师模型和候选接口沿用现有配置。4 核、4 GiB 断网容器通过模拟网关首轮训练，实测触发并完成预测联调。真实网关行为和隐藏集收益待验证。镜像仅附研究策略摘要，未引入外部原始样本。[原始研究](https://doi.org/10.1016/j.dib.2015.01.003) · [SwanLab 入口检查](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/iz7e0dqu)

第二轮在公开训练批次拟合乙醛与乙醇的核模型。按类别把公开验证样本沿文件顺序分成前后两组，用前组筛选 128 种特征变换与参数组合，后组用来检查复现。相对传感器特征加核分类把起始模型在后组的 Macro F1 从 0.7260 提到 0.7832，找回 55 条乙醛中的 28 条，乙醛精确率为 35%，其中 33 条乙醇被误报为乙醛。十轮固定两 epoch 回放中，后组平均 Macro F1 从 0.7255 到 0.7826，乙醇 F1 从 0.6802 降到 0.6095。前组精确率门槛为 85% 的方案在后组零召回。该试验沿用现有镜像和公开数据，补充分支没有进入 Harness。复现脚本为 `deploy/contest/tests/a_acetaldehyde_kernel_probe.py`，可在 4 核、4 GiB 容器挂载后用 `PYTHONPATH=/app python /probe.py /data` 运行，`/data` 为挂载的官方 A 基线包。[SwanLab 第二轮对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/qhtsrrln)

### ABC 子群误差对照

用现有镜像在 4 核、4 GiB 的断网容器内训练和验证。候选只在公开训练数据拟合，验证数据用于比较。A 按类及文件顺序分三组，换三个训练种子。B、C 分别验证 3 根轴承与 106 台发动机，检查训练与验证的设备标识无交集，同时按剩余寿命拆分误差。

| 数据集 | 容易被总体指标掩盖的子群 | 固定方案的对照结果 |
| --- | --- | --- |
| A | 乙烯在种子 42 下有 27/334 条误判为乙醇。甲苯验证正例仅 5 条，种子 42 全部识别，但 4 条丙酮被误报为甲苯 | 按类别频次平方根平衡固定批次大小。三种子的乙醛召回仍为 0，最差顺序组的 Macro F1 从 0.6804 降到 0.6756 |
| B | Bearing3_4 占验证行 90.2%，平均低估约 444 分钟。接近失效的样本反而平均高估约 247 分钟 | 按工况收缩偏差使总体 RMSE 从 699.41 降到 694.03 分钟，近失效 RMSE 从 363.42 升到 366.43 分钟。等权线性校准的总体 RMSE 为 700.53 分钟 |
| C | FD004 的基准 RMSE 为 62.22 循环。剩余寿命至少 250 循环的样本平均低估 114.87 循环 | 按工况收缩偏差使近失效 RMSE 从 21.45 降到 18.85 循环，长寿命 RMSE 从 125.82 升到 130.55 循环。现有传感器校准的长寿命 RMSE 为 111.89 循环，但近失效 RMSE 为 22.04 循环 |

近失效切片定义为实际剩余寿命小于 25，B 的单位为分钟，C 的单位为循环。这些候选存在子群退化，未纳入交付镜像。A 的顺序组属于同一验证批次，B 只有三根独立验证轴承，结果不能外推为隐藏集收益。复现脚本为 `deploy/contest/tests/compare_subgroup_failures.py`，容器内以 `PYTHONPATH=/app python /probe.py /packages` 执行，`/packages` 为挂载的官方基线包父目录。[SwanLab ABC 子群对照](https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/ta99b9gb)
