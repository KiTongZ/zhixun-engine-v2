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
docker build --platform linux/amd64 -f deploy/contest/Dockerfile -t zhixun-contest:1.0.4-continuation .
docker run --rm zhixun-contest:1.0.4-continuation describe
docker save -o zhixun-contest-1.0.4-continuation.tar zhixun-contest:1.0.4-continuation
```

`describe` 应返回协议 `contest.harness.workspace.v2`、SDK 版本 `1.0.4` 和任务类型 `classification`、`regression`。将导出的 tar 上传到赛事平台，选择“基础模型（可正式参赛）”和 `Qwen/Qwen3.8-27B`，在平台密钥栏配置 PyTRiO API Key。平台通过镜像内的 `/app/harness.py` 调用训练和预测入口。

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
- **Harness 镜像**　`zhixun-contest-c-late-rate.tar`，Linux/amd64
- **实验归档**　[SwanLab · zhixun-engine-v2](https://swanlab.cn/@KiTongZ/zhixun-engine-v2)
- **项目源码**　[GitHub · KiTongZ/zhixun-engine-v2](https://github.com/KiTongZ/zhixun-engine-v2)






