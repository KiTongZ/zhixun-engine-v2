# readme-dsh.md · 2026-11-25 夜间改动记录

> 交接对象：GPT / 下一位接手人
> 范围：只记录本轮 DSH 在 Codex 工作目录里新增或修改的东西；没有改任何已有文件名。

## 1. 拿到正式分数后改了什么（最重要）

平台报告：**12.0001 / 35**，分项：

| 项 | 得分 | 读法 |
|---|---:|---|
| 平均性能提升 | 3.346 / 15 | 有增益，但离上界远 |
| 最差阶段表现 | 0.281 / 5 | 有一个阶段几乎没增益 |
| 恢复速度 | 3.373 / 5 | 中等 |
| 稳定性 | 5 / 5 | 满分，别动 |
| **资源效率** | **0 / 5** | 最大短板：A 任务 10 轮用了 1347 秒 |

Token 账单：输入 16,526 / 输出 9,600，30 轮平均 **每轮输出 320 token**——正好顶满上限。A 的 1347 秒几乎全耗在每轮 27B 解码上，训练本身只有几秒。

所以这轮的第一优先级不是再微调模型，而是**把每轮决策的解码成本砍掉**。已改：

1. 每轮只发一个极短状态：`{"d":"A","r":3,"t":"classification","n":445,"pe":20,"pv":0.9782,"e0":2,"l0":0.003}`；
2. 只要求回 `{"e":1-3,"l":float}`，**max_tokens 从 320 降到 24**，并显式 `enable_thinking=False`；
3. 预算仍落在原先验证过的推荐值上（A 默认、B 第 1 轮 3×0.006、B 后 9 轮 1/32、C 第 1 轮 1×0.001、C 后 5 轮 0.004、C 第 10 轮头回归 1e-8），模型选择只作为微调，解析失败就用推荐值；
4. 训练轮数、采样比例、检查点选择逻辑一律不动，稳定性 5/5 不能丢。

预期：输出 token 30 轮从 9,600 降到约 720（−92%），输入也大幅缩短；A 的 1347 秒应显著下降，资源效率 0/5 有希望拿分。

## 2. 这轮本地验证（得分前做的）

| 实验 | 对照 | 结果 | 结论 |
|---|---|---|---|
| C 尾部风险 `moderate_update` | 第 10 轮 base / refine / moderate | held RMSE：base 54.29，refine **53.77**，moderate 54.91 | 否决，保留 `refine_head` |
| B 采样预算 1/16 vs 1/32 | `sixteenth_1_16` vs `tiny_1_32` | 平均 473.56 vs **469.25**；最差 495.55 vs **486.67**；MAE 405.17 vs **393.76** | 否决 1/16，保留 1/32 |
| C 输出层惩罚 | 1e-3 → 1e-8 | held RMSE **53.77 → 53.22**，MAE **36.66 → 35.98**，NASA 代理 **385k → 325k** | 采纳，已进生产 |

低 token 改造后本地复测：A 一轮 Macro-F1 0.7140 → **0.7257**；B 一轮 RMSE 681.16 → **619.83**，协议与回执正常。

## 3. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `deploy/contest/contest_submission.py` | ① 决策提示词改为极短 JSON，`max_tokens=24`、`enable_thinking=False`；② C 第 10 轮头回归惩罚 `1e-3 → 1e-8` |
| `deploy/contest/tests/compare_engine_tail.py` | 新增：C 尾部风险对照 |
| `deploy/contest/tests/compare_b_budget_tail.py` | 新增：B 1/32 vs 1/16 十轮回放 |
| `deploy/contest/tests/sweep_engine_penalty.py` | 新增：C 头回归惩罚扫描 |
| `deploy/contest/tests/*_results.json` | 新增：三组实验结果 |
| `dist/zhixun-contest-c-head-tuned.tar` | 新增镜像（C 惩罚版，62.0 MB） |
| `dist/zhixun-contest-lean.tar` | 新增镜像（低 token 版，62.0 MB） |
| `readme-dsh.md` | 本文件 |

已有文件名、`05_补充资料` 的 7 栏材料、SwanLab 项目结构都没有动。

## 4. 现在有三个 tar，怎么用

| 优先 | 文件 | 内容 | 建议 |
|---|---|---|---|
| **首选** | `zhixun-contest-lean.tar`（62.0 MB，SHA-256 `AD92AA9B…53F8`） | 低 token + C 惩罚 1e-8 + B 1/32 + 特征缓存 | 直接作为第一次或第二次评测；预期资源分上升 |
| 备用 | `zhixun-contest-c-head-tuned.tar`（62.0 MB，SHA-256 `406C605E…875F`） | 只改 C 惩罚，token 仍是旧版 320 | 若 lean 的短 JSON 被平台网关拒绝，用它 |
| 已给 | `zhixun-contest-feature-cache.tar`（62.07 MB，SHA-256 `166C0627…A2A9`） | 无 C 惩罚、无低 token | 仅在需要复现 12.0001 基线时用 |

## 5. 记录

* SwanLab 本地实验：<https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/ds96l80g>
* SwanLab 低 token 复测：<https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/0ilg5l4p>
* GitHub：本轮改动已提交并推送（见仓库 `main` 最新提交）
* 数据：官方 baseline 的 `public/train.jsonl`、`public/validation.jsonl`，按整台发动机/整根轴承留出

## 6. 风险与回退

1. 平台网关若不接受 `enable_thinking` 参数，调用会报错——回退用 `c-head-tuned`（旧参数）；
2. 短 JSON 若被截断，解析失败自动用推荐预算，训练仍继续，不会空转；
3. 训练预算未改，稳定性 5/5 的逻辑未动；
4. C 惩罚 1e-8 是扫描最优附近；若隐藏集 RMSE 反而变差，回退 `feature-cache` 的 1e-3；
5. 不改文件名，新增文件都列在第 3 节。
