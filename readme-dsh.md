# readme-dsh.md · 2026-11-25 夜间改动记录

> 交接对象：GPT / 下一位接手人
> 范围：只记录本轮 DSH 在 Codex 工作目录里新增或修改的东西；没有改任何已有文件名。

## 1. 这轮做了什么

Codex 最后停在 T57（发动机 batch/尾部实验，没留结论）。我接着做了三组本地验证，全部在官方公开训练/验证集上、按整台发动机/整根轴承留出：

| 实验 | 对照 | 结果 | 结论 |
|---|---|---|---|
| C 尾部风险 `moderate_update` | 第 10 轮 base / refine / moderate | held RMSE：base 54.29，refine **53.77**，moderate 54.91（fraction=0.125） | 否决，保留现成 `refine_head` |
| B 采样预算 1/16 vs 1/32 | `sixteenth_1_16` vs `tiny_1_32` | 平均 RMSE 473.56 vs **469.25**；最差 495.55 vs **486.67**；MAE 405.17 vs **393.76** | 否决 1/16，保留 1/32 |
| C 输出层惩罚系数 | 0 / 1e-8 / 1e-7 / 1e-6 / 1e-5 / 1e-4 / 1e-3 | 1e-8：held RMSE **53.22**、MAE **35.98**、asym **325,097**；原 1e-3：53.77 / 36.66 / 385,325 | 采纳 1e-8 |

第三组是意外收获：把第 10 轮 `refine_head` 的 ridge 惩罚从 `1e-3` 降到 `1e-8`，在留出验证上 RMSE 降约 1.0%、MAE 降约 1.8%、NASA 非对称代理降约 15.6%，且选择集（select）也同步变好，不是过拟合单侧。

## 2. 改了哪个文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `deploy/contest/contest_submission.py` | 第 265 行：`penalty=.001` → `penalty=1e-8` | 只影响 C 任务第 10 轮的头回归；其他逻辑不动 |
| `deploy/contest/tests/compare_engine_tail.py` | 新增 | C 尾部风险对照（parent/base/refine/moderate），输出 `c_tail_results.json` |
| `deploy/contest/tests/compare_b_budget_tail.py` | 新增 | B 1/32 vs 1/16 十轮回放，输出 `b_budget_tail_results.json` |
| `deploy/contest/tests/sweep_engine_penalty.py` | 新增 | C 头回归惩罚扫描，输出 `engine_penalty_results.json` |
| `dist/zhixun-contest-c-head-tuned.tar` | 新增镜像导出 | 62.0 MB，SHA-256 `406C605E…875F` |
| `readme-dsh.md` | 本文件 | 交接说明 |

生产代码里 Codex 留下的未提交改动（`selected_metrics` 修正、`moderate_update`）保留不动；`moderate_update` 只被测试脚本引用，不参与线上。

## 3. 两个 tar 怎么用

| 顺序 | 文件 | 与上一版的差别 | 用途 |
|---|---|---|---|
| 第一次评测 | `zhixun-contest-feature-cache.tar`（62.07 MB，SHA-256 `166C0627…A2A9`） | B 1/32 小预算 + C 后期学习率/头回归（旧惩罚）+ 特征缓存 | 已验证，可直接上平台 |
| 第二次评测 | `zhixun-contest-c-head-tuned.tar`（62.0 MB，SHA-256 `406C605E…875F`） | 与第一次只差：C 第 10 轮头回归惩罚 `1e-3 → 1e-8` | 等第一次出分后再上，对照 C 的 RMSE/NASA 分数 |

两个 tar 都在桌面根目录各放了一份，原始位置在 `03_阶段性成果\镜像\`。

## 4. 记录

* SwanLab 本轮本地实验：<https://swanlab.cn/@KiTongZ/zhixun-engine-v2/runs/ds96l80g>
* 数据：`02_项目源码与文档\AI+项目\vendor\baselines\cpu-c-guide-20261110-v1`、`cpu-b-rul-20261110-v1`
* 留出方式：C 按发动机哈希取 1/3 留出；B 按整根轴承留出 3 折 + 全量公开集
* 所有脚本都只读官方公开数据，不碰隐藏集

## 5. 注意

1. `moderate_update` 在本轮数据上会让 RMSE 变差（fraction 选中 0.125 仍不如 refine），不要接进线上；
2. B 1/16 在平均、最差、MAE 三项都输给 1/32，不要改回 1/16；
3. C 惩罚 1e-8 是扫描中最优附近，但更小的 0/1e-7 差异在噪声内；选 1e-8 是为了数值稳定；
4. 两个 tar 的镜像 tag 分别是 `zhixun-contest:feature-cache` 与 `zhixun-contest:c-head-tuned`，平台上传后按 tag 选，不要选错；
5. 没有改 `05_补充资料` 里的 7 栏材料，也没有改任何已有文件名。
