# 智训引擎
赛题九 Harness 提交源码。A、B、C 任务从平台提供的官方检查点续训，分别执行气体分类、轴承剩余寿命预测和发动机剩余寿命预测。
## 构建和验证
在仓库根目录运行以下命令。
~~~sh
docker build --platform linux/amd64 -f deploy/contest/Dockerfile -t zhixun-contest:1.0.4-continuation .
docker run --rm zhixun-contest:1.0.4-continuation describe
~~~
训练入口和预测入口分别为容器内的 harness.py train 和 harness.py predict，由比赛宿主调用。SDK 版本为 1.0.4。模型 API Key 在赛事平台配置。
