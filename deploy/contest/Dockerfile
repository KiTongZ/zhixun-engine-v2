FROM python:3.12-slim

LABEL org.opencontainers.image.title="zhixun-contest-harness" \
      org.opencontainers.image.description="Continuation submission on the official CPU baseline (workspace v2)" \
      org.opencontainers.image.version="1.0.4"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY deploy/contest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -rf /root/.cache/pip

# contest_harness SDK (workspace v2, chunked upload host)
COPY vendor/contest-harness-sdk-1.0.4 /tmp/contest-sdk
RUN pip install --no-cache-dir /tmp/contest-sdk \
 && rm -rf /tmp/contest-sdk

# official baseline code + A/B/C configs (model.py / data.py come from the package)
COPY deploy/contest/baselines /app/baselines

# fixed entry + submission
COPY deploy/contest/harness.py /app/harness.py
COPY deploy/contest/bearing_sampling.py /app/bearing_sampling.py
COPY deploy/contest/contest_submission.py /app/contest_submission.py

RUN python -c "import contest_harness, numpy; from baselines.model import Model; print('sdk+numpy+baseline ok')"

ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]

