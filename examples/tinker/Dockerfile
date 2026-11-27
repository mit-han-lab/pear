FROM ubuntu:24.04@sha256:562111b88caa79c71391c0c4d3c2114a7d5c35915c20561fd893cb1e07929e6b

ENV WORK_DIR=/work USER=root HF_HOME=/work/cache/huggingface
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git patch libglib2.0-0t64 libgl1 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY pear/ pear/
COPY scripts/ scripts/
COPY examples/tinker/ examples/tinker/
RUN bash scripts/install.sh tinker && bash scripts/install.sh libero

ENTRYPOINT ["bash", "examples/tinker/run_libero.sh"]
CMD ["--help"]
