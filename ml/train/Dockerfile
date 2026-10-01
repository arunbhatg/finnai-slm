FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/ml/code
COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

COPY sft_qlora.py merge_lora.py train_config.yaml ./
ENV PYTHONUNBUFFERED=1
