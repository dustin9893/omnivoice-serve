FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install OmniVoice from source
RUN git clone https://github.com/k2-fsa/OmniVoice /tmp/omnivoice \
    && pip install -e /tmp/omnivoice

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .

EXPOSE ${WORKER_PORT:-9000}

CMD ["python", "model_server.py"]
