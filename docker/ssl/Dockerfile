# GPU image for SSL pretraining / fine-tuning of the ResNet-3D ESP encoder.
# Separate from the top-level `quannet:latest` image, which is CPU-only and
# only runs APBS/pdb2pqr for building ESP tensors - this one needs PyTorch
# with CUDA and is meant to run on a GPU server, not the data-prep laptop.
#
# Build:  docker build -t quannet-ssl -f docker/ssl/Dockerfile .
# Run (pretrain / finetune - one image, dispatched by the first argument):
#   docker run --gpus all \
#     -v /path/to/esp_caches:/data:ro \
#     -v /path/to/runs:/app/runs \
#     -e CLEARML_API_HOST=... -e CLEARML_WEB_HOST=... -e CLEARML_FILES_HOST=... \
#     -e CLEARML_API_ACCESS_KEY=... -e CLEARML_API_SECRET_KEY=... \
#     quannet-ssl pretrain --method vicreg --data-dirs /data/sabdab /data/oas \
#                 --output-dir /app/runs/ssl_vicreg --accelerator gpu \
#                 --clearml-project quannet-ssl
#   docker run --gpus all ... quannet-ssl finetune --ssl_checkpoint /app/runs/ssl_vicreg/encoder.pt
# ClearML credentials are read from the environment above (or a mounted
# clearml.conf) at run time - never bake them into the image.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# torch is already in the base image (with CUDA) - install the rest of the
# host-side model requirements without letting pip pull in a CPU-only torch.
RUN pip install --no-cache-dir \
    pyyaml==6.0.1 pandas==2.0.3 lightning==2.0.7 "clearml>=1.16,<2" \
    "transformers>=4.30,<4.46" "scikit-learn>=1.3,<2" "scipy>=1.10,<2" "numpy<2"

COPY pyproject.toml setup.py setup.cfg README.md ./
COPY requirements ./requirements
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY scripts ./scripts

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "scripts/ssl_entrypoint.py"]
