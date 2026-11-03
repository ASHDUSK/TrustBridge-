# TrustBridge 容器骨架（开发中，当前官方支持 Windows 本地运行）
# 构建: docker build -t trustbridge .
# 运行: docker run --gpus all -p 7860:7860 -p 8000:8000 trustbridge
# 说明: 需要宿主机 NVIDIA Container Toolkit；权重与数据通过 -v 挂载：
#   -v D:\MedTranslateProject\checkpoints:/app/checkpoints
#   -v D:\MedTranslateProject\data:/app/data
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY trustbridge/ ./trustbridge/
COPY app/ ./app/
COPY api.py ./
COPY scripts/ ./scripts/

EXPOSE 7860 8000
CMD ["python", "app/app.py"]
