FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /animal

RUN apt-get update && apt install -y gcc
RUN apt-get install ffmpeg libsm6 libxext6 vim -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "6", "--reload"]
