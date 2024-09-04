FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /animal

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 vim -y

RUN pip3 install matplotlib scikit-learn numpy tensorboard opencv-python torchsummary

COPY Animals_Train.py Animals_Train.py
COPY Animals_Dataset.py Animals_Dataset.py
COPY MyCNNmodel.py MyCNNmodel.py

# Train tiep
CMD ["python3", "Animals_Train.py", "-c trained_models/model.pt"]
