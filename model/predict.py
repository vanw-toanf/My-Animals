import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from model.MyCNNmodel import myCNN


def img_test(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    model = myCNN(num_class=len(categories)).to(device)

    checkpoint_path = "model/trained_models/best.pt"
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provided!")
        exit(0)

    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.
    image = torch.from_numpy(image).to(device).float()[None, :, :, :]
    softmax = nn.Softmax()

    with torch.no_grad():
        predict = model(image)
    probs = softmax(predict)
    max_value, max_index = torch.max(probs, dim=1)
    return categories[max_index], max_value[0].item()