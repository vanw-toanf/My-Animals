"""
    @author: Van Toan <damtoan321@gmail.com>
"""
import os
import numpy as np
import torch
import cv2
import argparse
from MyCNNmodel import myCNN
import torch.nn as nn
import matplotlib.pyplot as plt


def get_arg():
    parse = argparse.ArgumentParser(description='Animal Classification')
    parse.add_argument('-s', '--size', type=int, default=224)
    parse.add_argument('-i', '--image_path', type=str, default="test_images/test_1.jpeg")
    parse.add_argument('-c', '--checkpoint_path', type=str, default="trained_models/best.pt")
    args = parse.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    model = myCNN(num_class=len(categories)).to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provided!")
        exit(0)

    if not args.image_path:
        print("An image must be provided!")
        exit(0)

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (args.size, args.size))
    image = np.transpose(image, (2, 0, 1))
    image = image / 255
    # image = np.expand_dims(image, 0)    # them 1 chieu vao dau do model chi nhan 4 chieu
    # image = torch.from_numpy(image).to(device).float()
    image = torch.from_numpy(image).to(device).float()[None, :, :, :]
    softmax = nn.Softmax()

    with torch.no_grad():
        predict = model(image)
    probs = softmax(predict)
    max_value, max_index = torch.max(probs, dim=1)
    print("This image is about {} with probability of {}". format(categories[max_index], max_value[0].item()))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(categories, probs[0].cpu().numpy())
    ax.set_xlabel("Animals")
    ax.set_ylabel("Probability")
    ax.set_title(categories[max_index])
    plt.savefig("Animal_prediction.png")


if __name__ == '__main__':
    args = get_arg()
    test(args)
