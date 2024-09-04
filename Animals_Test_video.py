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


def get_arg():
    parse = argparse.ArgumentParser(description='Animal Classification')
    parse.add_argument('-s', '--size', type=int, default=224)
    parse.add_argument('-i', '--video_path', type=str, default="test_video/test_1.mp4")
    parse.add_argument('-o', '--video_output_path', type=str, default="test_video/test_1_output.mp4")
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

    if not args.video_path:
        print("An image must be provided!")
        exit(0)

    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(args.video_output_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    counter = 0
    while cap.isOpened():
        # print frame
        print(counter)
        counter += 1

        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.resize(frame, (args.size, args.size))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = torch.from_numpy(image).to(device).float()[None, :, :, :]
        softmax = nn.Softmax()
        with torch.no_grad():
            predict = model(image)
        probs = softmax(predict)
        max_value, max_index = torch.max(probs, dim=1)
        category = categories[max_index]
        cv2.putText(frame, category, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    args = get_arg()
    test(args)
