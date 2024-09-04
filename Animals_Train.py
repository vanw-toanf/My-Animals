"""
    @author: Van Toan <damtoan321@gmail.com>
"""
import torch
import os
import shutil
import torch.nn as nn
from torch.optim import SGD, Adam
from Animals_Dataset import myAnimalsDataset
from MyCNNmodel import myCNN
from torchvision.transforms.v2 import Compose, ToTensor, Resize, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm
import torchvision.models as models
# import warnings
# warnings.filterwarnings("ignore")   # tat warning


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20,20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > threshold else "black"
            plt.text(j, i, cm[i,j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_arg():
    parse = argparse.ArgumentParser(description='Animal Classifion')
    parse.add_argument('-p', '--data_path', type=str, default="Dataset/animals")
    parse.add_argument('-b', '--batch_size', type=int, default=16)
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument('-l', '--lr', type=float, default=1e-3)
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--train_path', type=str, default="trained_models")
    args = parse.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

    train_transform = Compose([
        ToTensor(),

        # Data Augmentation
        RandomAffine(
            degrees=5,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05,
        ),
        Resize((args.image_size, args.image_size)),
    ])
    val_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
    ])
    train_set = myAnimalsDataset(root=args.data_path, train=True, transform=train_transform)
    valid_set = myAnimalsDataset(root=args.data_path, train=False, transform=val_transform)

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 6,
    }
    valid_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 6,
    }
    train_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **valid_params)

    model = myCNN(num_class=len(train_set.categories)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)    #sau 30 giam, sau 60e giam tiep...

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.train_path):
        os.mkdir(args.train_path)
    writer = SummaryWriter(args.tensorboard_path)

    num_iters = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        model.train()

        losses = []
        progress_bar = tqdm(train_dataloader, colour="cyan")

        for iter, (image, label) in enumerate(progress_bar):

            # move tensor to graphic
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            predict = model(image)
            loss = criterion(predict, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

            progress_bar.set_description("Epoch {}/{}. Loss value: {:.4f}".format(epoch + 1, args.epochs, loss_value))
            losses.append(loss_value)
            writer.add_scalar("Train/Loss", np.mean(losses), iter + epoch*num_iters)
                                            #chi ghi gia tri mean tinh den tung epoch

        # VALIDATE
        model.eval()
        losses = []
        all_predict = []
        all_gts = []
        with torch.no_grad():
            for iter, (image, label) in enumerate(valid_dataloader):
                # move tensor to graphic
                image = image.to(device)
                label = label.to(device)

                predict = model(image)
                max_idx = torch.argmax(predict, 1)
                loss = criterion(predict, label)
                losses.append(loss.item())
                all_gts.extend(label.tolist())
                all_predict.extend(max_idx.tolist())

        writer.add_scalar("Valid/Loss", np.mean(losses), epoch)
        acc = accuracy_score(all_gts, all_predict)
        writer.add_scalar("Valid/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_gts, all_predict)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.categories))], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }

        torch.save(checkpoint, os.path.join(args.train_path, "model.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.train_path, "best.pt"))
            best_acc = acc

        scheduler.step()


if __name__ == '__main__':
    args = get_arg()
    train(args)
