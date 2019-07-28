import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import logger
import os
import utils
from dataset import Dataset
from model import Model
from torch.utils.data import DataLoader

def main(args):

    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # dataloader
    if args.train_data is not None:
        train_dataloader = util.get_dataloader(args.train_data, args.batch_size, args.shuffle)
        train_size = len(train_dataloader.dataset)
    if args.dev_data is not None:
        dev_dataloader = util.get_dataloader(args.dev_data, args.batch_size)
        dev_size = len(dev_dataloader.dataset)

    # device
    device = torch.device(args.device)

    # initialize model
    model = Model()

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss function
    loss_function = None

    # train model
    model = model.to(device)

    itr = 0
    dev_loss_min = float("inf")
    for epoch in range(args.epoch):

        # train 
        model.train()
        with tqdm.tqdm(total=train_size, unit=f" [TRAIN] epoch {epoch} itr") as progress_bar:
            for i, (x_train, y_train) in enumerate(train_dataloader):

                with torch.set_grad_enabled(True):

                    # send data and label to device
                    x = torch.Tensor(x_train).to(device)
                    y = torch.Tensor(y_train).to(device)

                    # predict
                    pred = model.forward(x)

                    # calculate loss
                    loss = loss_function(pred, y)
                    logger_tb.update_loss('train loss', loss.item(), itr)
                    itr += 1

                    # back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update(len(x))

        # dev
        model.eval()
        with tqdm.tqdm(total=dev_size, unit=f" [DEV] epoch {epoch} itr") as dev_progress_bar:
            dev_loss = []
            for i, (x_dev, y_dev) in enumerate(dev_dataloader):

                with torch.no_grad():

                    # send data and label to device
                    x = torch.Tensor(x_dev).to(device)
                    y = torch.Tensor(y_dev).to(device)

                    # predict
                    pred = model.forward(x)

                    # calculate loss
                    dev_loss.append(loss_function(pred, y).item())

                dev_progress_bar.update(len(x))

            # log on tensorboard
            dev_loss_avg = sum(dev_loss) / len(dev_loss)
            logger_tb.update_loss('dev loss', dev_loss_avg, epoch)

        # save learned parameters
        if dev_loss_avg < dev_loss_min:
            ckpt_dict = {'model_name': model.__class__.__name__, 
                         'model_args': model.args_dict(), 
                         'model_state': model.state_dict()}
            ckpt_path = os.path.join(args.save_dir, f"{model.__class__.__name__}_best.pth")
            torch.save(ckpt_dict, ckpt_path)
            dev_loss_min = dev_loss_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--dev_data", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--save_dir", type=str, default="./ckpts/")
    parser.add_argument("--experiment_name", type=str, default="test")

    args = parser.parse_args()

    main(args)
