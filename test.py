import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import utils
from dataset import Dataset
from model import Model
from torch.utils.data import DataLoader

def main(args):

    # test dataloader
    test_dataloader = utils.get_dataloader(args.test_data, args.batch_size)
    test_size = len(test_dataloader.dataset)

    # device
    device = torch.device(args.device)

    # initialize model
    ckpt = torch.load(args.ckpt)
    model_args = ckpt['model_args']
    model_state = ckpt['model_state']
    model = Model(**model_args)
    model.load_state_dict(model_state)

    # loss function 
    loss_function = nn.MSELoss()

    # train model
    model.eval()
    model = model.to(device)

    # test
    with tqdm.tqdm(total=test_size, unit=f" [TEST] itr") as test_progress_bar:
        test_loss = []
        for i, (x_test, y_test) in enumerate(test_dataloader):

            with torch.no_grad():

                # send data and label to testice
                x = torch.Tensor(x_test).to(device)
                y = torch.Tensor(y_test).to(device)

                # predict
                pred = model.forward(x)

                # calculate loss
                test_loss.append(loss_function(pred, y).item())

            test_progress_bar.update(len(x))

        # log on tensorboard
        test_loss_avg = sum(test_loss) / len(test_loss)

    print(f"Average Test Loss: {test_loss_avg}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="./ckpts/Model_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default="Test")

    args = parser.parse_args()

    main(args)
