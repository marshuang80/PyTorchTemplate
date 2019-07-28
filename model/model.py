import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        #layers = []
        #self.model = nn.Sequential(*layers)


    def forward(self, x):

        pred = self.model(x)

        return pred


    def args_dict(self):

        # TODO: add all model init parameters
        model_args = {}

        return model_args
