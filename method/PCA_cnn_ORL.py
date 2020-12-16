import torch
import torch.nn as nn
import numpy as np

class PCA_CNN(nn.Module):

    def __init__(self, in_features, out_features, hidden_nums):
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_nums)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(in_features=hidden_nums, out_features=out_features)

    def forward(self, x):
        out = self.dense1(x)
        out = self.activation(out)
        out = self.dense1(out)

        return out


if __name__ == "__main__":

    train_set = np.asarray([[1,2,3,4,5], [2,3,4,5,6], [6,7,8,9,10], [6,5,4,3,2], [8,7,6,5,4]])
    train_label = np.asarray([0,0,0,1,1])
    test_set = np.asarray([[4,5,6,7,8], [5,6,7,8,9], [7,6,5,4,3], [1,3,5,7,9]])
    test_label = np.asarray([0,0,1,0])



