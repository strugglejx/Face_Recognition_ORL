import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from method.PCA_ORL import myPCA

in_features = 40
hidden_nums = 40
out_features = 40
epoch_nums = 50

class PCA_CNN(nn.Module):

    def __init__(self, in_features, out_features, hidden_nums):
        super(PCA_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.dense1 = nn.Linear(in_features=16*14*11,
                                out_features=out_features)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)


    def forward(self, x):
        outputs = self.conv1(x)
        outputs = F.relu((outputs))
        outputs = self.bn1(outputs)
        outputs = F.max_pool2d(outputs, 2)

        outputs = self.conv2(outputs)
        outputs = F.relu((outputs))
        outputs = self.bn2(outputs)
        outputs = F.max_pool2d(outputs, 2)

        outputs = self.conv3(outputs)
        outputs = F.relu((outputs))
        outputs = self.bn3(outputs)
        outputs = F.max_pool2d(outputs, 2)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.dense1(outputs)
        return outputs


def cnn_train_and_test(train_set, train_label, test_set, test_label):
    '''
        train by nn
    '''

    train_set = torch.from_numpy(train_set).to("cuda")
    train_label = torch.from_numpy(train_label).to("cuda")
    test_set = torch.from_numpy(test_set).to("cuda")
    test_label = torch.from_numpy((test_label)).to("cuda")

    # train
    model = PCA_CNN(in_features, out_features, hidden_nums).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    def test(test, label, istrainset):
        pred = model(test).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)

        precision = accuracy_score(y_pred=pred, y_true=label)
        if istrainset:
            print("epoch {} train acc: {}".format(epoch, precision))
        else:
            print("epoch {} test acc: {}".format(epoch, precision))
            return pred, precision

    for epoch in range(epoch_nums):
        for sample, label in zip(train_set, train_label.view(size=(-1, 1))):
            sample = sample.unsqueeze(0)
            outputs = model(sample)  # .view(size=(1, -1))
            optimizer.zero_grad()

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            test(train_set, train_label, istrainset=True)
            test(test_set, test_label, istrainset=False)

    test(train_set, train_label, istrainset=True)
    pred, acc = test(test_set, test_label, istrainset=False)
    return pred, acc


def train_pca_cnn(train_set, train_label, test_set, test_label):
    train_label = train_label - 1    # from zero
    test_label = test_label - 1  # from zero

    print(train_set.shape)
    print(train_label.shape)
    print(test_set.shape)
    print(test_label.shape)

    pred, acc = cnn_train_and_test(train_set, train_label, test_set, test_label)
    return pred, acc

if __name__ == "__main__":

    # train_set = np.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [6, 7, 8, 9, 10], [6, 5, 4, 3, 2], [8, 7, 6, 5, 4]], dtype=np.float32)
    # train_label = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    # test_set = np.asarray([[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [7, 6, 5, 4, 3], [1, 3, 5, 7, 9]], dtype=np.float32)
    # test_label = np.asarray([0, 0, 1, 0], dtype=np.int64)

    train_set = np.ones(shape=(10, 1, 8, 8), dtype=np.float32)  # 10张图片，每张图片8*8
    train_label = np.asarray([0, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)
    test_set = np.ones(shape=(10, 1, 8, 8), dtype=np.float32)  # 10张图片，每张图片8*8
    test_label = np.asarray([0, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)

    cnn_train_and_test(train_set, train_label, test_set, test_label)

