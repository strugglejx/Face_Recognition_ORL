import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from method.PCA_ORL import myPCA
import time

in_features = 40
hidden_nums = 40
out_features = 40
epoch_nums = 50

class PCA_NN(nn.Module):

    def __init__(self, in_features, out_features, hidden_nums):
        super(PCA_NN, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_nums)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(in_features=hidden_nums, out_features=out_features)
        # self.sft = nn.LogSoftmax(dim=0)

    def forward(self, x):
        out = self.dense1(x)
        out = self.activation(out)
        out = self.dense2(out)
        # out = self.sft(out)
        return out


def nn_train_and_test(train_set, train_label, test_set, test_label):
    '''
        train by nn
    '''
    train_set = torch.from_numpy(train_set)
    train_label = torch.from_numpy(train_label)
    test_set = torch.from_numpy(test_set)

    # train
    model = PCA_NN(in_features, out_features, hidden_nums)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    def test(test, label, istrainset):
        pred = model(test).cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        precision = accuracy_score(y_pred=pred, y_true=label)
        if istrainset:
            print("epoch {} train acc: {}".format(epoch, precision))
        else:
            print("epoch {} test acc: {}".format(epoch, precision))
            return pred, precision

    for epoch in range(epoch_nums):
        for sample, label in zip(train_set, train_label.view(size=(-1, 1))):
            outputs = model(sample).view(size=(1, -1))
            optimizer.zero_grad()

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            test(train_set, train_label, istrainset=True)
            test(test_set, test_label, istrainset=False)

    s = time.time()
    test(train_set, train_label, istrainset=True)
    e = time.time()
    tim = e-s
    print("Total time of PAC_nn_ORL test: ", tim)
    pred, acc = test(test_set, test_label, istrainset=False)
    return pred, acc

def get_pca_vector(train_set, train_label, test_set, test_label, n_components=40):
    obj_pca = myPCA(trainset=train_set,
                         testset=test_set,
                         trainlabel=train_label,
                         testlabel=test_label,
                         n_components=n_components,
                         isKernel=False)

    train_test_set, _ = obj_pca.pca_acc(np.vstack((train_set, test_set)), n_components)
    train_set = train_test_set[0:360]
    test_set = train_test_set[360:400]

    return train_set, test_set

def train_pca_nn(train_set, train_label, test_set, test_label):
    train_set, test_set = get_pca_vector(train_set, train_label, test_set, test_label)

    pred, acc = nn_train_and_test(train_set, train_label, test_set, test_label)
    return pred, acc

if __name__ == "__main__":

    train_set = np.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [6, 7, 8, 9, 10], [6, 5, 4, 3, 2], [8, 7, 6, 5, 4]], dtype=np.float32)
    train_label = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    test_set = np.asarray([[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [7, 6, 5, 4, 3], [1, 3, 5, 7, 9]], dtype=np.float32)
    test_label = np.asarray([0, 0, 1, 0], dtype=np.int64)

    nn_train_and_test(train_set, train_label, test_set, test_label)
