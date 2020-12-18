import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from method.PCA_ORL import myPCA
import xgboost as xgb


in_features = 40
hidden_nums = 40
out_features = 40
epoch_nums = 50

def xgb_func(data_tv, label_tv, data_test):
    nfold = 5

    folder = StratifiedKFold(n_splits=nfold,
                             shuffle=True,
                             random_state=200)

    pred_cv = np.zeros(shape=(len(data_test)))

    pcs_list = []
    for train_index, val_index in folder.split(X=data_tv, y=label_tv):
        data_train, label_train = data_tv[train_index], label_tv[train_index]
        data_val, label_val = data_tv[val_index], label_tv[val_index]

        dl_train = xgb.DMatrix(data=data_train, label=label_train)
        dl_val = xgb.DMatrix(data=data_val, label=label_val)
        d_test = xgb.DMatrix(data=data_test)

        # 参数设置
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'map',
                  'max_depth': 4,
                  'lambda': 10,
                  'subsample': 0.75,
                  'colsample_bytree': 0.75,
                  'min_child_weight': 2,
                  'eta': 0.025,
                  'gamma': 0.4,
                  'seed': 0,
                  'nthread': 8,
                  'silent': 0,
                  'reg_alpha': 0.4}

        watchlist = [(dl_train, 'train')]

        # val
        bst = xgb.train(params, dl_train, num_boost_round=59, evals=watchlist)
        pred_val = bst.predict(dl_val)
        pred_val = (pred_val >= 0.6) * 1
        precision = precision_score(y_pred=pred_val, y_true=label_val)
        pcs_list.append(precision)

        # test
        pred_test = bst.predict(d_test)
        pred_cv += pred_test / nfold

    print("precision in cv: ", pcs_list)
    print("mean precision: ", sum(pcs_list)/nfold)

    print(pred_cv)
    result = [1 if i > 0.55 else 0 for i in pred_cv]
    print("positive samples: ", sum(result))
    id = range(210, 314)
    df_res = pd.DataFrame({
        'ID': id,
        'CLASS': result
    })
    df_res.to_csv("res4_cv.csv", index=False)

    # train by all data
    dl_tv = xgb.DMatrix(data=data_tv, label=label_tv)
    bst = xgb.train(params, dl_tv, num_boost_round=59, evals=watchlist)
    pred_test = bst.predict(d_test)
    result = [1 if i > 0.55 else 0 for i in pred_test]
    print("positive samples: ", sum(result))
    df_res = pd.DataFrame({
        'ID': id,
        'CLASS': result
    })
    df_res.to_csv("res4.csv", index=False)


def nn_train_and_test(train_set, train_label, test_set, test_label):
    '''
        train by nn
    '''
    train_set = torch.from_numpy(train_set)
    train_label = torch.from_numpy(train_label)
    test_set = torch.from_numpy(test_set)

    # train
    model = XGBoost(in_features, out_features, hidden_nums)
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

    test(train_set, train_label, istrainset=True)
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
