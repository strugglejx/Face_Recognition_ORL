import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from method.PCA_ORL import myPCA
import xgboost as xgb


in_features = 40
hidden_nums = 40
out_features = 40
epoch_nums = 50

def xgb_func(data_tv, label_tv):
    nfold = 10

    folder = StratifiedKFold(n_splits=nfold,
                             shuffle=True,
                             random_state=200)

    pcs_list = []
    count = 0
    for train_index, val_index in folder.split(X=data_tv, y=label_tv):
        data_train, label_train = data_tv[train_index], label_tv[train_index]
        data_val, label_val = data_tv[val_index], label_tv[val_index]

        dl_train = xgb.DMatrix(data=data_train, label=label_train)
        dl_val = xgb.DMatrix(data=data_val, label=label_val)

        # 参数设置
        params = {'booster': 'gbtree',
                  'objective': 'multi:softmax',
                  'eval_metric': 'mlogloss',
                  'max_depth': 8,
                  'lambda': 1,
                  'subsample': 0.75,
                  'colsample_bytree': 0.75,
                  'min_child_weight': 1,
                  'eta': 0.025,
                  'gamma': 0.4,
                  'seed': 0,
                  'nthread': 8,
                  'silent': 0,
                  'reg_alpha': 0.1,
                  'num_class': 40}

        watchlist = [(dl_train, 'train')]

        # val
        bst = xgb.train(params, dl_train, num_boost_round=500, evals=watchlist)
        pred_val = bst.predict(dl_val)
        # pred_val = (pred_val >= 0.6) * 1
        precision = precision_score(y_pred=pred_val, y_true=label_val, average='macro')
        pcs_list.append(precision)
        print("cv {}  precision: {}".format(count, precision))
        count += 1


    print("precision in cv: ", pcs_list)
    print("mean precision: ", sum(pcs_list)/nfold)




def get_pca_vector(train_set, train_label, test_set, test_label, n_components=50):
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

def train_pca_xgb(train_set, train_label, test_set, test_label):
    train_set, test_set = get_pca_vector(train_set, train_label, test_set, test_label)

    all_train_set = np.concatenate((train_set, test_set), axis=0)
    all_test_set = np.concatenate((train_label, test_label), axis=0)
    print(all_train_set.shape, all_test_set.shape)

    xgb_func(all_train_set, all_test_set)
    # return pred, acc

if __name__ == "__main__":

    train_set = np.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [6, 7, 8, 9, 10], [6, 5, 4, 3, 2], [8, 7, 6, 5, 4]], dtype=np.float32)
    train_label = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    test_set = np.asarray([[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [7, 6, 5, 4, 3], [1, 3, 5, 7, 9]], dtype=np.float32)
    test_label = np.asarray([0, 0, 1, 0], dtype=np.int64)
    #
    # # nn_train_and_test(train_set, train_label, test_set, test_label)
    # train_pca_xgb(train_set, train_label, test_set, test_label)