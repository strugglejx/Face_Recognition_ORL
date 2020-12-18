from read_utils.Read_pic import Read
from method.PCA_ORL import myPCA
from method.PCA_nn_ORL import *
from method.PCA_nn_ORL import train_pca_nn
import os

test_rate = 0.1
dir_path = os.path.join(os.getcwd(), "data_faces")

if __name__ == "__main__":
    # read picture
    r = Read(dir_path)
    train_set, train_label, test_set, test_label = r.read_rt_tt(isflatten=False, test_rate=0.1)

    print(train_set.shape)
    print(train_label.shape)
    print(test_set.shape)
    print(test_label.shape)

    # test_train_split
    # PCA
    # mthd_PCA = myPCA(trainset=train_set,
    #                  testset=test_set,
    #                  trainlabel=train_label,
    #                  testlabel=test_label,
    #                  n_components=200,
    #                  isKernel=False)
    # mthd_PCA.train()
    # mthd_PCA.test()

    # PCA_ORL(cv)
    # pred_list = []
    # kf, all_pic_list, all_label_list = r.read_rt_cv(isflatten=True, test_rate=0.1)
    # for train_id, test_id in kf.split(all_pic_list):
    #     train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
    #     test_set, test_label = all_pic_list[test_id], all_label_list[test_id]
    #
    #
    #     mthd_PCA = myPCA(trainset=train_set,
    #                      testset=test_set,
    #                      trainlabel=train_label,
    #                      testlabel=test_label,
    #                      n_components=20,
    #                      isKernel=False)
    #     get_pca_vector(train_set, train_label, test_set, test_label)
    #
    #     mthd_PCA.train()
    #     pred, acc = mthd_PCA.test()
    #     pred_list.append(pred)
    #
    # print(pred_list)
    # print(pred_list.shape)

    # PCA_nn_ORL(cv)
    # pred_list = []
    # kf, all_pic_list, all_label_list = r.read_rt_cv(isflatten=True, test_rate=0.1)
    # for train_id, test_id in kf.split(all_pic_list):
    #     train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
    #     test_set, test_label = all_pic_list[test_id], all_label_list[test_id]
    #
    #     train_pca_nn(train_set, train_label, test_set, test_label)



    # PCA_cnn_ORL(cv)
    print("run PCA_cnn_ORL...")
    pred_list = []
    kf, all_pic_list, all_label_list = r.read_rt_cv(isflatten=True, test_rate=0.1)
    for train_id, test_id in kf.split(all_pic_list):
        train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
        test_set, test_label = all_pic_list[test_id], all_label_list[test_id]

        train_pca_nn(train_set, train_label, test_set, test_label)

