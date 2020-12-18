from read_utils.Read_pic import Read
from method.PCA_ORL import myPCA
import numpy as np
import time

test_rate = 0.1
dir_path = "../data_faces"

if __name__ == "__main__":
    # read picture
    r = Read(dir_path)
    train_set, train_label, test_set, test_label = r.read_rt_tt(isflatten=False, test_rate=0.1)

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
    s = time.time()
    pred_list = []
    acc_list = []
    kf, all_pic_list, all_label_list = r.read_rt_cv(isflatten=True, test_rate=0.1)
    for train_id, test_id in kf.split(all_pic_list):
        train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
        test_set, test_label = all_pic_list[test_id], all_label_list[test_id]


        mthd_PCA = myPCA(trainset=train_set,
                         testset=test_set,
                         trainlabel=train_label,
                         testlabel=test_label,
                         n_components=50,
                         isKernel=False)

        mthd_PCA.train(False)
        pred, acc = mthd_PCA.test()
        pred_list.append(pred)
        acc_list.append(acc)
    pred_list = np.asarray(pred_list)  # predict result of all list
    # result = get_max_id(pred_list)
    # print("result:", result)
    print("average accuray: ", np.mean(np.asarray(acc_list)))

    e = time.time()
    time = e - s
    print("Total time of ex1_PAC_ORL: ", time)
