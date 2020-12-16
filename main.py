from read_utils.Read_pic import Read
from method.PCA_ORL import myPCA
import os

test_rate = 0.1
dir_path = os.path.join(os.getcwd(), "data_faces")

if __name__ == "__main__":
    # read picture
    r = Read(dir_path)
    train_set, train_label, test_set, test_label = r.read_rt_tt(0.1)

    print(train_set.shape)
    print(train_label.shape)
    print(test_set.shape)
    print(test_label.shape)

    # PCA
    # mthd_PCA = myPCA(trainset=train_set,
    #                  testset=test_set,
    #                  trainlabel=train_label,
    #                  testlabel=test_label,
    #                  n_components=200,
    #                  isKernel=False)
    # mthd_PCA.train()
    # mthd_PCA.test()

    pred_list = []
    kf, all_pic_list, all_label_list = r.read_rt_cv(test_rate)
    for train_id, test_id in kf.split(all_pic_list):
        train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
        test_set, test_label = all_pic_list[test_id], all_label_list[test_id]


        mthd_PCA = myPCA(trainset=train_set,
                         testset=test_set,
                         trainlabel=train_label,
                         testlabel=test_label,
                         n_components=20,
                         isKernel=False)
        mthd_PCA.train()
        pred, acc = mthd_PCA.test()
        pred_list.append(pred)

    print(pred)

    '''
        test1:
        (360, 10304)
        (360,)
        (40, 10304)
        (40,)
        accuracy:  0.925
    '''

    # cuda: 10.2