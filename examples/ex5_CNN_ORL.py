from read_utils.Read_pic import Read
from method.CNN_ORL import cnn_train_and_test
import time
import numpy as np

test_rate = 0.1
dir_path = "../data_faces"

if __name__ == "__main__":
    # read picture
    r = Read(dir_path)
    train_set, train_label, test_set, test_label = r.read_rt_tt(isflatten=False, test_rate=0.1)

    # PCA_cnn_ORL(cv)
    s = time.time()
    print("run CNN_ORL...")
    pred_list = []
    acc_list = []
    kf, all_pic_list, all_label_list = r.read_rt_cv(isflatten=False, test_rate=0.1)
    for train_id, test_id in kf.split(all_pic_list):
        train_set, train_label = all_pic_list[train_id], all_label_list[train_id]
        test_set, test_label = all_pic_list[test_id], all_label_list[test_id]

        pred, acc = cnn_train_and_test(train_set, train_label, test_set, test_label)
        pred_list.append(pred)
        acc_list.append(acc)
    pred_list = np.asarray(pred_list)  # predict result of all list
    # result = get_max_id(pred_list)
    # print("result:", result)
    print("accuracy list: ", acc_list)
    print("average accuracy: ", np.mean(np.asarray(acc_list)))

    e = time.time()
    time = e - s
    print("Total time of ex1_CNN_ORL: ", time)
