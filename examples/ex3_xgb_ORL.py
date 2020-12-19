from read_utils.Read_pic import Read
from method.PCA_xgb_ORL import train_pca_xgb
import time
import numpy as np

test_rate = 0.1
dir_path = "../data_faces"

if __name__ == "__main__":
    # read picture
    r = Read(dir_path)
    train_set, train_label, test_set, test_label = r.read_rt_tt(True, 0.1)

    train_pca_xgb(train_set, train_label, test_set, test_label)
