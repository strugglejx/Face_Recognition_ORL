import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Read(object):
    '''
        read the image, then transfer to numpy
    '''
    def __init__(self, path):
        self.path = path
        self.test_rate = 0.1

    def __read_dir(self):
        all_pic_list = []
        all_label_list = []

        for sub_path in os.listdir(self.path):
            for i, file in enumerate(os.listdir(os.path.join(self.path, sub_path))):
                file_path = os.path.join(self.path, sub_path, file)
                image = Image.open(file_path)

                all_pic_list.append(np.asarray(image).flatten())
                all_label_list.append(int(sub_path[1::]))

        all_pic_list = np.asarray(all_pic_list)
        all_label_list = np.asarray(all_label_list)
        return all_pic_list, all_label_list

    def read_rt_tt(self, test_rate=0.1):
        self.test_rate = test_rate
        all_pic_list, all_label_list = self.__read_dir()
        pic_train_data, pic_test_data, pic_train_label, pic_test_label = train_test_split(
            all_pic_list, all_label_list, test_size=self.test_rate)
        return pic_train_data, pic_train_label, pic_test_data, pic_test_label


    def read_rt_cv(self, test_rate):
        self.test_rate = test_rate
        all_pic_list, all_label_list = self.__read_dir()
        kf = KFold(n_splits=int(1/self.test_rate), shuffle=True)
        return kf, all_pic_list, all_label_list



