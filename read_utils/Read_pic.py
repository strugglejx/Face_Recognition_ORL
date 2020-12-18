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

    def __read_dir(self, isflatten):
        all_pic_list = []
        all_label_list = []

        for sub_path in os.listdir(self.path):
            for i, file in enumerate(os.listdir(os.path.join(self.path, sub_path))):
                file_path = os.path.join(self.path, sub_path, file)
                image = Image.open(file_path)

                if isflatten is True:
                    all_pic_list.append(np.asarray(image).flatten())
                else:
                    all_pic_list.append(np.expand_dims(np.asarray(image), axis=0))
                all_label_list.append(int(sub_path[1::]))

        all_pic_list = np.asarray(all_pic_list, dtype=np.float32)
        all_label_list = np.asarray(all_label_list, dtype=np.int64)-1 # from zero
        return all_pic_list, all_label_list

    def read_rt_tt(self, isflatten, test_rate=0.1):
        '''
        :param isflatten: True means 1d; False means 2d.
        :param test_rate: the rate of test set
        :return:
        '''
        self.test_rate = test_rate
        all_pic_list, all_label_list = self.__read_dir(isflatten)
        pic_train_data, pic_test_data, pic_train_label, pic_test_label = train_test_split(
            all_pic_list, all_label_list, test_size=self.test_rate)
        return pic_train_data, pic_train_label, pic_test_data, pic_test_label


    def read_rt_cv(self, isflatten, test_rate=0.1):
        self.test_rate = test_rate
        all_pic_list, all_label_list = self.__read_dir(isflatten)
        kf = KFold(n_splits=int(1/self.test_rate), shuffle=True)
        return kf, all_pic_list, all_label_list



