from sklearn.decomposition import PCA, KernelPCA
import cupy as np
import numpy as npy


class myPCA(object):

    def __init__(self, trainset, testset, trainlabel, testlabel, n_components, isKernel):
        '''
            initial the class of PCA
        :param train: the dataset of train, shape: (360, 112*92)
        :param test: the dataset of test, shape: (40, 112*92)
        '''
        self.trainset = trainset
        self.testset = testset
        self.trainlabel = trainlabel
        self.testslabel = testlabel
        self.n_components = n_components
        self.isKernel = isKernel
        self.pca = None
        self.pca_vector = None
        self.trainset_project = None

    def __cov(self, a):
        a = cp.asarray(a)
        mean = np.mean(a, axis=1)
        mid = a.T - mean
        res = np.matmul(mid.T, mid) / a.shape[0]
        return res

    def __pca(self, XMat, k):
        average = np.mean(XMat, axis=0)
        m, n = np.shape(XMat)  # m: samples_nums, n: features
        data_adjust = XMat - average
        cov_x = self.__cov(data_adjust.T)  # 计算协方差矩阵
        feat_value, feat_vec = npy.linalg.eig(cov_x.get())  # 求解协方差矩阵的特征值和特征向量
        index = np.argsort(-feat_value)  # 依照featValue进行从大到小排序
        if k > n:
            print("k must lower than feature number")
            return
        else:
            selectVec = feat_vec[:, index[:k]]  # 所以这里须要进行转置
            finalData = np.matmul(data_adjust, np.asarray(selectVec))  # cupy
        return finalData, np.asarray(selectVec)

    def __project(self, set, mean):
        y = np.matmul(self.pca_vector.transpose(), set.transpose() - mean)
        return y

    def train(self):
        _, self.pca_vector = self.__pca(self.trainset, self.n_components)
        train_mean = np.mean(self.trainset, axis=1)
        self.trainset_project = self.__project(self.trainset, train_mean)

    def test(self):
        test_mean = np.mean(self.testset, axis=1)
        y = self.__project(self.testset, test_mean)

        pred = []

        for i in range(y.shape[1]):
            tmp = np.sum(np.square((np.expand_dims(y[:, i], axis=1) - self.trainset_project)), axis=0)
            min_i = np.argmin(tmp, axis=0)
            pred.append(self.trainlabel[min_i])

        pred = np.asarray(pred)
        # print(pred)
        acc = np.sum((pred == self.testslabel)) / len(pred)
        print("accuracy: ", acc)
        return pred, acc


if __name__ == "__main__":
    # test

    train_set = np.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [6, 7, 8, 9, 10], [6, 5, 4, 3, 2], [8, 7, 6, 5, 4]])
    train_label = np.asarray([0, 0, 0, 1, 1])
    test_set = np.asarray([[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [7, 6, 5, 4, 3], [1, 3, 5, 7, 9]])
    test_label = np.asarray([0, 0, 1, 0])

    # PCA
    mthd_PCA = myPCA(trainset=train_set,
                     testset=test_set,
                     trainlabel=train_label,
                     testlabel=test_label,
                     n_components=2,
                     isKernel=False)
    mthd_PCA.train()
    pred, acc = mthd_PCA.test()

