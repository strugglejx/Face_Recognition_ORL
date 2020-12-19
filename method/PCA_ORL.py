from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import time

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
        self.pca_vector = None
        self.trainset_project = None

    def __cov(self, a):
        mean = np.mean(a, axis=1)
        mid = a.T - mean
        res = np.matmul(mid.T, mid) / a.shape[0]
        return res

    def __project(self, set, mean):
        y = np.matmul(self.pca_vector.transpose(), set.transpose()-mean)
        return y

    def pca(self, XMat, k):
        average = np.mean(XMat, axis=0)
        m, n = np.shape(XMat)  # m: samples_nums, n: features
        data_adjust = XMat - average
        cov_x = self.__cov(data_adjust.T)  # 计算协方差矩阵
        feat_value, feat_vec = np.linalg.eig(cov_x)  # 求解协方差矩阵的特征值和特征向量
        index = np.argsort(-feat_value)  # 依照featValue进行从大到小排序
        variance_ratio = np.sum(feat_value[index[:k]]) / np.sum(feat_value)
        print("variance_ratio:", variance_ratio)
        if k > n:
            print("k must lower than feature number")
            return
        else:
            selectVec = feat_vec[:, index[:k]]  # 所以这里须要进行转置
            finalData = np.matmul(data_adjust, selectVec)
        return finalData, selectVec

    def pca_acc(self, data, r):  # matmul --> *
        data = np.float32(np.mat(data))
        rows, cols = np.shape(data)
        data_mean = np.mean(data, 0)  # 对列求平均值
        A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
        C = A * A.T  # 得到协方差矩阵  （样本数，样本数）
        D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
        sort = np.argsort(-D)  # 排序之后返回原对应下标
        V_r = V[:, sort[0:r]]  # 取前r个特征值最大的特征向量  (主成分变换矩阵)
        variance_ratio = np.sum(D[sort[0:r]]) / np.sum(D)
        print("variance_ratio:", variance_ratio)
        V_r = A.T * V_r  # 小矩阵特征向量向大矩阵特征向量过渡
        for i in range(r):
            V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # 特征向量归一化  (二范数)
        final_data = A * V_r  # 特征脸
        return final_data, V_r  # 主成分变换矩阵

    def train(self, isacc = False):
        if isacc is False:
            a, self.pca_vector = self.pca(self.trainset, self.n_components)
            train_mean = np.mean(self.trainset, axis=1)
            self.trainset_project = self.__project(self.trainset, train_mean)
        else:
            a, self.pca_vector = self.pca_acc(self.trainset, self.n_components)
            train_mean = np.mean(self.trainset, axis=1)
            self.trainset_project = self.__project(self.trainset, train_mean)


    def test(self):
        s = time.time()
        test_mean = np.mean(self.testset, axis=1)
        y = self.__project(self.testset, test_mean)
        pred = []
        for i in range(y.shape[1]):
            tmp = y[:, i].reshape((y.shape[0], 1))
            tmp = tmp - self.trainset_project
            tmp = np.sum(np.square(tmp), axis=0).flatten()
            min_i = np.argmin(tmp, axis=0)
            pred.append(self.trainlabel[min_i])

        pred = np.asarray(pred)
        # print(pred)
        acc = np.sum((pred == self.testslabel))/len(pred)
        print("accuracy: ", acc)
        e = time.time()
        tim = e - s
        print("Total time of PAC_ORL test: ", tim)
        return pred, acc

if __name__ == "__main__":
    # test

    train_set = np.asarray([[1,2,3,4,5], [2,3,4,5,6], [6,7,8,9,10], [6,5,4,3,2], [8,7,6,5,4]])
    train_label = np.asarray([0,0,0,1,1])
    test_set = np.asarray([[4,5,6,7,8], [5,6,7,8,9], [7,6,5,4,3], [1,3,5,7,9]])
    test_label = np.asarray([0,0,1,0])

    # PCA
    mthd_PCA = myPCA(trainset=train_set,
                     testset=test_set,
                     trainlabel=train_label,
                     testlabel=test_label,
                     n_components=2,
                     isKernel=False)
    mthd_PCA.train(True)
    pred, acc = mthd_PCA.test()
    print(pred.shape)

