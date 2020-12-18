import numpy as np

def get_max_id(a):
    '''
        get max count(columns)
    :param a: numpy array
    :return: result of max count
    '''
    res = []
    for i in range(a.shape[1]):
        res.append(np.argmax(np.bincount(a[:, i])))
    return np.asarray(res)



if __name__ == "__main__":

    # test example
    a = np.asarray([[1, 2, 4, 3], [1, 2, 4, 3], [2, 3, 4, 2], [1, 3, 2, 4]])
    # pred result: [1, 2, 4, 3]
    res = get_max_id(a)
    print(res)
