# -*- coding:utf-8 -*-  
import numpy as np
from sklearn.manifold import TSNE
import codecs

__author__ = "JOHNKYON"


def plot_build(mtr):
    """
    将高维数据转化为二维点
    :param mtr: np.narray 高维距离矩阵
    :return: np.narray　二维点表示
    """
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    points = model.fit_transform(mtr)

    output_file = codecs.open('temp/points.txt', 'wb', 'utf8')
    for ele in points:
        for x in ele:
            output_file.write(str(x))
            output_file.write('\t')
        output_file.write('\n')

    output_file.close()

    return points
