# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


__author__ = "JOHNKYON"


def imamge_2D(points, label, id):
    """
    用于生成二维图像
    :param points:
    :return:
    """

    plt.scatter(points[:, 0], points[:, 1], c=label)
    plt.savefig('images/104_parameter_test/%s.png' % str(id))
    plt.cla()
