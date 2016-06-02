# -*- coding:utf-8 -*-  

import numpy as np
import codecs

__author__ = "JOHNKYON"


def judgement(x, y):
    print x
    print y
    T = float(0)
    F = float(0)
    counter = float(0)
    outputfile = codecs.open('temp/precision.txt', 'wb', encoding='utf8')

    for ele in x:
        outputfile.write(str(ele)+'\t')
        if ele == y[counter]:
            T += 1
        else:
            F += 1
        counter += 1

    outputfile.write('\n')
    for ele in y:
        outputfile.write(str(ele)+'\t')

    precision = T/(T+F)
    return precision