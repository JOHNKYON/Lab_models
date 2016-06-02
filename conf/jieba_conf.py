# -*- coding: utf-8 -*-
import jieba
import os


def init():
    """
    初始化jieba分词器设置
    :return:
    """
    # 设置自定义字典
    jieba.set_dictionary("data/jieba_dict.txt")
    # 设置工作目录
    jieba.tmp_dir = os.getcwd()