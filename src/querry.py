# -*- coding: utf-8 -*-

import codecs
import json


def form_bucket(mtx):
    """
    此函数用于将将已生成的相似度矩阵查询为专业名称，方便查看效果
    :param mtx:
    :return:
    """
    output_file = codecs.open("data/bucket.txt", 'wb', encoding='utf8')
    input_file = codecs.open("data/major_dict.json", 'rb', encoding='utf8')
    raw = input_file.read()
    major_list = json.loads(raw)

    for ele in mtx:
        for x in ele:
            output_file.write(major_list[str(x[0])])
        output_file.write("\n\n")

    input_file.close()
    output_file.close()

