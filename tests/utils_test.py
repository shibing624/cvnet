# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""


def demo_jieba():
    a = '我要办理特价机票，李浩然可以想办法'
    import jieba
    b = jieba.lcut(a, cut_all=False)
    print('cut_all=False', b)
    b = jieba.lcut(a, cut_all=True)
    print('cut_all=True', b)

    b = jieba.lcut(a, HMM=True)
    print('HMM=True', b)

    b = jieba.lcut(a, HMM=False)
    print('HMM=False', b)


if __name__ == '__main__':
    demo_jieba()
