# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/12/09 21:56:28
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pandas as pd
import numpy as np
import time

def submit(res):
    '''
    按照要求的格式输出结果表格
    输入: 模型预测结果. 注意数据类型为浮点型, 表示逾期的概率.
    输出: 将提交文件存在/submission/文件夹下
    '''
    subimt_csv = pd.read_csv(r'./data/submit.csv')
    subimt_csv['label'] = res
    time_path = time.strftime("%Y%m%d%H%M", time.localtime()) + '.csv'
    subimt_csv.to_csv('./submission/' + time_path, index=False)
    

def evaluation(y_true, y_predict):
    '''
    进行线下评测.
    评测方法: 分别先计算出三个情况下的tpr(tpr:fpr=0.001/0.005/0.01). 然后计算
        总得分为: score=0.4*TPR1 + 0.3*TPR2 + 0.3*TPR3
    输入: 一维的预测数据和真实数据
    输出: 线下评测得分
    '''
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3



