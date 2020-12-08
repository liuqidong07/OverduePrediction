# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2020/12/08 20:40:58
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from torch.utils.data import Dataset
import pickle

train_path = r'./data/train_T.pkl'
test_path = r'./data/test_T.pkl'

class overdueTrain(Dataset):
    def __init__(self) -> None:
        super().__init__()





