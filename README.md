# Overdue Prediction

:star: task: 利用个人的基本身份信息, 个人的住房公积金缴存和贷款等数据信息, 来预测用户是否会预期还款.

提交结果: id, 预期概率

## 数据集
训练集中含有40000个人, 测试集中含有15000个人

所有数据据类型都为字符串类型, 需要对数值型的进行转换

## 评分方法
先计算出三个TPR. 分别是三种情况下的TPR: TPR:FPR=0.001/0.005/0.01
$$
score=0.4 \times TPR_1+0.3 \times TPR_2+0.3 \times TPR_3
$$

```
def tpr_weight_funtion(y_true,y_predict):
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
```

## 尝试方法
1.LR
2.GBDT



