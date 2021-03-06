# 公积金贷款逾期预测

## 任务
利用基本身份信息, 个人的住房公积金缴存和贷款等数据信息, 建立风险控制模型来预测用户是否会逾期还款.
该任务的label为0-1变量, 表示用户是否会逾期还款. 因此可以把这个问题看作一个<font color=red>二分类问题</font>. (类似于一个CTR)

## 数据集
训练集中有40000个样本, 测试集中有15000个.

特征字段表示:
![](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/pkbigdata/master.other.img/c803fd8f-c2b8-4aaa-a39a-d92a91ab18a2.png)

:warning: 测试集中增加了干扰样本, 这些样本的结果不参与计算. 官方解答是干扰样本是根据一些规律来生成的.

数据脱敏问题: 官方的回答是, (1)所有的数值型特征都经过了脱敏 (2)CNSY时间戳已经经过了脱敏 (3)贷款利率没有经过脱敏 (4)所有数值型的连续变量都是用同样的方式进行了脱敏

## 评分标准
三个TPR的加权值
$$
score = TPR(FPR=0.001) + TPR(FPR=0.005) + TPR(FPR=0.01)
$$



