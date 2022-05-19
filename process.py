import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
#选择三类转录因子数据
kinds = [i for i in os.listdir('35个转录因子序列集') if i.split('.')[0] in
         ['bHLH_TF119','TCP_TF237','MYB_TF224']
         ]
data = []
label = []
c = 0
for file in kinds:
    if file[-3:] == 'txt':
        with open('35个转录因子序列集/' + file, 'r') as f:
            lines = f.readlines()
            lines1 = [i for i in lines if i != '\n']
    # seq = [i.strip() for i in list(set(lines1))[:2622]]#下采样平衡数据
    seq = [i.strip() for i in random.sample(list(set(lines1)),2622)]
    data.append(seq)
    label += [c] * len(seq)#这个是什么
    c += 1


#截长补短，统一序列长度为800
# def pad_seq(seq, maxlen=800):
#     seq = seq[:maxlen]
#     return seq + 'N' * (maxlen - len(seq))
def pad_seq(seq, maxlen=800):
    if len(seq)<800 or len(seq)==800:
        l=seq+'N' * (maxlen - len(seq))
    else:
        a=int((len(seq)-800)/2-1)
        l = seq[a:(a+800)]
    return l
#one hot编码
onehotdict = {
    'A':[1,0,0,0],
    'C':[0,1,0,0],
    'G':[0,0,1,0],
    'T':[0,0,0,1],
    'N':[0,0,0,0]
}
# def one_hot(seq, n_class=5):
def one_hot(seq):
    onehotdata = []
    for i in seq:
        onehotdata.append(onehotdict[i])
    return np.array(onehotdata)
data_onehot = []
#调用上面函数的主函数
for i in data:
    for j in range(len(i)):
        i[j] = pad_seq(i[j])
        data_onehot.append(one_hot(i[j]))
data_onehot = np.array(data_onehot)
label = np.array(label)

#打乱顺序,生成一个储存下标的数组，将其打乱，然后按照这个下标将label和data_onehot的内容都打乱
index = [i for i in range(len(data_onehot))]
np.random.shuffle(index)
data_onehot = data_onehot[index]
label = label[index]
# print(data_onehot)
# print(label)
# 分割训练集和测试集
train_data = data_onehot[:int(len(data_onehot)*0.8)]
train_label = label[:int(len(label)*0.8)]
test_data = data_onehot[int(len(data_onehot)*0.8):]
test_label = label[int(len(label)*0.8):]
#保存数据
np.save('train_data', train_data)
np.save('train_label', train_label)
np.save('test_data', test_data)
np.save('test_label', test_label)
print('Done!')
testx = np.load('test_data.npy')
print(testx.shape)
