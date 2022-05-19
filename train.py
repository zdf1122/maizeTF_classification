
import numpy as np
from tensorflow import keras
from model import create_model
from utils import show_train_history,show_eval
np.set_printoptions(threshold=np.inf)
num_classes = 3
trainx = np.load('train_data.npy')# .npy文件是numpy专用的二进制文件，读取用np.load，保存用np.save
trainy = np.load('train_label.npy')
# print(trainy)
trainy = keras.utils.to_categorical(trainy, num_classes=num_classes)#将整数数组转换为指定类别数的onehot编码
testx = np.load('test_data.npy')
testy = np.load('test_label.npy')
testy = keras.utils.to_categorical(testy, num_classes=num_classes)
# print(testy)


model = create_model(input_shape=(800,4), n_classes=num_classes)
callback = keras.callbacks.ModelCheckpoint(filepath='best.h5',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           save_weights_only=False)
#save_best_only：当设置为True时，表示当模型这次epoch的训练评判结果（monitor的监测值）比上一次保存训练时的结果有提升时才进行保存。
#save_weights_only：若设置为True，占用内存小（只保存模型权重），但下次想调用的话，需要搭建和训练时一样的网络。若设置为False，占用内存大（包括了模型结构和配置信息），下次调用可以直接载入，不需要再次搭建神经网络结构。
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),#优化器种类和学习率可以修改一下
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()#model.summary()输出模型各层的参数状况
history = model.fit(trainx, trainy,
                    epochs=55,#40之后出现过拟合
                    batch_size=64,#
                    callbacks=[callback],
                    validation_data=(testx, testy))#评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。评价函数和损失函数相似，只不过评价函数的结果不会用于训练过程中。

np.save('history', history.history)
model = keras.models.load_model('best.h5')
show_train_history(history.history)
show_eval(model, testx, testy)

