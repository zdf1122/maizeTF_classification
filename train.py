
import numpy as np
from tensorflow import keras
from model import create_model
from utils import show_train_history,show_eval
np.set_printoptions(threshold=np.inf)
num_classes = 3
trainx = np.load('train_data.npy')
trainy = np.load('train_label.npy')
# print(trainy)
trainy = keras.utils.to_categorical(trainy, num_classes=num_classes)
testx = np.load('test_data.npy')
testy = np.load('test_label.npy')
testy = keras.utils.to_categorical(testy, num_classes=num_classes)
# print(testy)


model = create_model(input_shape=(800,4), n_classes=num_classes)
callback = keras.callbacks.ModelCheckpoint(filepath='best.h5',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           save_weights_only=False)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(trainx, trainy,
                    epochs=55,
                    batch_size=64,
                    callbacks=[callback],
                    validation_data=(testx, testy))
np.save('history', history.history)
model = keras.models.load_model('best.h5')
show_train_history(history.history)
show_eval(model, testx, testy)

