from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns

def show_train_history(train_history):
    plt.plot(list(range(1, len(train_history['accuracy']) + 1)), train_history['accuracy'], color='red')
    plt.plot(list(range(1, len(train_history['val_accuracy']) + 1)), train_history['val_accuracy'], color='blue')
    plt.title('train history')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig('acc.png')
    plt.close('all')

    plt.plot(list(range(1, len(train_history['loss']) + 1)), train_history['loss'], color='red')
    plt.plot(list(range(1, len(train_history['val_loss']) + 1)), train_history['val_loss'], color='blue')
    plt.title('train history')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig('loss.png')
    plt.close('all')


def show_eval(model, testx, testy):
    prey = model.predict(testx, verbose=1)
    prey = prey.argmax(1)
    testy = testy.argmax(1)
    result = metrics.confusion_matrix(testy, prey, labels=list(range(3)))
    print(result)
    print(classification_report(prey, testy, digits=4))
    f, ax = plt.subplots()
    sns.heatmap(result, annot=True,ax=ax,fmt='.20g')
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('confusion.png')
    plt.close('all')
    print(testx.shape)