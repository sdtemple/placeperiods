'''
Seth Temple
CIS 401 Research
Rule Understand Learn Excel
Place the Period: Keras Modeling Tools
Packaged together functions we use to build keras models for the RULE project.
'''
import os
import codecs
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='._conv')
warnings.filterwarnings(action='ignore', module='sklearn.metrics')

# sklearn summary functions
from sklearn.metrics import confusion_matrix
from SklearnConfusionMatrix import *
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def report_metrics(y_true, y_pred):
    return {'accuracy':accuracy_score(y_true, y_pred),
            'precision':precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)}

def summarize_periods_model(y_true, y_pred):
    class_names = ['NoPeriod', 'Period']
    matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(matrix, class_names)
    return report_metrics(y_true, y_pred)

def plot_keras_feedback(history):
    plot_keras_accuracy(history)
    plot_keras_loss(history)

def plot_keras_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend([ 'Train' , 'Test' ], loc= 'upper left' )
    plt.show()

def plot_keras_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend([ 'Train' , 'Test' ], loc= 'upper left' )
    plt.show()


# Predict and write periods. Judge model performance by human interpretation.
from keras.models import Sequential

def write_periods(text, grams, model, n):
    ''' (str, numpy.array, keras.models.Sequential, int) -> str
    Place periods in the text based on a trained keras model.
    Return the text with periods at their predicted positions.
    '''
    temp_text = text.replace('.','')
    temp_text=temp_text.replace('?','')
    temp_text=temp_text.replace('!','')
    temp_text = temp_text.split()

    res = model.predict_classes(grams)
    for j in range(len(res)):
        pred = int(res[j][0])
        if pred == 1:
            temp_text[j+n-1] = temp_text[j+n-1] + '.'

    final_text = ''
    for word in temp_text:
        final_text = final_text + ' ' + word

    return final_text + '.'


# Reshape data to conform with keras input_shape
def reshape_keras_data(x):
    return x.reshape(x.shape + (1,))


# Upload text data from a folder.
def upload_texts(folder):
    texts = []
    for file in os.listdir(folder):
        f = codecs.open(folder + str(file), 'r', 'utf-8', errors='ignore')
        texts.append(f.read())
        f.close()
    return texts

from MakeNGramSplitWindows import window_processing_periods

def convert_texts(texts, n, rand):
    data = []
    for text in texts:
        data.append(window_processing_periods(text, n, rand))
    return data

