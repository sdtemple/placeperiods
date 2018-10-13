from warnings import filterwarnings
filterwarnings(action='ignore', category=FutureWarning, module='._conv')
filterwarnings(action='ignore', module='sklearn.metrics')

def write_periods(text, grams, model, n):
    ''' (str, numpy.array, keras.models.Sequential, int) -> str
    Place periods in the text based on a trained keras model.
    Return the text with periods at their predicted positions.
    '''
    temp_text = text.split()
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