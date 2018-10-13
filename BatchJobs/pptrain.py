'''
Seth Temple
CIS 401 Research
Read Understand Learn Excel (RULE)
Place the Period
'''
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import load_model
from KerasModelingTools import *

print('Imports finished')

model_path = '/projects/fickaslab/stemple/models/'
t = upload_texts('/projects/fickaslab/stemple/data/FriendEssays/')
print('texts uploaded')
d = convert_texts(t, 3, False)
print('texts converted')

xy = []
for i in range(len(d)):
    x, y = reshape_keras_data(d[i][2]), d[i][1]
    if len(x) == len(y):
        xy.append((x,y))
        
import os
if 'period_placer.h5' in os.listdir(model_path):
    m = load_model(model_path + 'period_placer.h5')
    print('Loaded model period_placer.h5')
else:
    m = Sequential()
    m.add(Conv3D(16, (1,1,300), strides=(1,1,300), activation='relu', input_shape=(2,3,300,1)))
    m.add(Flatten())
    m.add(Dense(12, activation='relu'))
    # m.add(Dropout(.5))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Made new model')

for i in range(len(xy)):
    if ((i+1) % 10) == 0:
        print('Processed the ' + str((i+1)) + 'th' + ' essay in the corpus')
    m.fit(xy[i][0], xy[i][1], epochs=10, batch_size=30, verbose=0)
    
m.save(model_path + 'period_placer.h5')
print('Saved model as period_placer.h5')