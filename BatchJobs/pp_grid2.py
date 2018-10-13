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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

s=100317
np.random.seed(s)

model_path = '/projects/fickaslab/stemple/models/'
t = upload_texts('/projects/fickaslab/stemple/data/StudentSummaries/')
d = convert_texts(t, 3, False)

y = np.array([], dtype=np.float32)
x = np.array([], dtype=np.float32).reshape((0,2,3,300,1))
for i in range(len(d)):
    if len(d[i][2]) == len(d[i][1]):
        x = np.concatenate((x, d[i][2].reshape(d[i][2].shape+(1,))))
        y = np.concatenate((y, d[i][1]))
print(x.shape)
print(y.shape)
        
def make_model(nodes, drate, kernel_size):
    m = Sequential()
    m.add(Conv3D(16, kernel_size, strides=kernel_size, activation='relu', input_shape=(2,3,300,1)))
    m.add(Flatten())
    m.add(Dense(nodes, activation='relu'))
    m.add(Dropout(drate))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

m = KerasClassifier(build_fn=make_model, verbose=0)

d = [0.3, 0.4, 0.5, 0.6, 0.7]
n = [4, 6, 8, 10, 12, 16, 24]
s = [(1,1,300), (1,3,300)]

pgrid = dict(nodes=n, drate=d, kernel_size=s)
grid = GridSearchCV(estimator=m, param_grid=pgrid)
search = grid.fit(x, y, epochs=3, batch_size=50)

print("GRID TESTING NETWORK TOPOLOGY")

print("Best: %f using %s" % (search.best_score_, search.best_params_))
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
params = search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))