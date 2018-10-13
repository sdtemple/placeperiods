from flask import Flask, render_template, request
from KerasModelingTools import reshape_keras_data, write_periods
from MakeNGramSplitWindows import *
import keras
from keras.models import load_model
from os.path import expanduser
from os import walk

for root, dirs, paths in walk(expanduser('~')):
    if 'period_placer.h5' in paths:
        model_path = root + '\\' + paths[paths.index('period_placer.h5')]
m = load_model(model_path)

app = Flask(__name__)

@app.route('/')
def app_home():
    output = ''
    input = ''
    return render_template('solve.html', output = output, input = input)

@app.route('/', methods=['GET','POST'])
def app_solve():
    output = ''
    input = ''
    if request.method == 'POST':
        input = request.form['input']
        if len(input) > 0:
            text = input
            x = window_processing_periods(text, 3, False)
            x = reshape_keras_data(x)
            output = write_periods(text, x, m, 3)
    return render_template('solve.html', output = output, input = input)

@app.route('/about')
def app_about():
    return render_template('about.html')

@app.route('/details')
def app_details():
    return render_template('details.html')


if __name__ == '__main__':
    app.run()
