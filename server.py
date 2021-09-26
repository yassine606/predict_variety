import pickle

import numpy as np

from flask import render_template, redirect, url_for,request
from flask import Flask
app = Flask(__name__)

knn=pickle.load(open('/Users/admin/PycharmProjects/predict_variety/model.pkl','rb'))

@app.route("/", methods=['GET','POST'])
def home():

    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = float(request.form['sepal.length'])
        glucose = float(request.form['sepal.width'])
        bp = float(request.form['petal.length'])
        st = float(request.form['petal.width'])



        data = np.array([[preg, glucose, bp, st]])



        return render_template('output.html', prediction=knn.predict(data))


if __name__== "__main__":
    app.run(host='localhost', port=5000)
