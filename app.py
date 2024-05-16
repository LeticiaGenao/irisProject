from flask import Flask, render_template, request
import numpy as np
import pickle
from waitress import serve
import os  # Import os to access environment variables

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Sepal_Length = float(request.form['sepal_length'])
        Sepal_Width = float(request.form['sepal_width'])
        Petal_Length = float(request.form['petal_length'])
        Petal_Width = float(request.form['petal_width'])
        prediction = model.predict(np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]]))
        species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}[prediction[0]]  # Update according to your label encoding
        return render_template('index.html', prediction=f'Species predicted: {species}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    # Use Waitress to serve the app and use the PORT environment variable from Heroku
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if no PORT variable is set
    serve(app, host="0.0.0.0", port=port)
