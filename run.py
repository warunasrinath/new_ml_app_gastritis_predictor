#Then import important libraries
import numpy as np
from flask import Flask, render_template, request
import pickle


#Again Load the Random Forest CLassifier model
filename = 'gastritis_dataset.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preTSG = int(request.form['TSG'])
        preTSS = int(request.form['TSS'])
        preWCP = int(request.form['WCP'])
        prePFS = int(request.form['PFS']) 
        preGID = int(request.form['GID'])
        preNWHB = int(request.form['NWHB'])
        preSDM = int(request.form['SDM'])
        preEUF = int(request.form['EUF'])
        preISU = int(request.form['ISU'])

        
        data = np.array([[preTSG, preTSS, preWCP, prePFS, preGID, preNWHB, preSDM, preEUF, preISU]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)