from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

app = Flask("__name__")

q = ""


# @app.route("/")
# def loadPage():
#    return render_template('home.html', query='')


@app.route("/predict", methods=['POST'])
def cancerPrediction():
    # check jupyter notebook, the below code is copied from there

    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)

    model = pickle.load(open("model.sav", 'rb'))

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']

    # now instead of hard-coding, we are going to use above values

    data = [[ inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5 ]]
    new_df = pd.DataFrame(data, columns = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean'])

    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]

    if single==1:
        o1 = "The patient is diagnosed with breast Cancer"
        o2 = "Confidence: {}".format(probability*100)
    if single==0:
        o1 = "The patient is not diagnosed with breast Cancer"
        o2 = "Confidence: {}".format(probability*100)

    return render_template('home.html', output1=o1, output2=o2, query1= request.form['query1'], query2= request.form['query2'], query3= request.form['query3'], query4= request.form['query4'], query5= request.form['query5'])


app.run()
