from flask  import Flask,render_template,url_for,request
import joblib
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


import requests
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)


clf=pickle.load(open("spamdetect.pkl","rb"))
cv=pickle.load(open("trasform.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():

    


    if request.method=="POST":
        message=request.form["message"]
        data=[message]
        vect=cv.transform(data).toarray()
        my_pred=clf.predict(vect)
        return render_template('result.html',prediction = my_pred)


if __name__ == '__main__':
	app.run(debug=True)