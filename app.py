from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/Admission_Predict_Ver1.1.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		GRE_Score = request.form['GRE_Score']
		TOEFL_Score = request.form['TOEFL_Score']
		University_Rating = request.form['University_Rating']
		SOP = request.form['SOP']
		LOR = request.form['LOR']
		CGPA = request.form['CGPA']
		Research = request.form['Research']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'linear_model':
		    linear_model = joblib.load('data/linear_model.pkl')
		    result_prediction = linear_model.predict(ex1)
		elif model_choice == 'ridge_model':
			ridge_model = joblib.load('data/ridge_model.pkl')
			result_prediction = ridge_model.predict(ex1)
		elif model_choice == 'rf2':
			RandomForestRegressor = joblib.load('data/RandomForestRegressor.pkl')
			result_prediction = RandomForestRegressor.predict(ex1)
		elif model_choice == 'lr2':
			LinearRegression = joblib.load('data/LinearRegression.pkl')
			result_prediction = LinearRegression.predict(ex1)
		elif model_choice == 'bayesianridge_model':
			bayesianridge_model == joblib.load('data/bayesianridge_model.pkl')
			result_prediction = bayesianridge_model.predict(ex1)
		elif model_choice == 'randomforest_model':
			randomforest_model == joblib.load('data/randomforest_model.pkl')
			result_prediction = randomforest_model.predict(ex1)
		elif model_choice == 'logistic_classifier':
			logistic_classifier == joblib.load('data/logistic_classifier.pkl')
			result_prediction = logistic_classifier.predict(ex1)
			

	return render_template('index.html', GRE_Score=GRE_Score,
		TOEFL_Score=TOEFL_Score,
		University_Rating=University_Rating,
		SOP=SOP,
		LOR=LOR,
		CGPA=CGPA,
		Research=Research,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)