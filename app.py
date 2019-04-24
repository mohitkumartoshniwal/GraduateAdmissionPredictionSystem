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
    df = pd.read_csv("data/Admission_P.csv")
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
		    linear_model = joblib.load('data/linear_model.py')
		    result_prediction = linear_model.predict(ex1)
		elif model_choice == 'linear1_model':
			linear1_model = joblib.load('data/linear1_model.py')
			result_prediction = linear1_model.predict(ex1)
		elif model_choice == 'linear2_model':
			linear2_model == joblib.load('data/linear2_model.py')
			result_prediction = linear2_model.predict(ex1)

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