from flask import Flask, render_template, request
import pandas as pd

app=Flask(__name__)

model=pd.read_pickle('admit_ucla_64bit.pickle')

@app.route('/')
def home():
	# df=pd.read_csv('Admission_Predict.csv')

	# lis=[]
	# X=df.drop(['Chance of Admit ','Serial No.'],axis=1)
	# a=X.columns
	# for i in a:
	# 	lis.append(i) 
	
	return render_template('home.html')



@app.route('/predict', methods=['POST','GET'])
def predict():

	# df=pd.read_csv('Admission_Predict.csv')
	# lis=[]
	# X=df.drop(['Chance of Admit ','Serial No.'],axis=1)
	# a=X.columns
	

	if request.method=='POST':
		GRE_Score=float(request.form['GRE Score'])
		TOEFL_Score= float(request.form['TOEFL Score'])
		University_Rating=float(request.form['University Rating'])
		SOP=float(request.form['SOP'])
		LOR= float(request.form['LOR'])
		CGPA=float(request.form['CGPA'])
		Research= float(request.form['Research'])

	
	chances=model.predict([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]])
	return render_template('result.html', prediction_text=chances[0]*100)


if __name__ == '__main__':
	app.run(debug=True)