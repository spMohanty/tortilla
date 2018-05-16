from flask import Flask, render_template, request
import sys
import pickle
import os
import json
sys.path.insert(0,'tortilla')
import tortilla_predict

models = os.listdir('/home/harsh/experiments/')

app = Flask(__name__)
UPLOAD_FOLDER = '/home/harsh/data/uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER




#Index page

@app.route('/')


def index():
	
	return render_template("upload.html")

#Upload methods
@app.route('/upload',methods=['POST'])

def upload_file():
	file = request.files['image']
	f = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
	file.save(f)
	return render_template('upload.html',models=models)

@app.route('/upload/eval',methods=['GET','POST'])

def evaluate_results():
	current_model = request.args.get("model")
	if current_model == None:
		current_model = 'ROOT'
	tortilla_predict.predict('/home/harsh/experiments/'+current_model+'/trained_model.net','/home/harsh/data/uploads')
	predictions = json.load(open('/home/harsh/experiments/'+current_model+'/predictions/prediction.json'))
	return predictions
	



if __name__ == '__main__':
	app.run(host ='0.0.0.0',port=5000,debug=True)

		
