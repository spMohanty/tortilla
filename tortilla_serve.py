from flask import Flask, render_template, request, send_from_directory
import sys
import pickle
import os
import json
sys.path.insert(0,'tortilla')
import tortilla_predict
import re
import sys


models = os.listdir(str(sys.argv[1]))

UPLOAD_FOLDER = str(sys.argv[2])

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER




#Index page

@app.route('/')

#delete all uploaded files when you load the index page
def index():
	path = app.config['UPLOAD_FOLDER']
	for i in os.listdir(app.config['UPLOAD_FOLDER']):
		file_path = os.path.join(path,i)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)
	return render_template("upload.html")

#Upload methods
@app.route('/upload',methods =['POST','GET'])
def upload_file():#upload files one-by-one in the upload folder
	if 'image' in request.files:
		file = request.files['image']
		f = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
		file.save(f)
		return render_template('upload.html',models=models,output=None)
	else:
		if len(os.listdir(app.config['UPLOAD_FOLDER']))!=0:
			return render_template('upload.html',models=models,output=None)	
		else:
			return render_template('upload.html')			

@app.route('/upload/eval')
#Method to evaluate result according to the model
def evaluate_results():
	current_model = request.args.get("model")
	if current_model == None:
		current_model = 'ROOT'
	if not re.match('sf',current_model):
		model_path = '/home/harsh/experiments/'+current_model+'/trained_model.net'
	else:
		model_path = '/home/harsh/experiments/'+current_model+'/checkpoints/snapshot_latest.net'
	try:
		tortilla_predict.predict(model_path,'/home/harsh/data/uploads')
	except MemoryError as error:
		return render_template("output.html",predictions = [],error = error)	
	predictions = json.load(open('/home/harsh/experiments/'+current_model+'/predictions/prediction.json'))

	p = re.compile(r'uploads/')
	predictions=dict((i[p.search(i).end():],predictions[i]) for i in predictions.keys())
		
	return render_template("output.html",predictions=predictions,error = "")
	
#method to send uploaded image for viewing 
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
	app.run(host ='0.0.0.0',port=5001,debug=True)

		
