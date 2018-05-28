from flask import Flask, render_template, request
import numpy as np
import pickle
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.layouts import gridplot
import os
import json
from bokeh.models import (
	ColumnDataSource,
	HoverTool,
	LinearColorMapper,
	BasicTicker,
	PrintfTickFormatter,
	ColorBar,
)
import pandas as pd
from math import pi
import re	
import subprocess


app = Flask(__name__)



	

#model parameters
models = os.listdir('/home/harsh/experiments/')
colors = ['red','blue','green','yellow','pink','orange','aqua','purple','gray','black']
par = re.compile("::(?!.*::)")

def create_figure(model):
	meta = json.load(open('/home/harsh/tortilla/datasets/'+model+'/meta.json'))
	classes = meta['classes']
	temp_classes = [] 
	for i in classes:
		if(par.search(i)!=None):
			temp_classes.append(i[par.search(i).start()+2:])
		else:
			temp_classes.append(i)
	classes = temp_classes
	TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

	#Validation Accuracy Plot
	val_epochs = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/val_epochs.pickle','rb'))
	val_epochs = np.array(val_epochs)
	train_epochs = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/train_epochs.pickle','rb'))
	train_epochs = np.array(train_epochs)

	val_acc = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/val_accuracy.pickle','rb'))
	val_acc = np.array(val_acc)
	topk = val_acc.shape[1]
	val_acc_p = figure(title = 'Validation Accuracy', x_axis_label= 'Epochs',y_axis_label = 'Validation Accuracy',tools = TOOLS, toolbar_location = "below")
	for i in range(topk):
		val_acc_p.line(val_epochs,val_acc[:,i],legend = 'top-'+str(i+1),color = colors[i])
		val_acc_p.circle(val_epochs,val_acc[:,i],legend = 'top-'+str(i+1),color = colors[i])
	
	#Training Accuracy Plot
	train_acc = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/train_accuracy.pickle','rb'))
	train_acc = np.array(train_acc)
	train_acc_p = figure(title = 'Training Accuracy', x_axis_label= 'Epochs',y_axis_label = 'Training Accuracy',tools = TOOLS, toolbar_location = "below")
	for i in range(topk):
		train_acc_p.line(train_epochs,train_acc[:,i],legend = 'top-'+str(i+1),color = colors[i])
		train_acc_p.circle(train_epochs,train_acc[:,i],legend = 'top-'+str(i+1),color = colors[i])

	#Loss Plots
	train_loss = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/train_loss.pickle','rb'))
	train_loss = np.array(train_loss)
	val_loss = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/val_loss.pickle','rb'))
	val_loss = np.array(val_loss)
	loss_p = figure(title ='Losses',x_axis_label='epochs',y_axis_label='Loss Value',tools = TOOLS, toolbar_location = "below")
	loss_p.line(train_epochs,train_loss,legend='Training Loss',color=colors[0])
	loss_p.line(val_epochs,val_loss,legend='Validation Loss',color=colors[1])
	loss_p.circle(train_epochs,train_loss,legend='Training Loss',color=colors[0])
	loss_p.circle(val_epochs,val_loss,legend='Validation Loss',color=colors[1])


	#Learning Rate
	lr = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/learning_rate.pickle','rb'))	
	lr = np.array(lr)
	lr_p = figure(title ='Learning Rate',x_axis_label='Epochs',y_axis_label='learning rate',tools = TOOLS, toolbar_location = "below")
	lr_p.line(train_epochs,lr)
	lr_p.circle(train_epochs,lr)

	#class distribution
	class_dist = []
	for i in meta['classes']:
		class_dist.append(meta['train_class_frequency'][i] + meta['val_class_frequency'][i])
	class_cds = dict(class_labels= classes,class_freq=class_dist,)
	class_cds = ColumnDataSource(class_cds)
	class_p = figure(title = 'Class Distribution', tools =TOOLS,toolbar_location="below",x_range=classes)
	class_p.vbar(source=class_cds,top = 'class_freq',x ='class_labels',width=0.9)
	class_p.y_range.start=0
	class_p.select_one(HoverTool).tooltips = [
	     ('Class', '@class_labels'),
	     ('Frequency', '@class_freq'),
	]

	
	#Validation Confusion Matrix
	conf = pickle.load(open('/home/harsh/experiments/'+model+'/datastreams/val_confusion_matrix.pickle','rb'))
	conf = conf[-1]
	row_sum = conf.sum(axis=1)
	conf = conf/row_sum[:,np.newaxis]	
	data = pd.DataFrame(conf,index=classes,columns=classes)
	data.index.name = 'classes_ind'
	data.columns.name = 'classes_col'
	df = pd.DataFrame(data.stack(),columns=['value']).reset_index()
	
	colors1 = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
	mapper = LinearColorMapper(palette=colors1, low=df.value.min(), high=df.value.max())

	source = ColumnDataSource(df) 
	conf_p = figure(title="Latest Validation Confusion Matrix", x_range =classes, y_range=classes,x_axis_location = "below", tools = TOOLS, toolbar_location = "above")


	conf_p.grid.grid_line_color = None
	conf_p.axis.axis_line_color = None
	conf_p.axis.major_tick_line_color = None
	conf_p.axis.major_label_text_font_size = "5pt"
	conf_p.axis.major_label_standoff = 0
	conf_p.xaxis.major_label_orientation = pi / 3

	conf_p.rect(x = "classes_ind",y="classes_col",width=1,height=1,source=source, fill_color ={'field': 'value','transform': mapper},line_color=None)
	color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
			ticker=BasicTicker(desired_num_ticks=len(colors)),
                     	formatter=PrintfTickFormatter(format="%0.6f"),
                     	label_standoff=6, border_line_color=None, location=(0, 0))
	conf_p.add_layout(color_bar, 'right')
	conf_p.select_one(HoverTool).tooltips = [
	     ('classes', '@classes_ind and  @classes_col'),
	     ('value', '@value'),
	]


	
	return gridplot([val_acc_p,train_acc_p,loss_p,lr_p,conf_p,class_p],ncols=2)




#Index page

@app.route('/')


def index():
	current_model = request.args.get("model")
	if current_model == None:
		current_model = 'ROOT'

	plot = create_figure(current_model)
	
	script,div=components(plot)

	return render_template("newidx.html",models=models,script=script,div=div,current_model=current_model)



if __name__ == '__main__':
	app.run(host ='0.0.0.0',port=5000,debug=True)

		
