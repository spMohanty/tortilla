from PIL import Image
import pandas as pd
import sys,os
import shutil
import json
import datetime

def default_loader(path):
	try:
		im = Image.open(path).convert('RGB')
	except:
		print("Unable to load file at path :", path)
		exit()
	if im == None:
		print("Unable to load file at path : ", path)
	return im

def default_flist_reader(flist, classes):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	print(flist)
	imlist = []
	with open(flist, 'r') as fp:
		data = json.loads(fp.read());
		for filepath in data.keys():
			imlabel = int(data[filepath])
			assert imlabel < len(classes)
			imlist.append( (filepath, imlabel) )
	return imlist

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def append_val(var, key, value):
	try:
		foo = var[key]
	except:
		var[key] = []
	#
	# var[key].append(value)

def save_to_csv(config, experiment_dir):
	EXP = pd.read_csv(os.path.join(experiment_dir, 'Experiments.csv'), \
						sep=';', index_col=['Type','Variables'])
	date = datetime.date.today().strftime("%d-%b-%Y")
	name  = ":".join([date, config.experiment_name])

	meta = json.loads(open(os.path.join(config.dataset_dir, "meta.json")).read())
	meta_data = pd.DataFrame(list(meta.items()), columns=['Variables', name])
	meta_data = meta_data.set_index('Variables')

	var = vars(config)
	attributes = {key: var[key] for key in var if not key.startswith('__')}
	config_data = pd.DataFrame(list(attributes.items()), columns=['Variables', name])
	config_data = config_data.set_index('Variables')

	comments = {'General_impression':None,'Suggested_changes':None}
	comments = pd.DataFrame(list(comments.items()), columns=['Variables', name])
	comments = comments.set_index('Variables')

	NEW_EXP = pd.concat([meta_data, config_data, comments], \
						keys=['Meta_data', 'Config_data', 'Comments'], \
						names= ['Type', 'Variables'])
	EXP = pd.concat([EXP,NEW_EXP], axis=1)

	EXP.to_csv(os.path.join(experiment_dir, 'Experiments.csv'), sep=';')

def query_yes_no(question, default="yes"):
	"""Ask a yes/no question via raw_input() and return their answer.

	"question" is a string that is presented to the user.
	"default" is the presumed answer if the user just hits <Enter>.
		It must be "yes" (the default), "no" or None (meaning
		an answer is required of the user).

	The "answer" return value is True for "yes" or False for "no".
	"""
	valid = {"yes": True, "y": True, "ye": True,
			 "no": False, "n": False}
	if default is None:
		prompt = " [y/n] "
	elif default == "yes":
		prompt = " [Y/n] "
	elif default == "no":
		prompt = " [y/N] "
	else:
		raise ValueError("invalid default answer: '%s'" % default)

	while True:
		sys.stdout.write(question + prompt)
		try:
			choice = raw_input().lower()
		except:
			choice = input().lower()

		if default is not None and choice == '':
			return valid[default]
		elif choice in valid:
			return valid[choice]
		else:
			sys.stdout.write("Please respond with 'yes' or 'no' "
							 "(or 'y' or 'n').\n")

def create_directory_structure(root, resume=False):
	if resume:
		if not os.path.exists(root):
			raise Exception("You are running tortilla in `resume` mode, but the experiment_directory does not exist. Are you sure you passed the correct experiment name ?")
		else:
			print("Found experiment directory at : ", root)
			"""
			Assume directory structure exists
			"""
			return
	if os.path.exists(root):
		if query_yes_no(("Path {} exists. Continute deleting ?".format(root))):
			shutil.rmtree(root)
		else:
			print("Exiting without doing anything....")
			exit(0)
	print("Creating directory structure at : {}".format(root))
	os.mkdir(root)
	os.mkdir(root+"/datastreams")
	os.mkdir(root+"/logs")
	os.mkdir(root+"/checkpoints")

def logo():
	print("""
==============================================================================================================================
         tttt                                                        tttt            iiii  lllllll lllllll
      ttt:::t                                                     ttt:::t           i::::i l:::::l l:::::l
      t:::::t                                                     t:::::t            iiii  l:::::l l:::::l
      t:::::t                                                     t:::::t                  l:::::l l:::::l
ttttttt:::::ttttttt       ooooooooooo   rrrrr   rrrrrrrrr   ttttttt:::::ttttttt    iiiiiii  l::::l  l::::l   aaaaaaaaaaaaa
t:::::::::::::::::t     oo:::::::::::oo r::::rrr:::::::::r  t:::::::::::::::::t    i:::::i  l::::l  l::::l   a::::::::::::a
t:::::::::::::::::t    o:::::::::::::::or:::::::::::::::::r t:::::::::::::::::t     i::::i  l::::l  l::::l   aaaaaaaaa:::::a
tttttt:::::::tttttt    o:::::ooooo:::::orr::::::rrrrr::::::rtttttt:::::::tttttt     i::::i  l::::l  l::::l            a::::a
      t:::::t          o::::o     o::::o r:::::r     r:::::r      t:::::t           i::::i  l::::l  l::::l     aaaaaaa:::::a
      t:::::t          o::::o     o::::o r:::::r     rrrrrrr      t:::::t           i::::i  l::::l  l::::l   aa::::::::::::a
      t:::::t          o::::o     o::::o r:::::r                  t:::::t           i::::i  l::::l  l::::l  a::::aaaa::::::a
      t:::::t    tttttto::::o     o::::o r:::::r                  t:::::t    tttttt i::::i  l::::l  l::::l a::::a    a:::::a
      t::::::tttt:::::to:::::ooooo:::::o r:::::r                  t::::::tttt:::::ti::::::il::::::ll::::::la::::a    a:::::a
      tt::::::::::::::to:::::::::::::::o r:::::r                  tt::::::::::::::ti::::::il::::::ll::::::la:::::aaaa::::::a
        tt:::::::::::tt oo:::::::::::oo  r:::::r                    tt:::::::::::tti::::::il::::::ll::::::l a::::::::::aa:::a
          ttttttttttt     ooooooooooo    rrrrrrr                      ttttttttttt  iiiiiiiillllllllllllllll  aaaaaaaaaa  aaaa
==============================================================================================================================
	""")
