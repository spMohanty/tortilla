from PIL import Image

def default_loader(path):
	try:
		im = Image.open(path).convert('RGB')
	except:
		print("Unable to load file at path :", path)
	if im == None:
		print("Unable to load file at path : ", path)
	return im

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath, imlabel = line.strip().split()
			imlist.append( (impath, int(imlabel)) )

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
