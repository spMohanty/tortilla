#tortilla

```
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
```

`tortilla` is a python library for wrapping up all the nuts and bolts of image
classification using Deep Convolutional Neural Networks in a single package.

Powered by [PyTorch](http://pytorch.org/),
it has deep integrations with [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX) and [visdom](https://github.com/facebookresearch/visdom)
which help us have powerful visualisations to monitor the training, and also publish ready plots for training, evaluation and predictions.

As always, contributions welcome. :D

**NOTE** This is a work in progress :construction:, and will take a few more weeks before it is production ready.

# Installation
* Install Anaconda3 from [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
* Create your conda env and activate it :
```
  conda create python=3.6 --name tortilla
  source activate tortilla
```
* Install pytorch using :
```
  conda install pytorch torchvision -c pytorch #python3.6 + cuda8
  # or
  conda install pytorch torchvision cuda90 -c pytorch #python3.6 + cuda9.0
  # or
  conda install pytorch torchvision cuda91 -c pytorch #python3.6 + cuda9.1
  # or the safest
  conda install pytorch-cpu torchvision -c pytorch #python3.6 and cpu version of pytorch

  # More instructions for pytorch installation at : http://pytorch.org/
```
* Clone and install tortilla :
```
  git clone https://github.com/spMohanty/tortilla
  cd tortilla
  pip install -r requirements.txt
```
# Usage
## Prepare Data

For the task of image classification, [Tortilla](https://github.com/spMohanty/tortilla) expects the data to
be arranged in folders and subfolders. Create one root folder, and inside this root folder include one folder
each for every `class` in your classification problem. If your data are arranged in a taxonomy tree form, then go to the [taxonomy tree subsection](#taxonomy-tree)

The structure should look something like :
```
/-root_folder
----/class-1
----/class-2
----/class-3
----/class-4
....
...
and so on
```

Now, let say the root folder is present at `<root_folder_path>`. We will need to resize all the images, divide them into
train-test splits, etc.

This can be done by :

```
python scripts/data_preparation/prepare_data.py \
  --input-folder-path=CHANGE_ME_root_folder_path \
  --output-folder-path=datasets/CHANGE_ME_my_dataset_name \
  --dataset-name=CHANGE_ME_my_dataset_name
```

The total list of options available with the data preparation script are available at [docs/prepare-data.md](docs/prepare-data.md).

If the previous script executed without any errors, then you should have a new datasets folder at `datasets/CHANGE_ME_my_dataset_name`
which should have resized versions of all the images, and also split into training and validation sets.

You can now start to [train your model](#training)

### Taxonomy tree

The structure of your data should look like this:
```
/-root_folder
---class1
------class1.a
---------class1.a.1
---------class1.a.2
---------....
------class1.b
------...
---class2
... and so on
```
Now, let say the root folder is present at `<root_folder_path>`. The structure of the tree involves that the classification can be done at many different levels (e.g. class1/class2 at root level; class1.a/class1.b at class1 level and so on...).
We first need to create a dataset for each level and then prepare these datasets into the tortilla format: train/test splits as well as resize the images.

To do so with all default values of the data preparation script, you can use :
```
sh scripts/misc/create_recursive_datasets.sh datasets CHANGE_ME_root_folder_path
```
This creates a new folder `datasets` that contains all your different experiments ready to use for training.

If you want to pimp the options, they are available at [docs/prepare-data.md](docs/prepare-data.md) and then modify the bash script `scripts/misc/create_recursive_datasets.sh` according to your will.


## Training

Training will be done by the `tortilla-train.py` script, and a list of all the available options are available at : [docs/tortilla-trainer.md](docs/tortilla-trainer.md).

### Tensorboard visualization

Start training by :

```
python tortilla-train.py \
  --experiment-name=CHANGE_ME_my_dataset_name \
  --dataset-dir=datasets/CHANGE_ME_my_dataset_name \
```

Then, in order to start the visualization, do:

```
# Please do this in a separate terminal tab
source activate tortilla
tensorboard --logdir experiments/CHANGE_ME_my_dataset_name/tb_logs/
```
In a bit, after you code starts running, you should be able to see all the plots, etc at
`http://localhost:6006` under scalars and images sections.


### Visdom visualization

Before we can start the training, we need to start the `visdom` server, which will help
us visualize the actual training details. This can be done by :
```
# Please do this in a separate terminal tab
source activate tortilla
python -m visdom.server
```
This should start a local visdom server with which the actual training code can interact.

Now we can start training by :

```
python tortilla-train.py \
  --experiment-name=CHANGE_ME_my_dataset_name \
  --dataset-dir=datasets/CHANGE_ME_my_dataset_name \
  --plot-platform=visdom
```
In a bit, after you code starts running, you should be able to see all the plots, etc at
`http://localhost:8097`
(If you didnot change the `visdom-server` and `visdom-port` using the corresponding cli flags)

## Predictions

When training is over, you can now start predicting classes for new images. This will be done with the `tortilla-predict.py` script.

All images should be in one single folder at let's say `<new_images_path>`.

Then, launch predictions by:

```
python tortilla_predict.py \
  --model-path=experiments/CHANGE_ME_my_dataset_name/trained_model.net \
  --prediction-dir=CHANGE_ME_new_images_path
```

At the end, you will have a prediction.json file in the  `experiments/CHANGE_ME_my_dataset_name` folder containing the predicted class for all your images.

## Tortilla-Serve
You can use the tortilla_serve to run a webapp for making predictions using trained tortilla models on a batch of images.

``` python tortilla_serve.py MODEL_LIST_DIRECTORY UPLOAD_FOLDER
```

This starts a [Flask](http://flask.pocoo.org/) app on your localhost (default port number : 5001)
You can upload files to the upload folder and select your preferred model from those present in MODEL_LIST_DIRECTORY.
The tortilla_serve assumes that the MODEL_LIST DIRECTORY contains only folders with each folder having a
trained_model.net file.

## Alternate Visualization
To generate plots from training output of Tortilla, the visuals.py file can be used.

``` python visuals.py MODEL_LIST_DIRECTORY DATASET_DIRECTORY
```

The MODEL_LIST_DIRECTORY contains folders each having logs from tortilla training while the DATASET_DIRECTORY contains
the training images(in Tortilla input format)  corresponding to these models.
Select the model from the dropdown menu.
These visuals are served via a Flask app on the localhost using Bokeh for rendering the plots(Default port : 5002)
# Run the tests

In order to ensure the expected functionality of the code, tests were implemented using [Nosetests](http://nose.readthedocs.io/en/latest/). The following code can be run after having installed tortilla.

* First, install `nose`:

```
pip install nose
```

* Run all the tests by typing:

```
nosetests
```

This should run 13 tests without throwing errors.


# Authors

Sharada Mohanty (sharada.mohanty@epfl.ch)   
Camille Renner (camille.renner@epfl.ch)
