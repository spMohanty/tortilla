# tortilla-trainer.py

The exhaustive list of options for `tortilla-trainer.py`

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

usage: tortilla-train.py [-h] --experiment-name EXPERIMENT_NAME
                         [--experiments-dir EXPERIMENTS_DIR] --dataset-dir
                         DATASET_DIR [--model MODEL] [--optimizer OPTIMIZER]
                         [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                         [--learning-rate LEARNING_RATE] [--top-k TOP_K]
                         [--num-cpu-workers NUM_CPU_WORKERS]
                         [--visdom-server VISDOM_SERVER]
                         [--visdom-port VISDOM_PORT] [--no-plots] [--use-cpu]
                         [--debug] [--version][--wrs][--wloss]
			 [--normalize-params PARAMS ]
			 [--checkpoint-frequency CHECKPOINT_FREQUENCY]	
			
optional arguments:
  -h, --help            show this help message and exit
  --experiment-name EXPERIMENT_NAME
                        A unique name for the current experiment (default:
                        None)
  --experiments-dir EXPERIMENTS_DIR
                        Directory where results of all experiments will be
                        stored. (default: experiments)
  --dataset-dir DATASET_DIR
                        Dataset directory in the TortillaDataset format
                        (default: None)
  --model MODEL         Type of the pretrained network to train with. Options
                        : ['alexnet', 'densenet121', 'densenet161',
                        'densenet169', 'densenet201', 'inception_v3',
                        'resnet101', 'resnet152', 'resnet18', 'resnet34',
                        'resnet50', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                        'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                        'squeezenet1_0'] (default: resnet-50)
  --optimizer OPTIMIZER
                        Type of the pretrained network to train with. Options
                        : ["adam"] (default: adam)
  --batch-size BATCH_SIZE
                        Batch Size. (default: 128)
  --epochs EPOCHS       Number of epochs. (default: 100)
  --learning-rate LEARNING_RATE
                        Learning Rate. (default: 0.001)
  --top-k TOP_K         List of values to compute top-k accuracies during
                        train and val. (default: 1,2,3,4,5,6,7,8,9,10)
  --num-cpu-workers NUM_CPU_WORKERS
                        Number of CPU workers to be used by data loaders.
                        (default: 4)
  --visdom-server VISDOM_SERVER
                        Visdom server hostname. (default: localhost)
  --visdom-port VISDOM_PORT
                        Visdom server port. (default: 8097)
  --plot-platform PLOT_PLATFORM
                        Type of visualization platform. Options:["tensorboard",
                        "visdom", "none"] (default: tensorboard)
  --no-plots            Disable plotting on the visdom server (default: False)
  --no-render-images    Disable plotting training images example (default: False)
  --no-data-augmentation
                        Disable data augmentation (default: False)
  --use-cpu             Boolean Flag to forcibly use CPU (on servers which
                        have GPUs. If you do not have a GPU, tortilla will
                        automatically use just CPU) (default: False)
  --debug               Run tortilla in debug mode (default: False)
  --version             show program's version number and exit
  --wrs 		Use pytorch's WeightedRandomSampler method with inverse class frequency as weights
  --wloss		Use pytorch's WeightedLoss method with class frequency as weights
  --normalize-params  PARAMS
			List of 6 space separated entries which determine means and variances for normalization of each image (PARAMS must have 6 entries else it is not processed)		
  --checkpoint-frequency CHECKPOINT_FREQUENCY
			Saves checkpoints at given frequency(default: 5)	
