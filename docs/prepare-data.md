# prepare-data.py

The exhaustive list of options for `prepare-data.py`
```
usage: prepare_data.py [-h] --input-folder-path INPUT_FOLDER_PATH
                       --output-folder-path OUTPUT_FOLDER_PATH
                       [--min-images-per-class MIN_IMAGES_PER_CLASS]
                       [--train-percent TRAIN_PERCENT] --dataset-name
                       DATASET_NAME [--img-size IMG_SIZE]

Converts dataset from folder-subfolder format (one sub folder per class) to
tortilla's data format

optional arguments:
  -h, --help            show this help message and exit
  --input-folder-path INPUT_FOLDER_PATH
                        Path to input folder containing images (default: None)
  --output-folder-path OUTPUT_FOLDER_PATH
                        Path to output folder to write images (default: None)
  --min-images-per-class MIN_IMAGES_PER_CLASS
                        Minimum number of images required per class (default:
                        500)
  --max-images-per-class MIN_IMAGES_PER_CLASS
                        Maximum number of images required per class (default:
                        20000)
  --train-percent TRAIN_PERCENT
                        Percentage of all images to use as training data
                        (default: 0.8)
  --dataset-name DATASET_NAME
                        Name of the Dataset (default: None)
  --img-size IMG_SIZE   Size of the target images (default: 256x256)
  --non-interactive-mode
                        If do not want interaction with terminal
  --absolute-path       If want to use absolute paths
  --no-copy             If do not want to copy the images
```
