# Data Prepartion

`tortilla` expects a data folder to have the following structure :

- `/root`
  - `images` : Folder containing all the images
  - `train.txt` : A text file with each line containing information about
                  images in the training set.   
                  Format :    
                  `<image_path> class_idx \n`   
                  *NOTE* The image path is relative the document root
  - `val.txt` : A text file with each line containing information about
                  images in the validation set.   
                  Format :    
                  `<image_path> class_idx \n`   
                  *NOTE* The image path is relative the document root

  - `classes.txt` : A text file containing the name of all the classes in the
                  dataset. The line number of the class_name is the `class_idx`
                  used in `train.txt` and `val.txt`

# Author
Sharada Mohanty <sharada.mohanty@epfl.ch>
