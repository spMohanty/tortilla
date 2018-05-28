class Config:
    experiment_name = "test"
    experiments_dir = "experiments"
    dataset_dir = "datasets/food-101"
    experiment_dir_name = experiments_dir+"/"+experiment_name
    model="resnet50"
    optimizer="adam"
    learning_rate = 0.01
    epochs = 50
    batch_size = 128
    topk = (1,2,3,4,5,6,7,8,9,10)
    debug=False
    train_flush_per_epoch = 10
    normalize_confusion_matrix = True
    resume = False

    num_cpu_workers = 4
    plot_platform = "tensorboard"
    no_plots = False
    no_render_images = True
    visdom_server = "localhost"
    visdom_port = 8097
    wrs = False
    wloss = False		
    use_cpu = False
    if debug:
        batch_size = 101
        epochs = 5
 
    input_size = 224
    resize_shape = 256			
    no_data_augmentation = False
    data_transforms = None
    checkpoint_frequency = 5
    normalize_params = [0.485, 0.456, 0.406, 0.229, 0.224, 0.225]		
    version = 0.01
