class Config:
    experiment_name = "test-food-101"
    experiments_dir = "experiments"
    dataset_dir = "datasets/food-101"
    experiment_dir_name = experiments_dir+"/"+experiment_name
    model="resnet-50"
    optimizer="adam"
    learning_rate = 0.001
    epochs = 100
    batch_size = 128
    topk = (1,2,3,4,5,6,7,8,9,10)
    debug=False
    train_flush_per_epoch = 100
    normalize_confusion_matrix = True

    num_cpu_workers = 4
    no_plots = False
    no_render_images = False
    visdom_server = "localhost"
    visdom_port = 8097

    use_cpu = False
    if debug:
        batch_size = 101
        epochs = 5

    version = 0.01
