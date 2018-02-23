class Config:
    experiment_name = "test-food-101"
    experiments_dir = "experiments"
    experiment_dir_name = experiments_dir+"/"+experiment_name
    learning_rate = 0.001
    epochs = 100
    batch_size = 128
    topk = (1,2,3,4,5,6,7,8,9,10)
    debug=True
    train_flush_per_epoch = 100
    normalize_confusion_matrix = True
    if debug:
        batch_size = 101
        epochs = 5
        images_per_epoch = 1000
