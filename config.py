class Config:
    # data_dir = "/mount/SDE/instagram/new_data_pruned/ebs/downloads_filtered"
    data_dir = "/mount/SDE/instagram/food101/food-101/images"
    learning_rate = 0.001
    epochs = 100
    batch_size = 128

    debug=True
    if debug:
        batch_size = 100
        epochs = 5
        images_per_epoch = 1000
