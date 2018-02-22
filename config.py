class Config:
    # data_dir = "/mount/SDE/instagram/new_data_pruned/ebs/downloads_filtered"
    data_dir = "/mount/SDE/instagram/food101/food-101/images"
    learning_rate = 0.001
    epochs = 100
    batch_size = 128
    topk = (1,2,3,4,5,6,7,8,9,10)
    debug=True
    train_flush_per_epoch = 100
    if debug:
        batch_size = 101
        epochs = 5
        images_per_epoch = 1000
