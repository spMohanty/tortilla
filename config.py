from torch.nn import CrossEntropyLoss

class Config:
    data_dir = "/mount/SDE/instagram/new_data_pruned/ebs/downloads_filtered"
    learning_rate = 0.001
    criterion = CrossEntropyLoss
