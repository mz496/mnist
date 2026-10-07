from datasets import DatasetDict, load_dataset

ds: DatasetDict = load_dataset("ylecun/mnist")

# TODO add config param
def get_dataloaders():
    split = ds["train"].train_test_split(test_size=5000, seed=42)
    train, val = split["train"], split["test"]   # 55k / 5k
    test = ds["test"]
    print("train", train)
    print("val  ", val)
    print("test ", test)
    return train, val, test
