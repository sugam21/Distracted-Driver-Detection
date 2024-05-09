from test import test_loop
from resnet18_train import train_loop

# custom files
import config
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from dataset import CustomDataset, DataTransform
from network import Network
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# For weights visualization
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint

code_class_mapping = {
    "c0": "Safe driving",
    "c1": "Texting - right",
    "c2": "Talking on the phone - right",
    "c3": "Texting - left",
    "c4": "Talking on the phone - left",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger",
}


def get_train_valid_in_csv(df):
    """
    Takes the data frame file for the information about dataset and seperate training and testing parat
    Params:
      df (pd.DataFrame): DataFrame containing the information about entire dataset
    Returns:
      Tuple(pd.DataFrame,pd.DataFrame): Tuple containing training and testing set
    """

    # Seperating train and validation sets
    train_set, valid_set = train_test_split(
        df, shuffle=True, random_state=config.SEED, stratify=df["classname"]
    )
    return (train_set, valid_set)


def get_train_valid_in_image(train_set, valid_set):
    # Actual image training data
    training_data = CustomDataset(
        train_set,
        config.TRAIN_DIR,
        is_train=True,
        transform=DataTransform(
            config.INPUT_SIZE, config.CHANNEL_MEAN, config.CHANNEL_STD
        ),
    )
    validation_data = CustomDataset(
        valid_set,
        config.TRAIN_DIR,
        is_train=False,
        transform=DataTransform(
            config.INPUT_SIZE, config.CHANNEL_MEAN, config.CHANNEL_STD
        ),
    )
    return training_data, validation_data


def main(is_load: bool, is_save: bool) -> None:
    df = pd.read_csv(config.ANNOTATION_FILE_DIR)
    df["class_code"] = df["classname"].map(lambda x: int(x[-1]))
    df = df[["subject", "classname", "class_code", "img"]]

    ## CSV train and valid data
    train_set, valid_set = get_train_valid_in_csv(df)
    #
    # Actual image training data
    training_data, validation_data = get_train_valid_in_image(train_set, valid_set)

    train_dataloader = DataLoader(
        training_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        validation_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    model = Network().to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # if load pre-trained model is true
    if is_load:
        load_checkpoint(
            checkpoint_file=config.CHECKPOINT_FILENAME_CUSTOM_NETWORK,
            filepath=config.CHECKPOINT_SAVE_DIR,
            model=model,
            optimizer=optimizer,
            lr=config.LEARNING_RATE,
        )
    # Creataing folder to store log files
    log_folder_path = utils.create_log_folder()

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH: {epoch+1}\n -----------------------------------")
        training_accuracy: float = train_loop(
            train_dataloader, model, loss_fn, optimizer
        )

        # save checkpoint during training
        if is_save:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filepath=config.CHECKPOINT_SAVE_DIR,
                filename=config.CHECKPOINT_FILENAME_CUSTOM_NETWORK,
            )
        validation_accuracy: float = test_loop(valid_dataloader, model, loss_fn)
        utils.save_result(
            epoch, training_accuracy, validation_accuracy, log_folder_path
        )
    print("Done!")


if __name__ == "__main__":
    main(is_load=True, is_save=True)
