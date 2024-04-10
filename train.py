from typing import Tuple
from pandas.io.sql import get_option
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# custom files
import config
from network import Network
import utils
from dataset import CustomDataset, DataTransform


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


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # set the model to training mode - imp for batch norm and dropout layers
    model.train()
    for batch, (image_batch, labels) in tqdm(
        enumerate(dataloader), unit="batch", total=len(dataloader)
    ):
        (image_batch, labels) = (
            image_batch.to(config.DEVICE),
            labels.to(config.DEVICE),
        )
        # compute the prediction
        prediction = model(image_batch)
        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * config.BATCH_SIZE + len(image_batch)
            print(f"Loss: {loss:>7f} | [{current:>5d}/{size:>5d}] ")


def test_loop(dataloader, model, loss_fn):
    # set the model to eval mode - imp for batch norma and dropout
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluate the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also reduce unnecessary gradients computations and memory usage for
    # tensors with requires_grad = True
    with torch.no_grad():
        for image_batch, labels in dataloader:
            (image_batch, labels) = (
                image_batch.to(config.DEVICE),
                labels.to(config.DEVICE),
            )
            pred = model(image_batch)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error : \n Accuracy: {100 * correct:0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    df = pd.read_csv(config.ANNOTATION_FILE_DIR)
    df["class_code"] = df["classname"].map(lambda x: int(x[-1]))
    df = df[["subject", "classname", "class_code", "img"]]

    ## CSV train and valid data
    train_set, valid_set = get_train_valid_in_csv(df)
    #
    # Actual image training data
    training_data, validation_data = get_train_valid_in_image(train_set, valid_set)

    train_dataloader = DataLoader(
        training_data, batch_size=config.BATCH_SIZE, shuffle=True
    )
    valid_dataloader = DataLoader(
        validation_data, batch_size=config.BATCH_SIZE, shuffle=True
    )

    model = Network().to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch+1}\n -----------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
