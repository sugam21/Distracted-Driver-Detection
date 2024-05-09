import os
import random
from datetime import datetime

import config
import numpy as np
import torch


def save_checkpoint(
    model, optimizer, filepath: str, filename: str = "my_checkpoint.pth.tar"
) -> None:
    """
    Takes mode, optimizer and complete path of the folder as well as file name and saves the model weight as well as optimizer weight to that path
    params:
      model (torch model): It is pytorch model which you finished training and want to save
      optimizer (torch optimizer): It is a pytorch optimizer which holds the paramter weights for current model
      filepath(str): It is the path of the directory where you wish to save your weights.
                     It is taken from  CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/CV/saved_checkpoints" in config.py file
      filename(str): Actual name of the checkpoint which is tar.gz file

    returns:
      None
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_file_complete_path = os.path.join(filepath, filename)
    # If path does not exists create one
    if not (os.path.exists(filepath)):
        os.mkdir(filepath)

    torch.save(checkpoint, checkpoint_file_complete_path)


def load_checkpoint(
    checkpoint_file: str, filepath: str, model, optimizer, lr: float
) -> None:
    """
    Takes mode, optimizer and complete path of the folder as well as file name and loads the saved model in the passed model parameter
    params:
      model (torch model): It is pytorch model to which you want to insert pretrained parameters

      optimizer (torch optimizer): It is a pytorch optimizer to which you want to add pretrained gradients
      filepath(str): It is the path of the directory from where you wish to access your saved checkpoint.
                     It is taken from  CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/CV/saved_checkpoints" in config.py file
      filename(str): Actual name of the checkpoint which is tar.gz file

    returns:
      None
    """

    print("=> Loading checkpoint")
    # join both the filepath and actual file name to make complete path
    checkpoint_file_complete_path = os.path.join(filepath, checkpoint_file)
    checkpoint = torch.load(checkpoint_file_complete_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=config.SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_log_folder()->str:
    """
    This function creates a folder with current date and time as name to store log files
    params:
      None
    returns:
      log_folder_path(str): path of the created log folder
    """
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder_path = os.path.join(config.LOG_SAVE_DIR, f"{timestamp}")
    try:
        os.mkdir(log_folder_path)
        print(f"Folder with name {log_folder_path} created successfully")
    except OSError as e:
        print(f"Failed to create folder {log_folder_path}: {e}")
    else:
        with open(log_folder_path + "/log.log", "w") as f:
            f.write(
                f"{'Epoch':<8}{'training_accuracy':<20}{'validation_accuracy':<20}"
                f"{'training_loss':<20}{'validation_loss':<20}"
                f"{'Time':<20}\n"
            )
        return log_folder_path


def save_result(
    epoch: int,
    training_accuracy: float,
    validation_accuracy: float,
    training_loss: float,
    validation_loss: float,
    log_folder_path,
):
    with open(log_folder_path + "/log.log", "a") as f:
        f.write(
            f"{epoch:<8}{training_accuracy:<20}{validation_accuracy:<20}"
            f"{training_loss:<20}{validation_loss:<20}"
            f"{datetime.now().strftime('%Y-%m-%dT::%H:%M:%S'):>20}\n"
        )