import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content/imgs/train"
ANNOTATION_FILE_DIR = "/content/driver_imgs_list.csv"
CHECKPOINT_SAVE_DIR = "/content/drive/MyDrive/Academics/CV/saved_checkpoints"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
# NUM_WORKERS = 2
NUM_EPOCHS = 20
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
MOMENTUM = 0.9
LOAD_MODEL = False
SAVE_MODEL = True
SEED = 42
CHECKPOINT_FILENAME_CUSTOM_NETWORK = "custom_network.pth.tar"
CHECKPOINT_FILENAME_RESNET_NETWORK = "resnet18.pth.tar"
