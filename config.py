import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMAGE_DIR = "/content/imgs/train"
TRAIN_DIR = "/content/imgs/train"
# VAL_DIR = "data/val"
ANNOTATION_FILE_DIR = "/content/driver_imgs_list.csv"
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
MOMENTUM = 0.9
LOAD_MODEL = False
SAVE_MODEL = True
SEED = 42
CHECKPOINT_GEN_P = "genp.pth.tar"
CHECKPOINT_GEN_M = "genm.pth.tar"
CHECKPOINT_CRITIC_P = "criticp.pth.tar"
CHECKPOINT_CRITIC_M = "criticm.pth.tar"
