from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TRAIN.batch_size = 16  # [16] use 8 if your GPU memory is small
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 1
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/data1/Sakir/oct_aggregated'
config.TRAIN.lr_img_path = '/media/sakir/Big/Dataset/Archive/OCT_32x32/train'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/media/sakir/Big/Dataset/Archive/OCT_128x128/val'
config.VALID.lr_img_path = '/media/sakir/Big/Dataset/Archive/OCT_32x32/val'

# Add PyTorch specific configurations
config.TRAIN.device = 'cuda'  # or 'cpu'
config.TRAIN.num_workers = 4  # for data loading
config.TRAIN.pin_memory = True  # for faster data transfer to GPU

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")