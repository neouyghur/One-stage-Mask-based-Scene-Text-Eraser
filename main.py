import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.MRTR import MRTR
from datetime import datetime


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)
    config = set_model_log_output_dir(config)
    # As it is a small file, I saved in both log and model directory
    ## TODO save as yml
    config.save(config.CONFIG_DIR)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # # build the model and initialize
    model = MRTR(config)
    model.load()

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        print("log dir: ", config.LOG_DIR)
        print("model dir: ", config.MODEL_DIR)
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def test_data_loader(mode=None):
    r"""Testing the dataloader

        Args:
            mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
        """

    config = load_config(mode)
    config = set_model_log_output_dir(config)
    # As it is a small file, I saved in both log and model directory
    config.save(config.OUTPUT_DIR)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        print("log dir: ", config.LOG_DIR)
        print("model dir: ", config.MODEL_DIR)
        # # TODO: remove below block if necessary
        from src.MRTR import Dataset
        dataset = Dataset(config, config.TRAIN_DATA, augment=True, training=True)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2)
        import time
        start_time = time.time()
        for batch_ndx, sample in enumerate(loader):
            img, img_gt, mask_pad, mask_gt, mask_org = sample
            print(batch_ndx)
            if batch_ndx > 100:
                break
        print(time.time() - start_time)

def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yml', help='model config file')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    # TODO: update
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()

    # # copy config template if does't exist
    # if not os.path.exists(config_path):
    #     copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(args.config)

    # train mode
    if mode == 1:
        # create checkpoints path if does't exist
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR)

        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)

        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        #config.MODEL = args.model if args.model is not None else 3
        # Hack
        config.INPUT_SIZE = 0
        #
        config._dict['WORD_BB_PERCENT_THRESHOLD'] = 0
        config._dict['CHAR_BB_PERCENT_THRESHOLD'] = 0
        config._dict['MASK_CORNER_OFFSET'] = 5

        # TODO: update this part
        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config

def set_model_log_output_dir(config):
    output_dir = config.OUTPUT_DIR
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    config_dir = os.path.join(*[os.path.expanduser(output_dir), subdir])
    log_dir = os.path.join(*[os.path.expanduser(output_dir), subdir, 'log'])
    model_dir = os.path.join(*[os.path.expanduser(output_dir), subdir, 'model'])
    test_dir = os.path.join(*[os.path.expanduser(output_dir), subdir, 'test'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config.LOG_DIR = log_dir
    config.MODEL_DIR = model_dir
    config.TEST_DIR = test_dir
    config.CONFIG_DIR = config_dir

    return config


if __name__ == "__main__":
    main()
    # test_data_loader()
