import os
from os.path import join, dirname, abspath
import sys
import time
import random

import logging
import argparse
from configparser import ConfigParser, ExtendedInterpolation

import torch
from torch.optim import lr_scheduler
import numpy as np

import model
from train_net import TrainNet


def get_args():
    parser = argparse.ArgumentParser(description='Train the Net on images and target labels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", dest="model", nargs='+', type=str, default=False,
                             required=False, metavar="MD", 
                             help="name of model to use, encoder -> decoder -> l_dim'")
    parser.add_argument("--dtset", dest="dtset", type=str, default=False,
                             required=False, metavar="DT", 
                             help="name of dataset to use")
    parser.add_argument("--device", dest="device", nargs='+', type=str, default=False,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=True, metavar="CFG", 
                             help="configuration file")
    parser.add_argument("--load", dest="load", nargs='+', type=str, default=False,
                             required=False, metavar="LD", 
                             help="load trainable model, path -> *.pth -> ep0")
    parser.add_argument("--test", dest="test", type=str, default=False,
                             required=False, metavar="IF", 
                             help="test mode")
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
    config.optionxform = str
    config.read(args.path_cfg)

    cfg_dft = config['DEFAULT']
    cfg_train = config['TRAIN']
    cfg_model = config['MODEL']
    cfg_dt = config['DATASET']

    if args.load:
        dir_ld = f"{cfg_dft['root_project']}/results/{args.load[0]}"
        config.read(os.path.join(dir_ld, 'info_train.ini'))
    else:
        cfg_dft['HOME'] = abspath(join(dirname(abspath(__file__)), os.pardir))
        if args.model: cfg_model['name'] = args.model
        if args.dtset: cfg_dt['name'] = args.dtset
        if args.device: cfg_train.update({'decvice': args.device[0], 'device_ids': args.device[1]})
    logging.info(f"Configuration file {args.path_cfg} loaded")
    
    # for reproducibility
    rseed = cfg_dft.getint('seed')
    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # set main device and devices
    dvc = cfg_train["device"]
    ids = cfg_train["device_ids"]
    lst_ids = [int(id) for id in ids]
    dvc_main = torch.device(f"{dvc}:{ids[0]}")

    # MODEL
    kwargs = {
        k: eval(v) 
        for k, v in config.items(cfg_model['name'])
        if k not in cfg_dft.keys()
    }
    net = getattr(model, cfg_model['name'])(**kwargs).to(device=dvc_main)

    if dvc == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(lst_ids)[1:-1]

        net = torch.nn.DataParallel(net, device_ids=lst_ids)
        logging.info(f"Using main device <{dvc_main}> and devices with <IDs: {ids}>")
    else:
        logging.info(f"Using main device <{dvc_main}>")

    # OPTIMIZER
    kwargs_optim = {
        k: eval(v) 
        for k, v in config.items(cfg_train['optim'].upper())
        if k not in cfg_dft.keys()
    }
    name_optim = cfg_train['optim']
    optim = getattr(torch.optim, name_optim)(net.parameters(), **kwargs_optim)

    # SCHEDULER
    kwargs_scheduler = {
        k: eval(v) 
        for k, v in config.items(cfg_train['scheduler'].upper())
        if k not in cfg_dft.keys()
    }
    name_scheduler = cfg_train['scheduler']
    scheduler = getattr(lr_scheduler, name_scheduler)(optim, **kwargs_scheduler)

    # Load parameters
    if args.load:
        file_ld = os.path.join(dir_ld, f"checkpoints/{args.load[1]}.pth")
        cfg_train['start_epoch'] = args.load[2]

        checkpoint = torch.load(file_ld, map_location='cpu')
        net.load_state_dict(checkpoint['encoder_state_dict'])
        optim.load_state_dict(checkpoint['optim_encoder_state_dict'])

        net.to(device=dvc_main)
        logging.info(f'Model loaded from {file_ld}')
    else:
        init_time = time.strftime("%H%M_%d%b%Y", time.localtime(time.time()))
        cfg_dft["current_time"] = init_time
        sv_dir = config["SAVE"]["dir_save"]
        try:
            os.mkdir(sv_dir)
            logging.info(f"Created save directory {sv_dir}")
        except OSError:
            pass

        with open(f"{sv_dir}/info_train.ini", 'w') as f:
            f.write('; sys.argv\n; ')
            [f.write(ag + ' ') for ag in sys.argv]
            f.write('\n')
            config.write(f)

    try:
        nets = [net, ]
        optims = [[optim, scheduler], ]

        print('OK!! Let\'s exit!!')
        sys.exit()

        train_net = TrainNet(nets=nets, optims=optims, device_main=dvc_main, config=config)
        for epch in range(int(config['TRAIN']['start_epoch']), int(config['TRAIN']['num_epoch'])):
            train_net.training(epch)
            train_net.validation(epch)
            train_net.saving(epch)
        train_net.writer_main.close()
        train_net.writer_test.close()

    except KeyboardInterrupt:
        set_sig = [{'y', 'yes'}, {'n', 'no'}]
        while True:
            try:
                _sig = input(f"\nDo you want to save current state? [yes / no] ")
            except KeyboardInterrupt:
                continue
            if _sig.lower() not in set_sig[0].union(set_sig[1]):
                continue
            elif _sig.lower() in set_sig[0]:
                logging.info("Saving current state of network ...")
                torch.save({
                    'encoder_state_dict': net.state_dict(),
                    'optim_encoder_state_dict': optim.state_dict(),
                }, 'INTERRUPTED.pth')
                break
            else:
                break
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
        