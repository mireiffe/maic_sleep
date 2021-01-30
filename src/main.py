import os
import sys
import time
import random

import logging
import argparse
from configparser import ConfigParser, ExtendedInterpolation

import torch
import numpy as np

from src.train_net import TrainNet


def get_args():
    parser = argparse.ArgumentParser(description='Train the Net on images and target labels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", dest="model", nargs='+', type=str, default=False,
                             required=False, metavar="MD", 
                             help="name of model to use, encoder -> decoder -> l_dim'")
    parser.add_argument("--dtset", dest="set_dt", type=str, default=False,
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
    config.read(args.path_cfg)

    config['DEFAULT']['HOME'] = os.path.expanduser('~')
    if args.load:
        dir_ld = f"{config['DEFAULT']['root_project']}/results/{args.load[0]}"
        config.read(os.path.join(dir_ld, 'info_train.ini'))
    logging.info(f"Configuration file {args.cfg} loaded")
    
    dflt_cfg = config['DEFAULT']
    train_cfg = config['TRAIN']

    if args.model:
        _e, _d, _l = args.model
        config['MODEL'].update({'encoder': _e, 'decoder': _d, 'dim_l': _l})
    else: _e, _d, _l = config['MODEL']['encoder'], config['MODEL']['decoder'], config['MODEL']['dim_l']
    if args.set_dt: config['DATASET']['data'] = args.set_dt
    if args.device: config['TRAIN']['device'] = args.device[0]
    if args.device: config['TRAIN']['device_ids'] = args.device[1]

    rseed = dflt_cfg.getint('seed')
    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    dvc = config["TRAIN"]["device"]
    ids = config["TRAIN"]["device_ids"]
    lst_ids = [int(id) for id in ids]
    dvc_main = torch.device(f"{dvc}:{ids[0]}")

    ch_img = ch_dataset[config["DATASET"]["data"]]
    _model = __import__("src.model", fromlist=[_e, _d])
    net = getattr(_model, _e)(ch_img, int(_l)).to(device=dvc_main)

    if dvc == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(lst_ids)[1:-1]

        net = torch.nn.DataParallel(net, device_ids=lst_ids)
        logging.info(f"Using main device : {dvc_main}")
        logging.info(f"Using devices with IDs : {ids}")

    e_o = config[dflt_cfg['optim_encoder']]
    e_opt = __import__('torch.optim', fromlist=[e_o['optim']])
    if e_o['optim'] == 'Adam':
        optim = getattr(e_opt, e_o['optim'])(
            net.parameters(), 
            lr=float(e_o['lr_encoder']), 
            betas=(float(e_o['beta1']), float(e_o['beta2'])),
            eps=float(e_o['eps']), 
            weight_decay=float(e_o['weight_decay'])
        )
    elif e_o['optim'] == 'SGD':
        optim = getattr(e_opt, e_o['optim'])(
            net.parameters(), 
            lr=float(e_o['lr_encoder']), 
            momentum=float(e_o['momentum']),
            nesterov=e_o.getboolean('nesterov'), 
            weight_decay=float(e_o['weight_decay'])
        )
    e_sc = __import__('torch.optim.lr_scheduler', fromlist=[e_o['optim']])
    scheduler = getattr(e_sc, e_o['scheduler'])(optim, T_0=eval(e_o['T_0']), T_mult=int(e_o['T_mult']))

    if args.load:
        file_ld = os.path.join(dir_ld, f"checkpoints/{args.load[1]}.pth")
        config['TRAIN']['start_epoch'] = args.load[2]

        checkpoint = torch.load(file_ld, map_location='cpu')
        net.load_state_dict(checkpoint['encoder_state_dict'])
        optim.load_state_dict(checkpoint['optim_encoder_state_dict'])

        net.to(device=dvc_main)
        logging.info(f'Model loaded from {file_ld}')
    else:
        init_time = time.strftime("%H%M_%d%b%Y", time.localtime(time.time()))
        config["DEFAULT"]["current_time"] = init_time
        sv_dir = config["SAVE"]["dir_save"]
        try:
            os.mkdir(sv_dir)
            logging.info(f"Created save directory {sv_dir}")
        except OSError:
            pass

        with open(f"{sv_dir}/info_train.ini", 'w') as f:
            f.write('; ')
            [f.write(ag + ' ') for ag in sys.argv]
            f.write('\n')
            config.write(f)

    try:
        nets = [net, ]
        optims = [
            [optim, scheduler],
        ]
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
                }, "INTERRUPTED.pth")
                break
            else:
                break
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
        
