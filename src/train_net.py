import os
import logging
from numpy.lib.arraysetops import isin

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import numpy as np
from colorama import Fore
from tqdm import tqdm
from tqdm.utils import _term_move_up

from .dataset import SleepMAIC
from .loss import F1Loss
from .transforms import RandomContrast, RandomGamma


class TrainNet():
    # test loss, pnsr, ssim
    scores = -1, -1, -1
    b_scores = -1, -1
    dtype = torch.float

    def __init__(self, nets, optims, device_main, config):
        self.encoder = nets[0]
        self.decoder = nets[1]
        
        self.optim_encoder = optims[0][0]
        self.optim_decoder = optims[1][0]
        self.scheduler_encoder = optims[0][1]
        self.scheduler_decoder = optims[1][1]

        self.dvc_main = device_main
        self.config = config
        self.dir_sv = config['SAVE']['dir_save']

        self.num_epoch = int(config['TRAIN']['num_epoch'])

        # dir_train = config['DATASET']['dir_train']
        # dir_valin = config['DATASET']['dir_valid']
        
        if config['DATASET']['data'] == 'sample':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            data_train = datasets.MNIST(
                config['DATASET']['root_dataset'],
                train=True, download=True, transform=transform)
            data_valid = datasets.MNIST(
                config['DATASET']['root_dataset'],
                train=False, download=True, transform=transform)
        elif config['DATASET']['data'] == 'teeth_cropped_ext':
            sz = (int(config['MODEL']['H']), int(config['MODEL']['W']))
            transform_train = transforms.Compose([
                RandomGamma(p=.75),
            ])
            transform_target = None
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=.5),
                transforms.Resize(sz),
                transforms.ToTensor(),
            ])
            data_train = TeethCroppedExt(
                config['DATASET']['root_dataset'], 
                split='train', transform=transform,
                transform_train=transform_train)
            data_valid = TeethCroppedExt(
                config['DATASET']['root_dataset'], 
                split='test', transform=transform,
                transform_train=transform_train)

        self.n_train, self.n_valid = len(data_train), len(data_valid)
        self.loader_train = DataLoader(
            data_train, 
            batch_size=int(config['TRAIN']['train_batch_size']), 
            shuffle=True, num_workers=int(config['TRAIN']['num_workers']), 
            pin_memory=True
        )
        self.loader_valid = DataLoader(
            data_valid, 
            batch_size=int(config['TRAIN']['valid_batch_size']), 
            shuffle=False, num_workers=int(config['TRAIN']['num_workers']),
            pin_memory=True
        )
        self.writer_main = SummaryWriter(log_dir=os.path.join(self.dir_sv, f'tb_logs/main'))
        self.writer_test = SummaryWriter(log_dir=os.path.join(self.dir_sv, f'tb_logs/test'))

        logging.info("Starting training...")
        
        for n, p in self.encoder.named_parameters():
            if not p.requires_grad: logging.info(f"[Encoder] {n} : {p.requires_grad}")
        for n, p in self.decoder.named_parameters():
            if not p.requires_grad: logging.info(f"[Decoder] {n} : {p.requires_grad}")

        try:
            _ct = __import__('torch.nn', fromlist=[config['TRAIN']['loss']])
            self.criterion = getattr(_ct, config['TRAIN']['loss'])()
        except AttributeError:
            _ct = __import__('src.loss', fromlist=[config['TRAIN']['loss']])
            self.criterion = getattr(_ct, config['TRAIN']['loss'])()


    def training(self, epoch):
        pbar = tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.num_epoch}', unit='img',
            bar_format='{l_bar}%s{bar:10}%s{r_bar}{bar:-10b}' % (Fore.RED, Fore.RESET))
        mean_loss, mean_score = 0, [0, 0]

        self.encoder.train()
        self.decoder.train()
        for k, BCH in enumerate(self.loader_train):
            imgs = BCH[0].to(device=self.dvc_main, dtype=self.dtype)
            labels = BCH[0].to(device=self.dvc_main, dtype=self.dtype)

            self.scheduler_encoder.step(epoch + k / len(self.loader_train))
            self.scheduler_decoder.step(epoch + k / len(self.loader_train))

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()

            if 'VAELoss' in self.config['TRAIN']['loss']:
                l, mu, sig = self.encoder(imgs)
                output = self.decoder(l)
                loss = self.criterion(output, labels, mu, sig)
            else:
                l = self.encoder(imgs)
                output = self.decoder(l)
                loss = self.criterion(output, labels)
            loss.backward()

            self.optim_encoder.step()
            self.optim_decoder.step()

            with torch.no_grad():
                img_dt = imgs.data
                label_dt = labels.data
                output_dt = output.data
                labels_sum = img_dt - label_dt
                output_sum = img_dt - output_dt
                mean_score[0] += PSNR(labels, output).item()
                mean_score[1] += SSIM(labels, output)[0].item()
                mean_loss += loss.item()

                lrs = f"{self.scheduler_encoder.get_last_lr()[0]:.3f}, {self.scheduler_decoder.get_last_lr()[0]:.3f}"
                pbar.set_postfix(**{'Loss': mean_loss / (k + 1),
                                    'PSNR': mean_score[0] / (k + 1),
                                    'SSIM': mean_score[1] / (k + 1),
                                    'LRs' : lrs})
                pbar.update(imgs.shape[0])
                if k == 0:
                    img_dict = {'Train/': img_dt,
                                'Train/true': label_dt,
                                'Train/true_sum': labels_sum,
                                'Train/pred': output_dt,
                                'Train/pred_sum': output_sum}
                    self.writing(epoch, self.writer_main, img_dict, opt='image')
                    
        scalar_dict = {'Loss': mean_loss / (k + 1),
                        'Score/PSNR': mean_score[0] / (k + 1),
                        'Score/SSIM': mean_score[1] / (k + 1)}
        pbar.write(_term_move_up(), end='\r')
        self.writing(epoch, self.writer_main, scalar_dict, opt='scalar')
        pbar.close()
    
    def validation(self, epoch):
        pbar = tqdm(total=self.n_valid, desc=f'Validation', unit='img', leave=True,
            bar_format='{l_bar}%s{bar:10}%s{r_bar}{bar:-10b}' % (Fore.BLUE, Fore.RESET))
        mean_loss, mean_score = 0, [0, 0]

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for k, BCH in enumerate(self.loader_valid):
                imgs = BCH[0].to(device=self.dvc_main, dtype=self.dtype)
                labels = BCH[0].to(device=self.dvc_main, dtype=self.dtype)

                if 'VAELoss' in self.config['TRAIN']['loss']:
                    l, mu, sig = self.encoder(imgs)
                    output = self.decoder(l)
                    loss = self.criterion(output, labels, mu, sig)
                else:
                    l = self.encoder(imgs)
                    output = self.decoder(l)
                    loss = self.criterion(output, labels)
            
                img_dt = imgs.data
                label_dt = labels.data
                output_dt = output.data

                labels_sum = img_dt - label_dt
                output_sum = img_dt - output_dt
                mean_score[0] += PSNR(label_dt, output_dt).item()
                mean_score[1] += SSIM(label_dt, output_dt)[0].item()
                mean_loss += loss.item()

                pbar.set_postfix(**{'Loss': mean_loss / (k + 1),
                                    'PSNR': mean_score[0] / (k + 1),
                                    'SSIM': mean_score[1] / (k + 1)})
                pbar.update(imgs.shape[0])

                if k == 0:
                    # if epoch == 0:
                    init_dict = {f'Test{k}/': img_dt,
                                f'Test{k}/true': label_dt,
                                f'Test{k}/true_sum': labels_sum}
                    self.writing(epoch, self.writer_test, init_dict, opt='image')
                    img_dict = {f'Test{k}/pred': output_dt,
                                f'Test{k}/pred_sum': output_sum}
                    self.writing(epoch, self.writer_test, img_dict, opt='image')
        scalar_dict = {'Loss': mean_loss / (k + 1),
                        'Score/PSNR': mean_score[0] / (k + 1),
                        'Score/SSIM': mean_score[1] / (k + 1)}
        pbar.close()
        self.writing(epoch, self.writer_test, scalar_dict, opt='scalar')
        self.scores = (mean_loss / (k + 1), mean_score[0] / (k + 1), mean_score[1] / (k + 1))

    def saving(self, epoch):
        dir_sv = self.config['SAVE']['dir_save']
        dir_cp = os.path.join(dir_sv, 'checkpoints/')
        term_sv = int(self.config['SAVE']['term_save'])
        try:
            os.mkdir(dir_cp)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        _indic = {
            epoch: (epoch % term_sv == 0) or (epoch == self.num_epoch - 1),
            'SSIM': self.b_scores[0] < self.scores[1],
            'PSNR': self.b_scores[1] < self.scores[2]
        }
        if any(_indic.values()):
            torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optim_encoder_state_dict': self.optim_encoder.state_dict(),
                    'optim_decoder_state_dict': self.optim_decoder.state_dict(),
                    }, os.path.join(dir_cp,'temp'))
            logging.info(f'''Checkpoint {epoch} saved !!
                    {self.config['TRAIN']['loss']}\t\t: {self.scores[0]}
                    PSNR        : {self.scores[1]}
                    SSIM        : {self.scores[2]}
            ''')
            for k, v in _indic.items():
                if v:
                    name_sv = os.path.join(dir_cp, f"CP_{k}.pth")
                    os.system(f"cp {os.path.join(dir_cp,'temp')} {name_sv}")
        self.b_scores = max(self.b_scores[0], self.scores[1]), max(self.b_scores[1], self.scores[2])
                
    @staticmethod
    def writing(epoch, writer, tb_dict, opt):
        for key, val in tb_dict.items():
            if opt == 'image':
                writer.add_images(key, val, epoch)
            elif opt == 'scalar':
                writer.add_scalar(key, val, epoch)
