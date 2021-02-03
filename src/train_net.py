# basics
import os
import logging

# pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# tools
from numpy.lib.arraysetops import isin
from colorama import Fore
from tqdm import tqdm
from tqdm.utils import _term_move_up

# self written libs
import loss
from scores import F1Score
from dataset import SleepMAIC
from transforms import RandomContrast, RandomGamma

class TrainNet():
    # test loss, pnsr, ssim
    _losses = -1
    _scores = -1
    dtype = torch.float

    def __init__(self, nets, optims, device_main, config):
        '''
        Input
        -----
        nets: list of networks
        optims: list of (optimizer, scheduler)
        '''
        self.net = nets[0]
        self.optim = optims[0][0]
        self.scheduler = optims[0][1]
        self.device_main = device_main

        self.cfg_dft = config['DEFAULT']
        self.cfg_train = config['TRAIN']
        self.cfg_model = config['MODEL']
        self.cfg_dt = config['DATASET']
        self.cfg_sv = config['SAVE']

        self.name_loss = self.cfg_train['loss']

        self.dir_sv = self.cfg_sv['dir_save']
        self.num_epoch = int(self.cfg_train['num_epoch'])

        if self.cfg_dt['name'] == 'Sleep':
            lst_train = eval(self.cfg_dt['lst_train'])
            lst_valid = eval(self.cfg_dt['lst_valid'])
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            data_train = SleepMAIC(
                self.cfg_dt['path'], split=lst_train, transform=transform)
            data_valid = SleepMAIC(
                self.cfg_dt['path'], split=lst_valid, transform=transform)
        else:
            raise NotImplemented('There are no such dataset')
                
        self.n_train, self.n_valid = len(data_train), len(data_valid)
        self.loader_train = DataLoader(
            data_train, 
            batch_size=int(self.cfg_train['train_batch_size']), 
            shuffle=True, num_workers=int(self.cfg_train['num_workers']), 
            pin_memory=True
        )
        self.loader_valid = DataLoader(
            data_valid, 
            batch_size=int(self.cfg_train['valid_batch_size']), 
            shuffle=False, num_workers=int(self.cfg_train['num_workers']),
            pin_memory=True
        )
        self.writer_main = SummaryWriter(log_dir=os.path.join(self.dir_sv, f'tb_logs/train'))
        self.writer_test = SummaryWriter(log_dir=os.path.join(self.dir_sv, f'tb_logs/valid'))

        logging.info("Starting training...")
        
        for n, p in self.encoder.named_parameters():
            if not p.requires_grad: logging.info(f"[Parameters] {n} : {p.requires_grad}")

        try:
            self.criterion = getattr(nn, self.name_loss)()
        except AttributeError:
            self.criterion = getattr(loss, self.name_loss)()


    def training(self, epoch):
        pbar = tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.num_epoch}', unit='img',
            bar_format='{l_bar}%s{bar:10}%s{r_bar}{bar:-10b}' % (Fore.RED, Fore.RESET))
        mean_loss, mean_score = 0, 0

        self.net.train()
        n_iter = len(self.loader_train)
        for k, btchs in enumerate(self.loader_train):
            imgs = btchs[0].to(device=self.dvc_main, dtype=self.dtype)
            labels = btchs[1].to(device=self.dvc_main, dtype=self.dtype)

            self.scheduler.step(epoch + k / n_iter)
            self.optim.zero_grad()

            preds = self.net(imgs)
            loss = self.criterion(preds, labels)
            loss.backward()

            self.optim.step()

            with torch.no_grad():
                img_dt = imgs.data
                label_dt = labels.data
                pred_dt = preds.data

                mean_score += F1Score(pred_dt, label_dt)
                mean_loss += loss.item()

                lrs = f"{self.scheduler.get_last_lr()[0]:.3f}"
                pbar.set_postfix(**{self.name_loss: mean_loss / (k + 1),
                                    'F1Score': mean_score / (k + 1),
                                    'LRs' : lrs})
                pbar.update(imgs.shape[0])
                if k == 0:
                    img_dict = {'Train/': img_dt,
                                'Train/true': label_dt,
                                'Train/pred': pred_dt}
                    self.writing(epoch, self.writer_main, img_dict, opt='image')
        
        scalar_dict = {self.name_loss: mean_loss / (n_iter + 1),
                        'F1Score': mean_score / (n_iter + 1)}
        pbar.write(_term_move_up(), end='\r')
        self.writing(epoch, self.writer_main, scalar_dict, opt='scalar')
        pbar.close()
    
    def validation(self, epoch):
        pbar = tqdm(total=self.n_valid, desc=f'Validation', unit='img', leave=True,
            bar_format='{l_bar}%s{bar:10}%s{r_bar}{bar:-10b}' % (Fore.BLUE, Fore.RESET))
        mean_loss, mean_score = 0, [0, 0]

        self.net.eval()
        n_iter = len(self.loader_valid)
        with torch.no_grad():
            for k, btchs in enumerate(self.loader_valid):
                imgs = btchs[0].to(device=self.dvc_main, dtype=self.dtype)
                labels = btchs[1].to(device=self.dvc_main, dtype=self.dtype)

                preds = self.net(imgs)
                loss = self.criterion(preds, labels)
            
                img_dt = imgs.data
                label_dt = labels.data
                pred_dt = preds.data

                mean_score += F1Score(pred_dt, label_dt)
                mean_loss += loss.item()

                pbar.set_postfix(**{self.name_loss: mean_loss / (k + 1),
                                    'F1Score': mean_score / (k + 1)})
                pbar.update(imgs.shape[0])

                if k == 0:
                    init_dict = {f'Test{k}/': img_dt,
                                f'Test{k}/true': label_dt}
                    self.writing(epoch, self.writer_test, init_dict, opt='image')
                img_dict = {f'Test{k}/pred': pred_dt}
                self.writing(epoch, self.writer_test, img_dict, opt='image')
        
        self.scalar_dict = {self.name_loss: mean_loss / (n_iter + 1),
                            'F1Score': mean_score / (n_iter + 1)}
        pbar.close()
        self.writing(epoch, self.writer_test, self.scalar_dict, opt='scalar')

    def saving(self, epoch):
        dir_cp = os.path.join(self.dir_sv, 'checkpoints/')
        term_sv = int(self.cfg_sv['term_save'])
        try:
            os.mkdir(dir_cp)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        # if we are in save term or validation score hit a new peak.
        _indic = {
            epoch: (epoch % term_sv == 0) or (epoch == self.num_epoch - 1),
            'F1_Score': self._scores < self.scalar_dict['F1Score']
        }
        if any(_indic.values()):
            torch.save({
                    'net_state_dict': self.net.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    }, os.path.join(dir_cp, 'temp'))
            logging.info(f'''Checkpoint {epoch} saved !!
                    self.name_loss\t: {self.scalar_dict[self.name_loss]}
                    F1Score\t: {self.scalar_dict['F1Score']}''')

            for k, v in _indic.items():
                if v:
                    name_sv = os.path.join(dir_cp, f"CP_{k}.pth")
                    os.system(f"cp {os.path.join(dir_cp,'temp')} {name_sv}")
        self._scores = max(self._scores, self.scalar_dict['F1Score'])
                
    @staticmethod
    def writing(epoch, writer, tb_dict, opt):
        for key, val in tb_dict.items():
            if opt == 'image':
                writer.add_images(key, val, epoch)
            elif opt == 'scalar':
                writer.add_scalar(key, val, epoch)
