'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from datamodules import EPDataModule
from predictors import EPDiffuser
from tqdm.auto import tqdm
import datetime
import os
from utils import copy_files



if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('lightning_logs', 'EP_Diffuser', current_time) 
    copy_files(log_dir)
    
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='data/argo2/')
    parser.add_argument('--dataset', type=str, default='argoverse2')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default='data/argo2/train/processed/')
    parser.add_argument('--val_processed_dir', type=str, default='data/argo2/val/processed/')
    parser.add_argument('--test_processed_dir', type=str, default='data/argo2/test/processed/')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    EPDiffuser.add_model_specific_args(parser)
    args = parser.parse_args()

    model = EPDiffuser(**vars(args))
    
    datamodule = EPDataModule(**vars(args))
    
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE_1', save_top_k=5, save_last = True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir=log_dir, name='csv')
    tb_logger = TensorBoardLogger(save_dir=log_dir, name='tb')
    trainer = pl.Trainer(default_root_dir = log_dir,
                         accelerator=args.accelerator, devices=args.devices,
                         strategy="auto",
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                         logger=[tb_logger, csv_logger])
    trainer.fit(model, datamodule)
        
    