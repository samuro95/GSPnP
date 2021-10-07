import pytorch_lightning as pl
from lightning_GSDRUNet import GradMatch
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import random
import torch

if __name__ == '__main__':

    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.add_argument('--log_folder', type=str, default='logs')
    parser.set_defaults(save_images=False)

    # MODEL args
    parser = GradMatch.add_model_specific_args(parser)
    # DATA args
    parser = DataModule.add_data_specific_args(parser)
    # OPTIM args
    parser = GradMatch.add_optim_specific_args(parser)

    hparams = parser.parse_args()

    random.seed(0)

    if not os.path.exists(hparams.log_folder):
        os.mkdir(hparams.log_folder)
    log_path = hparams.log_folder + '/' + hparams.name
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_logger = pl_loggers.TensorBoardLogger(log_path)

    model = GradMatch(hparams)
    dm = DataModule(hparams)

    if hparams.start_from_checkpoint:
        checkpoint = torch.load(hparams.pretrained_checkpoint)
        model.load_state_dict(checkpoint['state_dict'],strict=False)

    early_stop_callback = EarlyStopping(
        monitor='val/avg_val_loss',
        min_delta=0.00,
        patience=hparams.early_stopping_patiente,
        verbose=True,
        mode='min'
    )
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    if hparams.resume_from_checkpoint:
        trainer = pl.Trainer.from_argparse_args(hparams, logger=tb_logger, gpus=-1, val_check_interval=hparams.val_check_interval,
                                                resume_from_checkpoint=hparams.pretrained_checkpoint,
                                                gradient_clip_val=hparams.gradient_clip_val, accelerator='ddp',
                                                max_epochs = 1500,
                                                callbacks=[lr_monitor])
    else :
        trainer = pl.Trainer.from_argparse_args(hparams, logger=tb_logger,gpus=-1,val_check_interval=hparams.val_check_interval,
                                                gradient_clip_val=hparams.gradient_clip_val, log_gpu_memory='all', accelerator='ddp',
                                                max_epochs = 1500,
                                                callbacks=[lr_monitor])

    trainer.fit(model, dm)



