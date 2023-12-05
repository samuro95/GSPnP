import pytorch_lightning as pl
from lightning_GSDRUNet import GradMatch
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import random
import torch

if __name__ == '__main__':

    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--log_folder', type=str, default='logs')
    

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

    model.train_dataloader = dm.train_dataloader
    model.val_dataloader = dm.val_dataloader

    if hparams.start_from_checkpoint:
        checkpoint = torch.load(hparams.pretrained_checkpoint)
        model.load_state_dict(checkpoint['state_dict'],strict=False)

    checkpoint_callback = ModelCheckpoint(dirpath=f"ckpts/{hparams.name}/", # where the ckpt will be saved
                                      save_top_k=5,
                                      monitor='val/avg_psnr', # ckpt will be save according to the validation loss that you need to calculate on the validation step when you train your model
                                      mode="max" # validation loos need to be min
                                      ) 


    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    max_epochs = 1200
    
    if hparams.resume_from_checkpoint:
        trainer = pl.Trainer(logger=tb_logger, gpus=-1, val_check_interval=hparams.val_check_interval,
                                resume_from_checkpoint=hparams.pretrained_checkpoint,
                                gradient_clip_val=hparams.gradient_clip_val,
                                max_epochs=max_epochs, precision=32, 
                                callbacks=[lr_monitor, checkpoint_callback],accelerator="gpu", strategy="ddp")
    else :
        trainer = pl.Trainer(logger=tb_logger,gpus=-1,val_check_interval=hparams.val_check_interval,
                                gradient_clip_val=hparams.gradient_clip_val,
                                max_epochs=max_epochs, precision=32, 
                                callbacks=[lr_monitor, checkpoint_callback], accelerator="gpu", strategy="ddp")
    
    trainer.fit(model)




