import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision import transforms
from argparse import ArgumentParser
import os


class DataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if not self.hparams.test_resize :
            self.hparams.batch_size_test = 1
        self.hparams.train_dataset_path = os.path.join(self.hparams.dataset_path,'DRUNET')
        self.hparams.test_dataset_path = os.path.join(self.hparams.dataset_path,self.hparams.dataset_name)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(self.hparams.train_patch_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        if self.hparams.test_resize :
            if self.hparams.test_resize_mode == 'center_crop':
                self.val_transform = transforms.Compose([
                    transforms.CenterCrop(self.hparams.test_patch_size),
                    transforms.ToTensor()
                ])
            elif self.hparams.test_resize_mode == 'random_crop':
                self.val_transform = transforms.Compose([
                    transforms.RandomCrop(self.hparams.test_patch_size,pad_if_needed=True),
                    transforms.ToTensor()
                ])
            else :
                self.val_transform = transforms.Compose([
                    transforms.Resize(self.hparams.test_patch_size),
                    transforms.ToTensor(),
                ])

        else :
            self.val_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.dataset_train = datasets.ImageFolder(root = self.hparams.train_dataset_path, transform=self.train_transform)
            self.dataset_val = datasets.ImageFolder(root=self.hparams.test_dataset_path, transform=self.val_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = datasets.ImageFolder(root = self.hparams.test_dataset_path, transform=self.val_transform)
            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.hparams.batch_size_train,
                          shuffle=self.hparams.train_shuffle,
                          num_workers=self.hparams.num_workers_train,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.hparams.batch_size_test,
                          shuffle=False,
                          num_workers=self.hparams.num_workers_test,
                          drop_last=True,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.hparams.batch_size_test,
                          shuffle=False,
                          num_workers=self.hparams.num_workers_test,
                          drop_last=False,
                          pin_memory=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_name', type=str, default='CBSD68')
        parser.add_argument('--dataset_path', type=str, default='../datasets/')
        parser.add_argument('--train_patch_size', type=int, default=128)
        parser.add_argument('--test_patch_size', type=int, default=256)
        parser.add_argument('--train_shuffle', dest='train_shuffle', action='store_true')
        parser.add_argument('--no-train_shuffle', dest='train_shuffle', action='store_false')
        parser.set_defaults(train_shuffle=True)
        parser.add_argument('--no_test_resize', dest='test_resize', action='store_false')
        parser.set_defaults(test_resize=True)
        parser.add_argument('--num_workers_train',type=int, default=32)
        parser.add_argument('--num_workers_test', type=int, default=32)
        parser.add_argument('--batch_size_train', type=int, default=16)
        parser.add_argument('--batch_size_test', type=int, default=8)
        parser.add_argument('--test_resize_mode', type=str, default='center_crop')
        return parser