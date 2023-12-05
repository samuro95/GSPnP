import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from argparse import ArgumentParser
import os


class DataModule(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        if not self.params.test_resize :
            self.params.batch_size_test = 1
        self.params.train_dataset_path = os.path.join(self.params.dataset_path,'DRUNET')
        self.params.test_dataset_path = os.path.join(self.params.dataset_path,self.params.dataset_name)

        if self.params.grayscale :  
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(self.params.train_patch_size, pad_if_needed=True),
                transforms.functional.rgb_to_grayscale,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else :
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(self.params.train_patch_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])

        if self.params.test_resize :
            if self.params.test_resize_mode == 'center_crop':
                val_transform_list = [
                    transforms.CenterCrop(self.params.test_patch_size),
                ]
            elif self.params.test_resize_mode == 'random_crop':
                val_transform_list = [
                    transforms.RandomCrop(self.params.test_patch_size,pad_if_needed=True),
                ]
            else :
                val_transform_list = [
                    transforms.Resize(self.params.test_patch_size),
                ]

        if self.params.grayscale : 
            val_transform_list.append(transforms.functional.rgb_to_grayscale)

        val_transform_list.append(
            transforms.ToTensor()
        )

        self.val_transform = transforms.Compose(val_transform_list)

        self.dataset_train = datasets.ImageFolder(root = self.params.train_dataset_path, transform=self.train_transform)
        self.dataset_val = datasets.ImageFolder(root=self.params.test_dataset_path, transform=self.val_transform)


    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.params.batch_size_train,
                          shuffle=self.params.train_shuffle,
                          num_workers=self.params.num_workers_train,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.params.batch_size_test,
                          shuffle=False,
                          num_workers=self.params.num_workers_test,
                          drop_last=True,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.params.batch_size_test,
                          shuffle=False,
                          num_workers=self.params.num_workers_test,
                          drop_last=False,
                          pin_memory=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_name', type=str, default='CBSD68')
        parser.add_argument('--dataset_path', type=str, default='../datasets/')
        parser.add_argument('--train_patch_size', type=int, default=128)
        parser.add_argument('--test_patch_size', type=int, default=128)
        parser.add_argument('--train_shuffle', dest='train_shuffle', action='store_true')
        parser.add_argument('--no-train_shuffle', dest='train_shuffle', action='store_false')
        parser.set_defaults(train_shuffle=True)
        parser.add_argument('--no_test_resize', dest='test_resize', action='store_false')
        parser.set_defaults(test_resize=True)
        parser.add_argument('--num_workers_train',type=int, default=20)
        parser.add_argument('--num_workers_test', type=int, default=20)
        parser.add_argument('--batch_size_train', type=int, default=16)
        parser.add_argument('--batch_size_test', type=int, default=8)
        parser.add_argument('--test_resize_mode', type=str, default='center_crop')
        return parser