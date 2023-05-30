import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchvision import models as torch_models
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiLabelDataModule, WaterLevelDataModule, PipeShapeDataModule, PipeMaterialDataModule

from class_weight import positive_ratio, inverse_frequency, effective_samples, identity_weight, defect_CIW

import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


class STL_Model(pl.LightningModule):

    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
    MODEL_NAMES =  TORCHVISION_MODEL_NAMES
    

    def __init__(self, model = "resnet18", num_classes=2, learning_rate=1e-2, momentum=0.9, weight_decay = 0.0001, criterion_weight=None, **kwargs):
        super(STL_Model, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        if model in STL_Model.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes = self.num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(model))
            
        if kwargs["training_task"] == "defects":
            self.criterion = torch.nn.BCEWithLogitsLoss(weight = criterion_weight[0], pos_weight = criterion_weight[1])         
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight = criterion_weight[0])   


    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def train_function(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return loss


    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay = self.hparams.weight_decay)

        if self.hparams.lr_schedule == "Step":
            if self.hparams.schedule_int == "epoch":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.hparams.learning_rate_steps, gamma=self.hparams.learning_rate_gamma)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[x*len(self.train_dataloader()) for x in self.hparams.learning_rate_steps], gamma=self.hparams.learning_rate_gamma)
        elif self.hparams.lr_schedule == "Cosine":
            if self.hparams.schedule_int == "epoch":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs, eta_min = self.hparams.learning_rate*0.01)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs*len(self.train_dataloader()), eta_min = self.hparams.learning_rate*0.01)

        scheduler = {"scheduler": scheduler,
                    "interval": self.hparams.schedule_int,
                    "frequency": 1}

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=128, help="Size of the batch per GPU")
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--learning_rate_gamma', type=float, default=0.01)
        parser.add_argument('--learning_rate_steps', nargs='+', type=int, default=[20, 30])
        parser.add_argument('--lr_schedule', type=str, default="Step", choices=["Step", "Cosine"])
        parser.add_argument('--schedule_int', type=str, default="epoch", choices=["epoch", "step"])
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

        parser.add_argument('--model', type=str, default="resnet101", choices=STL_Model.MODEL_NAMES)

        parser.add_argument('--training_task', type=str, default="defects", choices=["defects", "water", "shape", "material"])
        parser.add_argument('--class_weight', type=str, default="None", choices=["None", "Inverse", "Positive", "Effective"])
        parser.add_argument('--defect_weights', type=str, default="", choices=["", "CIW", "PosCIW", "Both"])
        parser.add_argument('--effective_beta', type=float, default=0.9999)
        return parser




def main(args):
    args.seed = pl.seed_everything(args.seed)

    # Init data with transforms
    img_size = 224

    train_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    if args.training_task == "defects":
        dm = MultiLabelDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform, only_defects=False)
    elif args.training_task == "water":
        dm = WaterLevelDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform)
    elif args.training_task == "shape":
        dm = PipeShapeDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform)
    elif args.training_task == "material":
        dm = PipeMaterialDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform)
        
    dm.prepare_data()
    dm.setup("fit")

    if args.class_weight == "None":
        weights = identity_weight(dm.train_dataset.labels, dm.num_classes)

    elif args.class_weight == "Positive":
        weights = positive_ratio(dm.train_dataset.labels, dm.num_classes)

    elif args.class_weight == "Inverse":
        weights = inverse_frequency(dm.train_dataset.labels, dm.num_classes)

    elif args.class_weight == "Effective":
        assert args.effective_beta < 1.0 and args.effective_beta >= 0.0, "The effective sampling beta need to be in the range [0,1) and not: {}".format(args.effective_beta)
        weights = effective_samples(dm.train_dataset.labels, dm.num_classes, args.effective_beta)
        
    weights = [weights, None]
    if args.defect_weights != "":
        ciw_weights = defect_CIW(dm.train_dataset.defect_LabelNames)

        if args.defect_weights == "Both":
            weights[1] = ciw_weights
        elif args.defect_weights == "CIW":
            weights[0] = ciw_weights
            weights[1] = None
        elif args.defect_weights == "PosCIW":
            weights[0] = None
            weights[1] = ciw_weights

    # Init our model
    light_model = STL_Model(num_classes=dm.num_classes, criterion_weight= weights, **vars(args))

    # train
    prefix = "{}-{}-STL-".format(args.training_task, args.class_weight)
    print("-"*15 + prefix + "-"*15)

    if not os.path.isdir(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    logger = TensorBoardLogger(save_dir=args.log_save_dir, name=args.model, version=prefix + "version_" + str(args.log_version))

    logger_path = os.path.join(args.log_save_dir, args.model, prefix + "version_" + str(args.log_version))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        save_last = True,
        verbose=False,
        monitor="val_loss",
        mode='min',
        period=1,
        prefix=''
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if args.use_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True


    trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan = True, benchmark = benchmark,
                                            deterministic=deterministic, max_epochs=args.max_epochs,
                                            logger=logger, callbacks=[checkpoint_callback,lr_monitor]
                                            )


    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))

def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--use_deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=1234567890)


    # add TRAINER level args
    parser = pl.Trainer.add_argparse_args(parser)

    # add MODEL level args
    parser = STL_Model.add_model_specific_args(parser)
    args = parser.parse_args()

    # args.workers = args.gpus*1
    args.workers = 0

    main(args)

if __name__ == "__main__":
    run_cli()