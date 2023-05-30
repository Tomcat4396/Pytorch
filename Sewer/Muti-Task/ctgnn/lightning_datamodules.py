import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloader import MultiLabelDataset, WaterLevelDataset, PipeShapeDataset, PipeMaterialDataset, MultiTaskDataset



class MultiTaskDataModule(pl.LightningModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="./Data", train_transform = None, eval_transform = None, only_defects=False):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, onlyDefects=self.only_defects)
            self.val_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform, onlyDefects=self.only_defects)
        if stage == 'test':
            self.test_dataset = MultiTaskDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, onlyDefects=self.only_defects)

        self.defect_num_classes = self.train_dataset.defect_num_classes
        self.water_num_classes = self.train_dataset.water_num_classes
        self.shape_num_classes = self.train_dataset.shape_num_classes
        self.material_num_classes = self.train_dataset.material_num_classes

        self.defect_LabelNames = self.train_dataset.defect_LabelNames
        self.water_LabelNames = self.train_dataset.water_LabelNames
        self.shape_LabelNames = self.train_dataset.shape_LabelNames
        self.material_LabelNames = self.train_dataset.material_LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class MultiLabelDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="./Data", train_transform = None, eval_transform = None, only_defects=False):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, onlyDefects=self.only_defects)
            self.val_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform, onlyDefects=self.only_defects)
        if stage == 'test':
            self.test_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, onlyDefects=self.only_defects)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class WaterLevelDataModule(pl.LightningModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="./Data", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = WaterLevelDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class PipeShapeDataModule(pl.LightningModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="./Data", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = PipeShapeDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = PipeShapeDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = PipeShapeDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class PipeMaterialDataModule(pl.LightningModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations", data_root="./Data", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = PipeMaterialDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = PipeMaterialDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = PipeMaterialDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl