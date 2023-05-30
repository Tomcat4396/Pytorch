import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as torch_models
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import MultiLabelDataset, WaterLevelDataset, PipeShapeDataset, PipeMaterialDataset


TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
MODEL_NAMES =  TORCHVISION_MODEL_NAMES


def evaluate(dataloader, model, act_func, device):
    model.eval()

    classPredictions = None
    imgPathsList = []

    dataLen = len(dataloader)
    print(dataLen)
    
    with torch.no_grad():
        for i, (images, _, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            output = model(images)

            classOutput = act_func(output).detach().cpu().numpy()
            
            if classPredictions is None:
                classPredictions = classOutput
            else:
                classPredictions = np.vstack((classPredictions, classOutput))

            imgPathsList.extend(list(imgPaths))
    return classPredictions, imgPathsList



def load_model(model_path, best_weights=False):

    if best_weights:
        if not os.path.isfile(model_path):
            raise ValueError("The provided path does not lead to a valid file: {}".format(model_path))
        last_ckpt_path = model_path
    else:
        last_ckpt_path = os.path.join(model_path, "last.ckpt")
        if not os.path.isfile(last_ckpt_path):
            raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(last_ckpt_path)
    
    hparams = model_last_ckpt["hyper_parameters"]
    model_name = hparams["model"]
    num_classes = hparams["num_classes"]
    training_task = hparams["training_task"]
    
    # Load best checkpoint
    if best_weights:
        best_model = model_last_ckpt
    else:
        best_model_path = model_last_ckpt["callbacks"][ModelCheckpoint]["best_model_path"]
        best_model = torch.load(best_model_path)

    best_model_state_dict = best_model["state_dict"]

    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v


    return updated_state_dict, model_name, num_classes, training_task


def STL_inference(args):

    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    split = args["split"]
    best_weights = args["best_weights"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict, model_name, num_classes, training_task = load_model(model_path, best_weights)

    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    # Init model
    if model_name in TORCHVISION_MODEL_NAMES:
        model = torch_models.__dict__[model_name](num_classes = num_classes)
    else:
        raise ValueError("Got model {}, but no such model is in this codebase".format(model_name))

    # Load best checkpoint
    model.load_state_dict(updated_state_dict)
    
    # initialize dataloaders
    img_size = 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

    
    if training_task == "defects":
        dataset = MultiLabelDataset(ann_root, data_root, split=split, transform=eval_transform, onlyDefects=False)
        act_func = nn.Sigmoid()
    elif training_task == "water":
        dataset = WaterLevelDataset(ann_root, data_root, split=split, transform=eval_transform)
        act_func = nn.Softmax(dim=-1)
    elif training_task == "shape":
        dataset = PipeShapeDataset(ann_root, data_root, split=split, transform=eval_transform)
        act_func = nn.Softmax(dim=-1)
    elif training_task == "material":
        dataset = PipeMaterialDataset(ann_root, data_root, split=split, transform=eval_transform)
        act_func = nn.Softmax(dim=-1)


    dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)

    labelNames = dataset.LabelNames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    #  Predict results
    sigmoid_predictions, imgPaths = evaluate(dataloader, model, act_func, device)

    sigmoid_dict = {}
    sigmoid_dict["Filename"] = imgPaths
    for idx, header in enumerate(labelNames):
        sigmoid_dict[header] = sigmoid_predictions[:,idx]

    sigmoid_df = pd.DataFrame(sigmoid_dict)
    sigmoid_df.to_csv(os.path.join(outputPath, "{}_{}_{}_sigmoid.csv".format(model_version, training_task, split.lower())), sep=",", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    parser.add_argument("--results_output", type=str, default = "./results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])



    args = vars(parser.parse_args())

    STL_inference(args)