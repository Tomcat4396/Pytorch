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

from dataloader import MultiTaskDataset
import models.encoders as encoder_models
import models.decoders as decoder_models


TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
MODEL_NAMES =  TORCHVISION_MODEL_NAMES

ENCODER_NAMES = sorted(name for name in encoder_models.__dict__ if not name.startswith("__") and callable(encoder_models.__dict__[name]))
DECODER_NAMES = sorted(name for name in decoder_models.__dict__ if not name.startswith("__") and callable(decoder_models.__dict__[name]))



def evaluate(dataloader, encoder, decoder, act_funcs, device):
    encoder.eval()
    decoder.eval()

    predictions = [None for _ in range(len(act_funcs))]
    first = True

    imgPathsList = []

    dataLen = len(dataloader)
    print(dataLen)
    
    with torch.no_grad():
        for i, (images, _, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            feat_vecs = encoder(images)
            outputs, _ = decoder(feat_vecs)

            classOutput = [act_funcs[idx](outputs[idx]).detach().cpu().numpy() for idx in range(len(act_funcs))]

            if first:
                predictions = [classOutput[idx] for idx in range(len(act_funcs))]
                first = False	
            else:
                predictions = [np.vstack((predictions[idx], classOutput[idx])) for idx in range(len(act_funcs))]

            imgPathsList.extend(list(imgPaths))
    return predictions, imgPathsList


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
    backbone_hparam = hparams["backbone"]
    encoder_hparam = hparams["encoder"]
    decoder_hparam = hparams["decoder"]

    task_classes = hparams["task_classes"]
    valid_tasks = hparams["valid_tasks"]
    
    # Load best checkpoint
    if best_weights:
        best_model = model_last_ckpt
    else:
        best_model_path = model_last_ckpt["callbacks"][ModelCheckpoint]["best_model_path"]
        best_model = torch.load(best_model_path)

    model_name = os.path.splitext(best_model_path)[0]

    best_model_state_dict = best_model["state_dict"]

    updated_encoder_state_dict = OrderedDict()
    updated_decoder_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        if "encoder" in k:
            name = k.replace("encoder.", "")
            updated_encoder_state_dict[name] = v
        if "decoder" in k:
            name = k.replace("decoder.", "")
            updated_decoder_state_dict[name] = v
    
    return updated_encoder_state_dict, updated_decoder_state_dict, model_name, hparams, backbone_hparam, encoder_hparam, decoder_hparam, task_classes, valid_tasks

def MTL_inference(args):

    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    split = args["split"]
    best_weights = args["best_weights"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    updated_encoder_state_dict, updated_decoder_state_dict, model_name, hparams, backbone_hparam, encoder_hparam, decoder_hparam, task_classes, valid_tasks = load_model(model_path, best_weights)


    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    # Init model
    if backbone_hparam in TORCHVISION_MODEL_NAMES:
        if backbone_hparam == "resnet18" or backbone_hparam == "resnet34":
            input_channels = 512
        else:
            input_channels = 2048

        backbone = torch_models.__dict__[backbone_hparam]
    else:
        raise ValueError("Got backbone {}, but no such backbone is in this codebase".format(backbone_hparam))

    if encoder_hparam in ENCODER_NAMES:
        encoder = encoder_models.__dict__[encoder_hparam](backbone = backbone, n_tasks = len(task_classes))
    else:
        raise ValueError("Got encoder {}, but no such encoder is in this codebase".format(encoder_hparam))
    
    
    if encoder_hparam == "MTAN":
        list_input = True
    else:
        list_input = False

    if decoder_hparam in DECODER_NAMES:
        if decoder_hparam == "SimpleHead":
            decoder = decoder_models.SimpleHead(n_tasks = len(task_classes), num_classes =  task_classes, input_channels = input_channels, decoder_channels = hparams["decoder_channels"], pool= hparams["decoder_pool"], list_input = list_input)
        elif decoder_hparam == "CTGNN":            
            decoder = decoder_models.CTGNN(n_tasks = len(task_classes), num_classes =  task_classes, input_channels = input_channels, decoder_channels = hparams["decoder_channels"], adj_mat_path = hparams["adj_mat_path"], gnn_head = hparams["gnn_head"], gnn_layers=hparams["gnn_layers"], gnn_channels = hparams["gnn_channels"], gnn_dropout=hparams["gnn_dropout"], gnn_residual=hparams["gnn_residual"], attention_heads = hparams["gat_num_heads"], bottleneck_channels = hparams["bottleneck_channels"], adj_normalization=hparams["adj_normalization"], pool= hparams["decoder_pool"], list_input = list_input, shared_bottleneck = hparams["shared_bottleneck"], shared_linear = hparams["shared_linear"], gnn_residual_act = hparams["gnn_residual_act"])
    else:
        raise ValueError("Got decoder {}, but no such decoder is in this codebase".format(decoder_hparam))


    # Load best checkpoint   
    encoder.load_state_dict(updated_encoder_state_dict)
    decoder.load_state_dict(updated_decoder_state_dict)
    
    # initialize dataloaders
    img_size = 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

    
    dataset = MultiTaskDataset(ann_root, data_root, split=split, transform=eval_transform)

    act_funcs = []
    if "defect" in valid_tasks:
        act_funcs.append(nn.Sigmoid())
    if "water" in valid_tasks:
        act_funcs.append(nn.Softmax(dim=-1))    
    if "shape" in valid_tasks:
        act_funcs.append(nn.Softmax(dim=-1))
    if "material" in valid_tasks:
        act_funcs.append(nn.Softmax(dim=-1))


    dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)

    defectLabelNames = dataset.defect_LabelNames
    waterLabelNames = dataset.water_LabelNames
    shapeLabelNames = dataset.shape_LabelNames
    materialLabelNames = dataset.material_LabelNames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Predict results
    sigmoids, imgPaths = evaluate(dataloader, encoder, decoder, act_funcs, device)

    task_idx = 0

    if "defect" in valid_tasks:
        sigmoid_defect_dict = {}
        sigmoid_defect_dict["Filename"] = imgPaths

        for idx, header in enumerate(defectLabelNames):
            sigmoid_defect_dict[header] = sigmoids[task_idx][:,idx]

        sigmoid_defect_df = pd.DataFrame(sigmoid_defect_dict)
        sigmoid_defect_df.to_csv(os.path.join(outputPath, "{}_defect_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)
        
        task_idx += 1


    if "water" in valid_tasks:
        sigmoid_water_dict = {}
        sigmoid_water_dict["Filename"] = imgPaths

        for idx, header in enumerate(waterLabelNames):
            sigmoid_water_dict[header] = sigmoids[task_idx][:,idx]

        sigmoid_water_dict = pd.DataFrame(sigmoid_water_dict)
        sigmoid_water_dict.to_csv(os.path.join(outputPath, "{}_water_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)
        
        task_idx += 1


    if "shape" in valid_tasks:
        sigmoid_shape_dict = {}
        sigmoid_shape_dict["Filename"] = imgPaths

        for idx, header in enumerate(shapeLabelNames):
            sigmoid_shape_dict[header] = sigmoids[task_idx][:,idx]
            
        sigmoid_shape_dict = pd.DataFrame(sigmoid_shape_dict)
        sigmoid_shape_dict.to_csv(os.path.join(outputPath, "{}_shape_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)
        
        task_idx += 1


    if "material" in valid_tasks:
        sigmoid_material_dict = {}
        sigmoid_material_dict["Filename"] = imgPaths
            
        for idx, header in enumerate(materialLabelNames):
            sigmoid_material_dict[header] = sigmoids[task_idx][:,idx]

        sigmoid_material_dict = pd.DataFrame(sigmoid_material_dict)
        sigmoid_material_dict.to_csv(os.path.join(outputPath, "{}_material_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)
        
        task_idx += 1





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

    MTL_inference(args)
