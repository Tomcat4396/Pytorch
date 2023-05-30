# Multi-Task Classification of Sewer Pipe Defects and Properties using a Cross-Task Graph Neural Network Decoder

This repository is the official implementation of [Multi-Task Classification of Sewer Pipe Defects and Properties using a Cross-Task Graph Neural Network Decoder](https://arxiv.org/abs/2111.07846). 


The CT-GNN project page can be found [here](http://vap.aau.dk/ctgnn).

## Requirements

The main packages needed are listed below:

- Pytorch 1.6.0
- Torchvision 0.7.0
- Pytorch-Lightning 1.2.8
- Pandas 1.2.4
- Numpy 1.19.2
- Scikit-Learn 0.24.1


The Sewer-ML dataset can be accessed after filling out this [Google Form](https://forms.gle/hBaPtoweZumZAi4u9).
The Sewer-ML dataset is licensed under the Creative Commons BY-NC-SA 4.0 license.


## Training

The models can be trained in two variations: Single Task Learning (STL) and Multi-Task Learning (MTL). Each variation has their own trainer and specific hyperparameters.


When training the images are normalized with the following mean and standard deviation, found using the calculate_normalization.py script.:
- mean = [0.523, 0.453, 0.345]
- std = [0.210, 0.199, 0.154]

Two examples of this would be training a STL network for the defect task and a MTL CT-GNN network with GAT. For the CTGNN the adjacency matrix can be constructed using the construct_adjacency_matrix.py script in the models folder

```
python STL_Trainer.py --precision 16 --batch_size 128 --max_epochs 40 --gpus 2 --accelerator ddp --model renet50  --training_task defect --class_weight Effective --effective_beta 0.9999 --progress_bar_refresh_rate 500 --flush_logs_every_n_steps 1000 --log_every_n_steps 100 --ann_root <path_to_annotations> --data_root <path_to_data> --log_save_dir <path_to_model_logs>
```

```
python MTL_Trainer.py --precision 16 --batch_size 128 --max_epochs 40 --gpus 2 --accelerator ddp --backbone resnet50 --encoder ResNetBackbone --decoder CTGNN --gnn_head GAT --gnn_layers 1 --gnn_channels 128 --gnn_dropout 0.0 --gnn_residual --bottleneck_channels 32 --gat_num_heads 8 --adj_mat_path ./adj_all_65/adj_binary.npy --adj_normalization Sym  --class_weight Effective --effective_beta 0.9999  --f2CIW_weights PosWeight --valid_tasks defect water shape material --task_weight Fixed --task_weights_fixed 27 1 1 1 --use_auxilliary --main_weight 0.75 --progress_bar_refresh_rate 500  --flush_logs_every_n_steps 1000 --log_every_n_steps 100 --ann_root <path_to_annotations> --data_root <path_to_data> --log_save_dir <path_to_model_logs>
```

## Evaluation

To evaluate a set of models on the validation set of the Sewer-ML dataset, first the raw predictions for each task in each image should be generated, which is subsequently compared to the ground truth. The raw predictions should be probabilities.

When the predictions have been obtained the performance of the model can be determined using the calculate_results.py script.

```
python calculate_results.py --output_path <path_to_metric_results> --split <dataset_split_to_use> --score_path <path_to_predictions> --gt_path <path_to_annotations>
```

The validation prediction of the classifiers when trained using the trainer scripts can be obtained the iterate_results_dir.py script. The script iterates over a directory contain a subdirectory per trained model.

```
python iterate_results_dir.py --ann_root <path_to_annotations> --data_root <path_to_data> --results_output <path_to_results> --log_input <path_to_model_logs> --split <dataset_split_to_use>
```

If a single model needs to be evaluted this can be done using the STL_Inference.py and MTL_Inference scripts. Additionally, if a specific set of weights needs to be used, this can be done by setting the --best_weights flag. Otherwise it is expected that there is a last.ckpt file which points to the best performing model weights.

```
python inference.py --ann_root <path_to_annotations> --data_root <path_to_data> --results_output <path_to_results> --model_path <path_to_models> --split <dataset_split_to_use>
```


## Pre-trained Models

You can download pretrained models here:

- [Model Repository](https://sciencedata.dk/shared/ctgnn_wacv2022_models) trained on Sewer-ML using the parameters described in the paper.

Each model weight file consists of a dict with the model state_dict and the necessary model hyper_parameters.

## Results
We compared the proposed CT-GNN with a classic hard shared MTL network, variations trained with Dynamic Weight Averaging and Uncertainty task balancing, as well as the MTAN soft-shared encoder. The methods are evalauted using and overall delta difference compared to the STL networks, the F2-CIW and F1-Normal metrics for defect task, and micro F1 (mF1) and macro F1 (MF1) scores for the water, shape, and material tasks. Details can be found in the paper.

### Sewer Defect and Property Classification - Validation Split

| Model | Params | ∆ | F2-CIW | F1-Normal | W-MF1 | W-mF1 | S-MF1 | S-mF1 | M-MF1 | M-mF1 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| STL | 94.0| +0.00 | 58.42 | 92.42 | 69.11 | 79.71 | 46.55 | 98.06 | 65.99 | 96.71 |
| R50-MTL |  23.5 | +10.36 | 59.73 | 91.87 | 70.51|	80.47 |	71.64 |	99.34 |	80.28 |	98.09 |
| R50-MTL-DWA | 23.5 | -15.70 | 34.22 |	86.57 |	53.43 |	70.83 |	37.68 |	98.18 |	53.50 |	90.79 |
| R50-MTL-Uncrt. | 23.5 | -4.07 | 24.80 | 86.80 | 62.00 | 75.31 | 67.30 | 99.19 | 67.46 | 95.66 |
| MTAN | 48.2 | +10.40 |61.21|	92.10| 70.06 | 80.59 | 68.34 | 99.40 | 83.48 | 98.25 |
| MTAN-CTGCN | 49.9 | +12.72 | 61.86 | 91.99 | 71.39 | 80.53 | 75.42 | 99.46 | 83.77 | 98.25 |
| MTAN-CTGAT | 48.6 | +11.48 | 61.92 | 92.03 | 70.95 | 80.50 | 71.17 | 99.39 | 83.65 | 98.29 |
| CTGCN | 25.2 | +12.39 | 61.35 | 91.84 | 70.57 |80.47 | 76.17 | 99.33	| 82.63	| 98.18 |
| CTGAT | 24.0| +12.81 | 61.70 | 91.94 | 70.57 | 80.43 | 74.53 | 99.40 | 86.63 | 98.24 |   


### Sewer Defect and Property Classification - Test Split

| Model | Params | ∆ | F2-CIW | F1-Normal | W-MF1 | W-mF1 | S-MF1 | S-mF1 | M-MF1 | M-mF1 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| STL | 94.0| +0.00 | 57.48 | 92.16 | 69.87 | 80.09 | 56.15 | 97.59 | 69.02 | 96.67 |
| R50-MTL |  23.5 | +7.39 | 58.29 |	91.57| 71.17| 81.09 | 79.48 | 99.19 | 76.35 | 98.08 |
| R50-MTL-DWA | 23.5 | -11.57 |34.84 | 86.20 | 54.30 | 71.03 | 59.27 | 97.81 | 60.39 | 90.49 |
| R50-MTL-Uncrt. | 23.5 | -3.78 | 26.30 | 86.48 | 63.01 | 76.15 | 79.69 | 98.99 | 70.84 | 95.59 |
| MTAN | 48.2 | +6.83 | 59.91 |	91.72 |	70.61 |	81.16 | 78.50 | 99.21 | 72.73 | 98.27 |
| CTGCN | 25.2 | +7.64 | 60.07 | 91.60 | 70.69 | 80.91 | 80.32 | 99.19 | 75.13 | 98.15 |
| CTGAT | 24.0| +7.84 | 60.57 |	91.61 |	71.30 |	80.91 | 81.10 | 99.22 | 73.95 | 98.26 |



## Code Credits

Parts of the code builds upon prior work:

- The GraphConvolutional layer is obtained from the ML-GCN authors implementation, an adaption of Thomas Kipf's original implementation. Found at: [https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)
- The GraphAttention layer is from the pyGAT repo by Diego Antognini, modified to handle arbitarty batch sizes by user Llijachen 1019. Found at:[https://github.com/Diego999/pyGAT/issues/36](https://github.com/Diego999/pyGAT/issues/36)
- The MTAN encoder code is based on the ResNet adaption by Simon Vandenhende: [https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/models/mtan.py](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/models/mtan.py)



## Contributing

The Code is licensed under an MIT License, with exceptions of the MTAN, GCN, and GAT code which follows the license of the original authors.

The Sewer-ML Dataset follows the Creative Commons Attribute-NonCommerical-ShareAlike 4.0 (CC BY-NC-SA 4.0) International license.



## Bibtex
```bibtex
@InProceedings{Haurum_2022_WACV,
author = {Haurum, Joakim Bruslund and Madadi, Meysam and Escalera, Sergio and Moeslund, Thomas B.},
title = {Multi-Task Classification of Sewer Pipe Defects and Properties using a Cross-Task Graph Neural Network Decoder},
booktitle={2022 IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {January},
year = {2022}
}
```
