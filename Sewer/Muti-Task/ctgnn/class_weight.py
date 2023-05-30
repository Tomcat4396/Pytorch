import numpy as np
import torch


def _get_unique_count(labels):
    if labels.ndim == 1:
        unique_labels, unique_counts = np.unique(labels, return_counts = True)
    elif labels.ndim == 2:
        unique_counts = [len(labels[labels[:, idx] == 1]) for idx in range(labels.shape[-1])]
        unique_labels = [idx for idx in range(labels.shape[-1])]
    
    return unique_labels, unique_counts


def identity_weight(labels, num_classes):
    class_weights = np.zeros(num_classes)
    for label_idx in range(num_classes):
        class_weights[label_idx] = 1.0

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()


def defect_CIW(label_names, rescale_classes = True):
    class_weights = np.zeros(len(label_names))
    CIW_weight = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}

    for label_idx, label_name in enumerate(label_names):
        class_weights[label_idx] = CIW_weight[label_name]

    class_weights = class_weights / np.sum(class_weights)

    if rescale_classes:
        class_weights *=  len(label_names)

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()


def positive_ratio(labels, num_classes, rescale_classes = True):
    data_len = labels.shape[0]
    class_weights = np.zeros(num_classes)

    unique_labels, unique_counts = _get_unique_count(labels)

    for label_idx in unique_labels:
        pos_count = unique_counts[label_idx]
        neg_count = data_len - pos_count
        class_weights[label_idx] = neg_count/pos_count if pos_count > 0 else 0

    class_weights = class_weights / np.sum(class_weights)

    if rescale_classes:
        class_weights *=  num_classes

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()


def inverse_frequency(labels, num_classes, rescale_classes = True):
    class_weights = np.zeros(num_classes)
    unique_labels, unique_counts = _get_unique_count(labels)
    
    for label_idx in unique_labels:
        class_weights[label_idx] = 1 / unique_counts[label_idx]

    class_weights = class_weights / np.sum(class_weights)

    if rescale_classes:
        class_weights *=  num_classes

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()
    

def effective_samples(labels, num_classes, beta, rescale_classes = True):
    class_weights = np.zeros(num_classes)
    unique_labels, unique_counts = _get_unique_count(labels)
    
    for label_idx in unique_labels:
        effective_number = 1 - np.power(beta, unique_counts[label_idx])
        class_weights[label_idx] = (1 - beta) / effective_number

    class_weights = class_weights / np.sum(class_weights)

    if rescale_classes:
        class_weights *=  num_classes

    return torch.as_tensor(class_weights, dtype=torch.float).squeeze()
