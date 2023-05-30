import numpy as np
import sklearn.metrics

def multiclass_evaluation(scores, targets):
    predictions = np.argmax(scores, axis=1)

    assert predictions.shape == targets.shape, "The predictions and targets do not have the same size: Input: {} - Targets: {}".format(predictions.shape, targets.shape)

    _, n_class = scores.shape

    # Arrays to hold binary classification information, size n_class +1 to also hold the implicit normal class
    Nc = np.zeros(n_class) # Nc = Number of Correct Predictions  - True positives
    Np = np.zeros(n_class) # Np = Total number of Predictions    - True positives + False Positives
    Ng = np.zeros(n_class) # Ng = Total number of Ground Truth occurences

    # False Positives = Np - Nc
    # False Negatives = Ng - Nc
    # True Positives = Nc
    # True Negatives =  n_examples - Np + (Ng - Nc)
    
    for k in range(n_class):
        Ng[k] = np.sum(targets == k)
        Np[k] = np.sum(predictions == k) # when >= 0 for the raw input, the sigmoid value will be >= 0.5
        Nc[k] = np.sum((targets == k) * (predictions == k))

    # If Np is 0 for any class, set to 1 to avoid division with 0
    Np[Np == 0] = 1

    # Confusion matrix
    cm = sklearn.metrics.confusion_matrix(targets, predictions)

    # per-class F1
    per_class_metrics = sklearn.metrics.precision_recall_fscore_support(targets, predictions, average=None)
    precision_k = per_class_metrics[0]
    recall_k = per_class_metrics[1]
    F1_k = per_class_metrics[2]
    
    # Macro F1
    valid_classes = Ng > 0
    valid_classes = valid_classes[:len(F1_k)]
    Mrecall = np.mean(recall_k[valid_classes])
    Mprecision = np.mean(precision_k[valid_classes])
    MF1 = np.mean(F1_k[valid_classes])
    
    # micro F1
    mprecision, mrecall, mF1, _ = sklearn.metrics.precision_recall_fscore_support(targets, predictions, average="micro")

    main_metrics = {"MF1": MF1,
                    "mF1": mF1,
                    "MP": Mprecision,
                    "mP": mprecision,
                    "MR": Mrecall,
                    "mR": mrecall}

    auxillery_metrics = {"P_class": list(precision_k),
                         "R_class": list(recall_k),
                         "F1_class": list(F1_k),
                         "CM": cm,
                         "Np": list(Np),
                         "Nc": list(Nc),
                         "Ng": list(Ng)}

    return _, main_metrics, auxillery_metrics
