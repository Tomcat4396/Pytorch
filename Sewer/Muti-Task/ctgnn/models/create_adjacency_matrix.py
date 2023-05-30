import os
import pandas as pd
import numpy as np
import argparse

DefectLabels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
WaterLabels = ["0%<5%","5-15%","15-30%","30%<="]
MaterialLabels = ["Unknown", "Concrete", "Plastic", "Lining", "Vitrified clay", "Iron", "Brickwork", "Other"]
ShapeLabels =  ["Circular", "Conical", "Egg shaped", "Eye shaped", "Rectangular", "Other"]
      
waterIntervals = [5, 15, 30]


def create_cooccurrence_matrix(annRoot, split):
    gtPath = os.path.join(annRoot, "SewerML_{}.csv".format(split))
    gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["WaterLevel", "Shape", "Material"] + DefectLabels)

    # Defect labels
    labels_defects = gt[DefectLabels].values
    num_classes_defects = len(DefectLabels)

    # Water labels
    labels_water = gt["WaterLevel"].values

    if len(waterIntervals) > 0:
        num_classes_water = len(waterIntervals)+1
        labels_water[labels_water < waterIntervals[0]] = 0
        labels_water[labels_water >= waterIntervals[-1]] = num_classes_water-1
        for idx in range(1, len(waterIntervals)):
            labels_water[(labels_water >= waterIntervals[idx-1]) & (labels_water < waterIntervals[idx])] = idx
    else:
        uniqueLevels = np.unique(labels_water)
        num_classes_water = len(uniqueLevels)
        for idx, level in enumerate(uniqueLevels):
            labels_water[labels_water == level] = idx

    # Pipe shape labels
    labels_shape = gt["Shape"].values
    num_classes_shape = len(ShapeLabels)

    # Pipe material labels
    labels_material = gt["Material"].values
    num_classes_material = len(MaterialLabels)

    num_classes_total = num_classes_defects + num_classes_water + num_classes_shape + num_classes_material 
    print("#Defects: {}\n#Water: {}\n#Shape: {}\n#Material: {}\nTotal: {}".format(num_classes_defects, num_classes_water, num_classes_shape, num_classes_material, num_classes_total))

    cooccurrence_matrix = np.zeros((num_classes_total, num_classes_total))
    class_sum = np.zeros((num_classes_total), dtype=np.int)

    for idx_1 in range(num_classes_total):
        if idx_1 < num_classes_defects:
            class_labels_idx1 = labels_defects[:,idx_1]
        elif idx_1 < (num_classes_defects + num_classes_water):
            class_labels_idx1 = labels_water == (idx_1 - num_classes_defects)
        elif idx_1 < (num_classes_defects + num_classes_water + num_classes_shape):
            class_labels_idx1 = labels_shape == (idx_1 - (num_classes_defects + num_classes_water))
        elif idx_1 < (num_classes_defects + num_classes_water + num_classes_shape + num_classes_material):
            class_labels_idx1 = labels_material == (idx_1 - (num_classes_defects + num_classes_water + num_classes_shape))

        class_sum[idx_1] = np.sum(class_labels_idx1)
        for idx_2 in range(idx_1, num_classes_total):
            if idx_2 < num_classes_defects:
                class_labels_idx2 = labels_defects[:,idx_2]
            elif idx_2 < (num_classes_defects + num_classes_water):
                class_labels_idx2 = labels_water == (idx_2 - num_classes_defects)
            elif idx_2 < (num_classes_defects + num_classes_water + num_classes_shape):
                class_labels_idx2 = labels_shape == (idx_2 - (num_classes_defects + num_classes_water))
            elif idx_2 < (num_classes_defects + num_classes_water + num_classes_shape + num_classes_material):
                class_labels_idx2 = labels_material == (idx_2 - (num_classes_defects + num_classes_water + num_classes_shape))

            co_occurrence = np.logical_and(class_labels_idx1, class_labels_idx2)

            cooccurrence_matrix[idx_1, idx_2] = cooccurrence_matrix[idx_2, idx_1] = np.sum(co_occurrence)

    return cooccurrence_matrix, class_sum, [num_classes_defects, num_classes_water, num_classes_shape, num_classes_material]


def get_probability_matrix(coocc, class_count, task_num_classes):
    
    prob = np.zeros_like(coocc)
    task_indecies = []
    for task_idx in range(len(task_num_classes)+1):
        task_indecies.append(sum(task_num_classes[:task_idx]))
    
    # Calculate probability of row element given the column class
    for task_idx1 in range(len(task_num_classes)):
        for task_idx2 in range(len(task_num_classes)):
            sub_coocc = coocc[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]]

            sub_prob = sub_coocc / class_count[task_indecies[task_idx2]:task_indecies[task_idx2+1]]

            prob[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]] = sub_prob
    return prob


def get_binarized_matrix(prob, task_num_classes, binary_threshold = None):
    if binary_threshold == None:
        binary_threshold = [1/num_classes for num_classes in task_num_classes]
        print("Using uniform random class probability as binary threshold")
    else:
        print("Using user provided binary threshold")
    
    if isinstance(binary_threshold, (float, int)):
        binary_threshold = [binary_threshold for _ in task_num_classes]
        
    binary = np.zeros_like(prob)
    task_indecies = []
    for task_idx in range(len(task_num_classes)+1):
        task_indecies.append(sum(task_num_classes[:task_idx]))


    for task_idx1 in range(len(task_num_classes)):
        for task_idx2 in range(len(task_num_classes)):
            sub_binary = prob[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]].copy()

            sub_binary[sub_binary < binary_threshold[task_idx1]] = 0
            sub_binary[sub_binary >= binary_threshold[task_idx1]] = 1

            binary[task_indecies[task_idx1]:task_indecies[task_idx1+1], task_indecies[task_idx2]:task_indecies[task_idx2+1]] = sub_binary

    return binary


def get_reweighted_matrix(binary, neighbour_weight):
    reweighted = np.zeros_like(binary)

    binary = binary - np.identity(binary.shape[0]) # Remove the self loop to make neighbour weight easier to calculate

    reweighted = binary * neighbour_weight / binary.sum(1, keepdims=True) 
    reweighted = np.where(np.isnan(reweighted), 0.0, reweighted)
    reweighted = reweighted + np.identity(reweighted.shape[0], np.int)*(1-neighbour_weight)
    
    row_sums = np.sum(reweighted, axis=1)
    for idx in range(reweighted.shape[0]):
        if row_sums[idx] < 0.95: # Hack value to ensure that the sum is below 1, but allow for floating point error
            reweighted[idx,idx] = 1.0


    return reweighted



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, default="../annotations")
    parser.add_argument('--outputPath', type=str, default="../adjacency_matrices")
    parser.add_argument('--split', type=str, default = "Train")
    parser.add_argument('--binary_threshold', type=float, default = None)
    parser.add_argument('--neighbour_weight', type=float, default = 0.2)
    parser.add_argument('--valid_tasks', nargs='+', type=str, default = ["defect", "water", "shape", "material"])
    args = parser.parse_args()

    task_LUT = {"defect": 0,
                "water": 1,
                "shape": 2,
                "material": 3}
    valid_tasks = args.valid_tasks

    args = vars(args)

    outputPath = args["outputPath"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    coocc, class_sum, task_num_classes = create_cooccurrence_matrix(args["inputPath"], args["split"])

    task_indecies = [task_LUT[task] for task in valid_tasks]
    task_num_classes_tmp = [task_num_classes[t_idx] for t_idx in task_indecies]
    cooc_tmp = np.zeros((sum(task_num_classes_tmp), sum(task_num_classes_tmp)))
    mask = np.zeros_like(coocc)
    classSum_tmp = np.zeros(sum(task_num_classes_tmp), dtype=np.int)

    for idx1, task_class1 in enumerate(task_num_classes):
        new_start = 0
        new_end = -1
        for idx2, task_class2 in enumerate(task_num_classes):
            if idx1 not in task_indecies:
                continue
            if idx2 not in task_indecies:
                continue

            start_idx1 = sum(task_num_classes[:idx1])
            end_idx1 = sum(task_num_classes[:idx1+1])
            start_idx2 = sum(task_num_classes[:idx2])
            end_idx2 = sum(task_num_classes[:idx2+1])

            mask[start_idx1:end_idx1, start_idx2:end_idx2] = 1

            new_end = new_start + task_class2
            classSum_tmp[new_start:new_end] = class_sum[start_idx2:end_idx2]
            new_start = new_end     

    mask = mask.astype(np.bool)
    cooc_tmp = np.ravel(coocc)[np.ravel(mask)]

    # reshape the squished array to a 2D square
    cooc_tmp = cooc_tmp.reshape(                         # reshape the array
            np.sqrt(cooc_tmp.size).astype(int),     # pass to int to stop warning
            -1                                      # fill next axis with leftover
        )
        
    coocc = cooc_tmp
    task_num_classes = task_num_classes_tmp
    class_sum = classSum_tmp

    prob = get_probability_matrix(coocc, class_sum, task_num_classes)
    binary = get_binarized_matrix(prob, task_num_classes, args["binary_threshold"])
    reweighted = get_reweighted_matrix(binary, args["neighbour_weight"])

    prob_masked = prob * binary
    reweighted_cond = get_reweighted_matrix(prob_masked, args["neighbour_weight"])

    full_binary = np.ones_like(reweighted_cond)

    np.save(os.path.join(outputPath, "co_occurrence.npy"), coocc)
    np.save(os.path.join(outputPath, "cond_probability.npy"), prob)
    np.save(os.path.join(outputPath, "adj_binary.npy"), binary)
    np.save(os.path.join(outputPath, "adj_reweighted.npy"), reweighted)
    np.save(os.path.join(outputPath, "adj_reweightedCond.npy"), reweighted_cond)
    np.save(os.path.join(outputPath, "adj_binaryFull.npy"), full_binary)

    
    np.savetxt(os.path.join(outputPath, "co_occurrence.csv"), coocc, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "cond_probability.csv"), prob, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_binary.csv"), binary, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_reweighted.csv"), reweighted, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_reweightedCond.csv"), reweighted_cond, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(outputPath, "adj_binaryFull.csv"), full_binary, delimiter=",", fmt='%f')
