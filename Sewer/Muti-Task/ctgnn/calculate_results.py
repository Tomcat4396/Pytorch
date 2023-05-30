import os
import json
import argparse
import pandas as pd
import numpy as np
from multilabel_metrics import multilabel_sewerml_evaluation
from multiclass_metrics import multiclass_evaluation

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def defectResults(scoresDf, targetsDf):
    LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
    Labels = list(LabelWeightDict.keys())
    LabelWeights = list(LabelWeightDict.values())

    targets = targetsDf[Labels].values.copy()
    scores = scoresDf[Labels].values
    new, main, auxillary = multilabel_sewerml_evaluation(scores, targets, LabelWeights)

    resultsDict = {"Labels": Labels, "LabelWeights": LabelWeights, "New": new, "Main": main, "Auxillary": auxillary}
    
    resultsStr = ""
    resultsStr += "New metrics: " + "{:.2f} & {:.2f} ".format(new["F2"]*100,  auxillary["F1_class"][-1]*100) + "\n"
    resultsStr += "ML main metrics: " + "{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(main["mF1"]*100, main["MF1"]*100, main["OF1"]*100, main["OP"]*100, main["OR"]*100, main["CF1"]*100, main["CP"]*100, main["CR"]*100, main["EMAcc"]*100, main["mAP"]*100) + "\n"
    resultsStr += "Class F1: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]]) + "\n"
    resultsStr += "Class F2: " + " & ".join(["{:.2f}".format(x*100) for x in new["F2_class"]]) + "\n"
    resultsStr += "Class Precision: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]]) + "\n"
    resultsStr += "Class Recall: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]]) + "\n"
    resultsStr += "Class AP: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["AP"]]) + "\n"

    return resultsDict, resultsStr


def waterResults(scoresDf, targetsDf, waterIntervals = [5, 15, 30]):
    LabelWeightDict = {"0%<5%":1.0,"5-15%":1.0,"15-30%":1.0,"30%<=":1.0}
    Labels = list(LabelWeightDict.keys())
    LabelWeights = list(LabelWeightDict.values())

    scores = scoresDf[Labels].values
    targets = targetsDf["WaterLevel"].values.copy()
    
    if len(waterIntervals) > 0:
        num_classes = len(waterIntervals)+1
        targets[targets < waterIntervals[0]] = 0
        targets[targets >= waterIntervals[-1]] = num_classes-1
        for idx in range(1, len(waterIntervals)):
            targets[(targets >= waterIntervals[idx-1]) & (targets < waterIntervals[idx])] = idx
    else:
        uniqueLevels = np.unique(targets)
        num_classes = len(uniqueLevels)
        for idx, level in enumerate(uniqueLevels):
            targets[targets == level] = idx

    new, main, auxillary = multiclass_evaluation(scores, targets)

    resultsDict = {"Labels": Labels, "LabelWeights": LabelWeights, "Main": main, "Auxillary": auxillary}
    
    resultsStr = ""
    resultsStr += "Main metrics: " + "{:.2f} & {:.2f} ".format(main["MF1"]*100,  main["mF1"]*100) + "\n"
    resultsStr += "Auxillary metrics: " + "{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(main["MP"]*100, main["MR"]*100, main["mP"]*100, main["mR"]*100) + "\n"
    resultsStr += "Class F1: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]]) + "\n"
    resultsStr += "Class Precision: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]]) + "\n"
    resultsStr += "Class Recall: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]]) + "\n"
    resultsStr += "Confusion Matrix:\n " + np.array2string(auxillary["CM"]) + "\n"

    return resultsDict, resultsStr


def shapeResults(scoresDf, targetsDf):

    LabelWeightDict = {"Circular":1.0,"Conical":1.0,"Egg shaped":1.0,"Eye shaped":1.0, "Rectangular": 1.0, "Other": 1.0}
    Labels = list(LabelWeightDict.keys())
    LabelWeights = list(LabelWeightDict.values())

    scores = scoresDf[Labels].values
    targets = targetsDf["Shape"].values.copy()
    
    new, main, auxillary = multiclass_evaluation(scores, targets)

    resultsDict = {"Labels": Labels, "LabelWeights": LabelWeights, "Main": main, "Auxillary": auxillary}
    
    resultsStr = ""
    resultsStr += "Main metrics: " + "{:.2f} & {:.2f} ".format(main["MF1"]*100,  main["mF1"]*100) + "\n"
    resultsStr += "Auxillary metrics: " + "{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(main["MP"]*100, main["MR"]*100, main["mP"]*100, main["mR"]*100) + "\n"
    resultsStr += "Class F1: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]]) + "\n"
    resultsStr += "Class Precision: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]]) + "\n"
    resultsStr += "Class Recall: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]]) + "\n"
    resultsStr += "Confusion Matrix:\n " + np.array2string(auxillary["CM"]) + "\n"

    return resultsDict, resultsStr

def materialResults(scoresDf, targetsDf):

    LabelWeightDict = {"Unknown":1.0,"Concrete":1.0,"Plastic":1.0,"Lining":1.0, "Vitrified clay": 1.0, "Iron": 1.0, "Brickwork": 1.0, "Other": 1.0}
    Labels = list(LabelWeightDict.keys())
    LabelWeights = list(LabelWeightDict.values())

    scores = scoresDf[Labels].values
    targets = targetsDf["Material"].values.copy()
    
    new, main, auxillary = multiclass_evaluation(scores, targets)

    resultsDict = {"Labels": Labels, "LabelWeights": LabelWeights, "Main": main, "Auxillary": auxillary}
    
    resultsStr = ""
    resultsStr += "Main metrics: " + "{:.2f} & {:.2f} ".format(main["MF1"]*100,  main["mF1"]*100) + "\n"
    resultsStr += "Auxillary metrics: " + "{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(main["MP"]*100, main["MR"]*100, main["mP"]*100, main["mR"]*100) + "\n"
    resultsStr += "Class F1: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]]) + "\n"
    resultsStr += "Class Precision: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]]) + "\n"
    resultsStr += "Class Recall: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]]) + "\n"
    resultsStr += "Confusion Matrix:\n " + np.array2string(auxillary["CM"]) + "\n"

    return resultsDict, resultsStr


def calcualteResults(args):
    scorePath = args["score_path"]
    targetPath = args["gt_path"]

    outputPath = args["output_path"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    split = args["split"]

    targetSplitpath = os.path.join(targetPath, "SewerML_{}.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",", encoding="utf-8")
    targetsDf = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)

    for subdir, dirs, files in os.walk(scorePath):
        print(subdir)
        output_Dict = {}
        for scoreFile in files:
            if split.lower() not in scoreFile:
                continue
            if "water" not in scoreFile and "shape" not in scoreFile and "defect" not in scoreFile and "material" not in scoreFile:
                continue
            if not "sigmoid" in scoreFile:
                continue
            if os.path.splitext(scoreFile)[-1] != ".csv":
                continue
            print(scoreFile)

            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scoresDf = scoresDf.sort_values(by=["Filename"]).reset_index(drop=True)

            if "defect" in scoreFile:
                resultsDict, resultsStr = defectResults(scoresDf, targetsDf)
                output_Dict["defect"] = resultsStr
            elif "water" in scoreFile:
                resultsDict, resultsStr = waterResults(scoresDf, targetsDf)
                output_Dict["water"] = resultsStr
            elif "shape" in scoreFile:
                resultsDict, resultsStr = shapeResults(scoresDf, targetsDf)
                output_Dict["shape"] = resultsStr
            elif "material" in scoreFile:
                resultsDict, resultsStr = materialResults(scoresDf, targetsDf)
                output_Dict["material"] = resultsStr

            outputName = "{}_{}".format(split, scoreFile)
            if split.lower() == "test":
                outputName = outputName[:len(outputName) - len("_test_sigmoid.csv")]
            elif split.lower() == "val":
                outputName = outputName[:len(outputName) - len("_val_sigmoid.csv")]
            elif split.lower() == "train":
                outputName = outputName[:len(outputName) - len("_train_sigmoid.csv")]


            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump(resultsDict, fp, cls=NumpyEncoder)

            with open(os.path.join(outputPath,'{}_latex.txt'.format(outputName)), "w") as text_file:
                text_file.write(resultsStr)
        print()
        
        if len(output_Dict):
            with open(os.path.join(outputPath,'{}_{}_latex.txt'.format(split, os.path.basename(os.path.normpath(subdir)))), "w") as text_file:
                for key in ["defect", "water", "shape", "material"]:
                    if key in output_Dict.keys():
                        text_file.write(key + "\n\n" + output_Dict[key] + "\n\n")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "./resultsMetrics")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str, default = "./results")
    parser.add_argument("--gt_path", type=str, default = "./annotations")

    args = vars(parser.parse_args())

    calcualteResults(args)
