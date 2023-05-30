import os
from argparse import ArgumentParser

def runInference(model_version, args_dict):
    try:
        if "stl" in model_version.lower():
            from STL_Inference import STL_inference
            STL_inference(args_dict)
        elif "mtl" in model_version.lower():
            from MTL_Inference import MTL_inference
            MTL_inference(args_dict)
    except ValueError as e:
        print("\t"+str(e))

def iterateResulsDirs(args):
    log_input_path = args["log_input"]
    base_output_dir = args["results_output"]

    args_dict = {"ann_root": args["ann_root"],
                    "data_root": args["data_root"],
                    "batch_size": args["batch_size"],
                    "workers": args["workers"],
                    "split": args["split"],
                    "best_weights": args["best_weights"]}


    for subdir, dirs, _ in os.walk(log_input_path):

        if not args["best_weights"] and len(dirs) == 0:
            model_subdirs = subdir[len(log_input_path):]
            model_subdirs = model_subdirs.replace("/", "|")
            model_subdirs = model_subdirs.replace("\\", "|")
            model_subdirs = model_subdirs.split("|")
            model_subdirs = [x for x in model_subdirs if len(x)]

            model = model_subdirs[0]
            version = model_subdirs[1]

            model_version = "{}_{}".format(model, version)
            args_dict["model_path"] = subdir
            args_dict["results_output"] = os.path.join(base_output_dir, model_version)
            args_dict["model_version"] = model_version

            runInference(model_version, args_dict)

        elif args["best_weights"]:
            for filename in sorted(os.listdir(subdir)):
                if not os.path.isfile(os.path.join(subdir, filename)):
                    continue
        
                model_version =  os.path.splitext(filename)[0]
                args_dict["model_path"] = os.path.join(subdir, filename)
                args_dict["results_output"] = os.path.join(base_output_dir, model_version)
                args_dict["model_version"] =  model_version

                runInference(model_version, args_dict)



if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=256, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--results_output", type=str, default = "./results")
    parser.add_argument("--log_input", type=str)
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")

    args = vars(parser.parse_args())

    iterateResulsDirs(args)