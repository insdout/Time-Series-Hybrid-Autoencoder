import os
import yaml
from functools import reduce  # forward compatibility for Python 3
import operator
import json 
import pandas as pd 
from hydra import initialize, compose
from omegaconf import OmegaConf
import argparse
import seaborn as sns
import matplotlib.pyplot as plt



def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def read_yaml_file(filename):
    with open(filename, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)


def get_results_multirun(multirun_path):
    dir, folders, files = list(os.walk(multirun_path))[0]
    
    assert "multirun.yaml" in files, "No multirun.yaml in folder!"
    multirun_dict_path = os.path.join(dir, "multirun.yaml")
    print(multirun_dict_path)

    d = read_yaml_file(multirun_dict_path)
    sweeper_params = list(d["hydra"]["sweeper"]["params"].keys())
    print("sweeper params: ", sweeper_params)
    print("sweeper: ", d["hydra"]["sweeper"]["params"])

    dfs = []
    for folder in folders:
        if not folder.isdigit():
            continue
        folder_path = dir + folder + "/"
        config_path = folder_path + ".hydra/config.yaml"
        recomposed_config = OmegaConf.load(config_path)


        sweep_params_dict = {}
        for param in sweeper_params:
            sweep_params_dict[param] = getFromDict(recomposed_config, param.split("."))


        with open(folder_path + 'results.json') as user_file:
            results = json.load(user_file)
        rmse = results["test_rmse"]
        print(f"folder: {folder} rmse: {rmse}")
        full_dict = {**sweep_params_dict, **results}
        for key in full_dict.keys():
            full_dict[key] = [full_dict[key]]
        full_dict["experiment_folder"] = int(folder)
        df_temp = pd.DataFrame(full_dict)
        dfs.append(df_temp)

    df_full = pd.concat(dfs)
    return df_full, sweeper_params

def plot_results_noise(df_result, params, path="", save=True, show=True):
    n_params = len(params)
    for i, param in enumerate(params):
        param_unique_values = df_result[param].unique()
        n_unique = len(param_unique_values)
        fig, axs = plt.subplots(1, 3, figsize=(8, 3))
        sns.barplot(data=df_result, x=param, y="test_score", width=0.3, orient="v", ax=axs[0])
        sns.barplot(data=df_result, x=param, y="test_rmse", width=0.3, orient="v", ax=axs[1])
        sns.barplot(data=df_result, x=param, y="val_metric", width=0.3, orient="v", ax=axs[2])
        axs[0].set_ylabel("Score ")
        axs[1].set_ylabel("RMSE ")
        axs[2].set_ylabel("Metric")

        axs[0].set_title("Score on Test Dataset")
        axs[1].set_title("RMSE on Test Dataset")
        axs[2].set_title("Metric on Validation Dataset")
        for ax in axs:
            ax.yaxis.grid(False) # Hide the horizontal gridlines
            ax.xaxis.grid(False) # Show the vertical gridlines
        if save:
            img_path = path+'/images/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.tight_layout()
            fig.savefig(img_path + f"{param}_score_rmse_metric.png", bbox_inches='tight')
            plt.close(fig)     
        if show:
            plt.tight_layout()
            fig.show()

def main(multirun_path):
    df, params = get_results_multirun(multirun_path)
    
    # Save files:
    df.to_csv(os.path.join(multirun_path, "full_runs.csv"), index=False)
    with open(os.path.join(multirun_path, "params.json"), 'w') as f:
        json.dump(params, f) 

    #plot_results_noise(df, params, path=multirun_path, save=True, show=False)
    print(params)
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multirun_path", type=str, required=True,
                        help="Path to the multirun directory")
    args = parser.parse_args()

    main(args.multirun_path)