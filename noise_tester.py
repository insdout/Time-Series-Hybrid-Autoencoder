import torch
import hydra
from hydra.utils import instantiate
import numpy as np
from utils.metric_dataloader import MetricDataPreprocessor
from tshae_test import Tester
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from utils.tshae_utils import load_tshae_model


def history_to_df(history, mean, std, i):
    """
    Convert the engine run history dictionary to a pandas DataFrame.

    :param history: Dictionary containing the engine run history.
    :param mean: Mean value used for generating noise.
    :param std: Standard deviation value used for generating noise.
    :param i: Run number.
    :return: DataFrame containing the engine run history.
    """
    
    df = pd.DataFrame.from_dict(history, orient='index').explode(["rul", "rul_hat", "z"]).reset_index(names=["engine_id"])
    split_df = pd.DataFrame(df['z'].tolist(), columns=['z0', 'z1'])
    # concat df and split_df
    df = pd.concat([df, split_df], axis=1)
    df.drop("z", axis=1, inplace=True)
    df["rul_hat"] = df["rul_hat"].apply(lambda x: x[0])
    df["run"] = i
    df["mean"] = mean 
    df["std"] = std
    return df


def plot_results_noise(df_result, path="", save=True, show=True):
    """
    Plot the results of noise experiments.

    :param df_result: DataFrame containing the results of noise experiments.
    :param path: Path to save the plot.
    :param save: Whether to save the plot or not.
    :param show: Whether to show the plot or not.
    """
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    sns.boxplot(data=df_result, x="Standard Deviation", y="Score", width=0.3, orient="v", ax=axs[0])
    sns.boxplot(data=df_result, x="Standard Deviation", y="RMSE", width=0.3, orient="v", ax=axs[1])
    sns.boxplot(data=df_result, x="Standard Deviation", y="Metric", width=0.3, orient="v", ax=axs[2])
    axs[0].set_title("Score on Test Dataset", fontsize=16)
    axs[0].set_xlabel("Noise SD", fontsize=16)
    axs[0].set_ylabel("Score", fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].tick_params(axis='both', which='minor', labelsize=14)
    axs[1].set_title("RMSE on Test Dataset", fontsize=16)
    axs[1].set_xlabel("Noise SD", fontsize=16)
    axs[1].set_ylabel("RMSE", fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='minor', labelsize=14)
    axs[2].tick_params(axis='both', which='major', labelsize=14)
    axs[2].tick_params(axis='both', which='minor', labelsize=14)
    axs[2].set_title("Metric on Validation Dataset", fontsize=16)
    axs[2].set_xlabel("Noise SD", fontsize=16)
    axs[2].set_ylabel("Metric", fontsize=16)
    for ax in axs:
        ax.yaxis.grid(False) # Hide the horizontal gridlines
        ax.xaxis.grid(False) # Show the vertical gridlines
    if save:
        img_path = path+'/images/'
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(img_path + "score_rmse_metric.png", bbox_inches='tight')
        plt.close(fig)     
    if show:
        plt.tight_layout()
        fig.show()


def plot_latent_space_noize(df_full, path, show=False, save=True):
    """
    Plot the latent space with noise experiments.

    :param df_full: DataFrame containing the engine run history.
    :param path: Path to save the plot.
    :param show: Whether to show the plot or not.
    :param save: Whether to save the plot or not.
    """
    
    stds = df_full["std"].unique()
    n_images = len(stds)
    fig, axs = plt.subplots(n_images, 2, figsize=(16, 4*n_images))
    for i, std in enumerate(stds):
        sns.lineplot(df_full[df_full["std"]==std], x="rul", y="rul_hat", linewidth=3, err_kws={"alpha": .3}, ax=axs[i,0])


        axs[i, 0].plot(range(125), range(125), linewidth=3, color="g")
        axs[i, 0].set_title(f"Predicted RUL vs True RUL, noise SD: {std}", fontsize=20)
        axs[i, 0].set_xlabel("True RUL", fontsize=20)
        axs[i, 0].set_ylabel("Predicted RUL", fontsize=20)
        axs[i, 0].tick_params(axis='both', which='major', labelsize=18)
        axs[i, 0].tick_params(axis='both', which='minor', labelsize=18)
        #axs[0, i].legend().set_visible(False)


        z0 = df_full[(df_full["std"]==std) & (df_full["run"]==0)]["z0"]
        z1 = df_full[(df_full["std"]==std) & (df_full["run"]==0)]["z1"]
        targets = df_full[(df_full["std"]==std) & (df_full["run"]==0)]["rul"]
        pb = axs[i, 1].scatter(z0, z1, c=targets, s=10)
        cbb = plt.colorbar(pb, shrink=1.0)
        cbb.set_label(f"RUL", fontsize=18)
        cbb.ax.tick_params(labelsize=16)
        axs[i, 1].set_title(f"Latent Space, noise SD: {std}", fontsize=20)
        axs[i, 1].set_xlabel("z - dim 1", fontsize=20)
        axs[i, 1].set_ylabel("z - dim 2", fontsize=20)
        axs[i, 1].tick_params(axis='both', which='major', labelsize=18)
        axs[i, 1].tick_params(axis='both', which='minor', labelsize=18)

    if save:
        img_path = path+'/images/'
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.tight_layout(h_pad=0.3)
        #plt.subplots_adjust(hspace=0.2)
        plt.savefig(img_path + "latent_space.png", bbox_inches='tight')
        plt.close(fig)
        
    if show:
        plt.tight_layout()
        plt.show()

@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    """
    Main function to run the noise experiments.

    :param config: Configuration object generated by Hydra.
    """
    # current hydra output folder
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # Preparing dataloaders:
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    # Loading trained and saved model from previous step:
    tshae_checkpoint_path = config.noise_tester.checkpoint_tshae.path
    model_tshae = load_tshae_model(tshae_checkpoint_path)

    # Running test utils:
    rul_threshold = config.knnmetric.rul_threshold
    n_neighbors = config.knnmetric.n_neighbors
    
    means = config.noise_tester.means
    stds = config.noise_tester.stds
    n_runs = config.noise_tester.n_runs

    results = []
    dfs = []

    for mean in means:
        for std in stds:
            for i in range(n_runs):
                tester = Tester(
                    **config.trainer.tester, 
                    path="./outputs/", 
                    model=model_tshae, 
                    val_loader=val_loader, 
                    test_loader=test_loader, 
                    rul_threshold=rul_threshold, 
                    n_neighbors=n_neighbors,
                    add_noise_val=True,
                    add_noise_test=True,
                    noise_mean=mean,
                    noise_std=std
                    )
                score, rmse = tester.get_test_score()
                metric = tester.metric.fit_calculate(z=tester.z, rul=tester.true_rul.ravel())
                results.append([mean, std, i, score, rmse, metric])
                temp_hist = tester.get_engine_runs()
                dfs.append(history_to_df(temp_hist, mean, std, i))

    df_full = pd.concat(dfs)
    df_full.to_csv(output_dir + "/full_runs.csv", index=False)
    df_result = pd.DataFrame(results, columns=["Mean", "Standard Deviation", "run", "Score", "RMSE", "Metric"])
    df_result.to_csv(output_dir + "/result.csv", index=False)

    plot_results_noise(df_result, path=output_dir, show=False, save=True)
    plot_latent_space_noize(df_full, path=output_dir, show=False, save=True)


if __name__ == "__main__":
    main()