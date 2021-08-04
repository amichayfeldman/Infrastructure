from Utils.train_funcs import train_model
from Models.model import HrModel
from Dataset.dataset import get_dataloaders
from ax.service.managed_loop import optimize
from sklearn.metrics import f1_score
from Utils.Losses import TrainLoss
from Configuration import *
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bo_opt(data_loaders_dict, regression_loss, conf_loss):
    result_data = []

    def train(parameterization):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(3)
        np.random.seed(2)

        model = HrModel()
        lr, wd = parameterization.values()
        config['Params']['lr'] = lr
        config['Params']['wd'] = wd
        config['Paths']['output_folder'] = os.path.join(args.main_output_folder, f"lr={lr}_wd={wd}")
        loss = TrainLoss(device=device)
        print("\n\n\n##############")
        print(f"lr={lr}_wd={wd}")
        print("_________________")
        _, _, _, _, train_loss_list, val_loss_list = train_model(model=model, data_loaders_dict=data_loaders_dict,
                                                                 config=config, loss=loss,
                                                                 save_model=False,
                                                                 write_csv=False, scheduler=False,
                                                                 print_best_epoch=False)
        train_loss_slope = (train_loss_list[-5] - train_loss_list[-1]) / 5
        score = 2 * train_loss_slope * (1 / val_loss_list[-1]) / (train_loss_slope + (1 / val_loss_list[-1]))
        losses = {"train_loss": (train_loss_list[-1], 0.0),
                  "val_loss": (val_loss_list[-1], 0.0),
                  "train_slope": (train_loss_slope, 0.0),
                  "score": (score, 0.0)}
        results = {"lr": lr, "wd": wd, "train_loss_list": train_loss_list, "val_loss_list": val_loss_list,
                   "last_val_loss": val_loss_list[-1], "score": score, "slope": train_loss_slope}
        result_data.append(results)
        print(f"score = {score}")
        return losses

    best_parameters, values, experiment, _ = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2]},
            {"name": "wd", "type": "range", "bounds": [0.0, 1e-2]},
        ],
        evaluation_function=train,
        objective_name='score',
        total_trials=40,
        random_seed=20,
        minimize=False
    )

    df = pd.DataFrame(result_data)
    return df


def compute_train_loss_slope(row):
    slope = (row['train_loss_list'][-5] - row['train_loss_list'][-1]) / -5
    row['slope'] = slope
    return row


def export_best_hparams(df, sort_column, num_filtered_samples):
    df.sort_values(by=[sort_column], inplace=True, ascending=False)
    best_perf = df[:num_filtered_samples]
    # best_perf = best_perf.apply(compute_train_loss_slope, axis=1)
    # best_perf.sort_values(by=['slope'], inplace=True)
    best_perf.reset_index(inplace=True, drop=True)
    for idx, row in best_perf.iterrows():
        lr = row["lr"]
        wd = row["wd"]
        score = row["score"]
        slope = row['slope']
        train_loss = row["train_loss_list"][-1]
        val_loss = row["val_loss_list"][-1]
        print(f"({idx})  lr={lr}, wd={wd} --- train_loss={round(train_loss, 2)}, val_loss={round(val_loss, 2)}, "
              f"slope={round(slope, 6)}, score={round(score, 4)}")

    return df


def plot_variables_map(df, param_a_name, param_b_name):
    param_a = df[param_a_name]
    param_b = df[param_b_name]
    fig = plt.figure()
    plt.xlabel(param_a_name)
    plt.ylabel(param_b_name)
    plt.plot(param_a, param_b)


def plot_acquisition_map(df, param_a_name, param_b_name, metric_name):
    param_a = df[param_a_name]
    param_b = df[param_b_name]
    func_values = df[metric_name]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(param_a, param_b, func_values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel(param_a_name)
    ax.set_ylabel(param_b_name)
    ax.set_zlabel(metric_name)
    ax.view_init(60, 35)


def main():
    ################################
    # data:
    train_dataloader, val_dataloader, test_dataloader, szmc_dataloader = get_dataloaders(config=config, test=False)
    dataloaders = {'train_dl': train_dataloader, 'val_dl': val_dataloader}

    # Losses:
    huber_loss = torch.nn.SmoothL1Loss(beta=3)
    ce_loss = torch.nn.CrossEntropyLoss()
    ################################

    results_df = bo_opt(data_loaders_dict=dataloaders,
                        regression_loss=huber_loss, conf_loss=ce_loss)
    data = export_best_hparams(df=results_df, sort_column='score', num_filtered_samples=15)
    # plot_variables_map(df=data, param_a_name="lr", param_b_name="wd")
    # plot_acquisition_map(df=data, param_a_name="lr", param_b_name="wd", metric_name="f1_score")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create dataset according to origin csv and performance csv')
    parser.add_argument('-main_output_folder', type=str, required=False, help='Path to the main output folder')
    args = parser.parse_args()
    main()
