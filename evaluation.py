import os
from Dataset.dataset import get_dataloaders
import torch

from Configuration import *
import seaborn as sns
import multiprocessing
sns.set_theme()
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import _pickle as cPickle
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(3)
np.random.seed(2)

huber_loss = torch.nn.SmoothL1Loss(beta=3, reduction='none')
l1loss = torch.nn.L1Loss(reduction='none')


def compute_confusion_matrix(confidence_bins, gt_bins):
    conf = confidence_bins.reshape(-1)
    gt = gt_bins.reshape(-1)
    labels = np.arange((config['Params']['range'][0] // 10) + 1, (config['Params']['range'][1] // 10) + 1)
    return confusion_matrix(y_true=gt, y_pred=conf, labels=labels, normalize='true'), \
           confusion_matrix(y_true=gt, y_pred=conf, labels=labels, normalize='pred')


def save_single_sample(setup_id, start_t, input_pdf, sample_diff, prediction, reference, confidence, output_path,
                       min_bin, max_bin, yticks_reg, yticks_conf):
    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(15, 10)

    # - - - Regression - - - #
    heatmap1 = sns.heatmap(input_pdf, yticklabels=np.arange(max_bin - min_bin), center=0.5, ax=axes[0])
    heatmap1.set_yticklabels(yticks_reg, fontsize=5)
    axes[0].plot(np.arange(input_pdf.shape[1]), reference, color='g', label='reference')
    axes[0].plot(np.arange(input_pdf.shape[1]), prediction, color='fuchsia', label='prediction')
    title1 = f"{setup_id}"
    axes[0].set_title(title1)
    axes[0].legend()
    axes[0].tick_params(axis='y', labelsize=5)

    # - - - Confidences - - - #
    sns.heatmap(confidence, yticklabels=yticks_conf, center=0.5, linewidths=0.5, ax=axes[1])

    title2 = f"confidence heatmap"
    axes[1].set_title(title2)
    axes[1].set_xlabel("Time [Sec]")
    axes[1].set_ylabel("")
    fig.tight_layout()

    # - - - Saving - - - #
    if sample_diff < 4:
        folder = "-4"
    elif 4 < sample_diff < 10:
        folder = "4-10"
    elif sample_diff > 10:
        folder = "10+"
    file_name = f"setup={setup_id}_start_t={start_t}" \
                f"__loss={round(float(sample_diff), 2)}"
    if os.path.isfile(os.path.join(output_path, folder, file_name + '.jpg')):
        exist, num = True, 1
        while exist:
            new_name = f"setup={setup_id}_start_t={start_t}_{num}___loss={round(float(sample_diff), 2)}.jpg"
            if not os.path.isfile(os.path.join(output_path, folder, new_name)):
                break
            num += 1
        fig.savefig(os.path.join(output_path, folder, new_name))
    else:
        fig.savefig(os.path.join(output_path, folder, file_name + '.jpg'))
    plt.close(fig)


def eval_model(model, dataloaders_dict, output_folder):
    min_bin = int(round(config['Params']['range'][0] / (60 * config['Params']['gap'])))
    max_bin = int(round(config['Params']['range'][1] / (60 * config['Params']['gap'])))
    yticks_regression = [str(int(60 * i * config['Params']['gap'])) for i in range(min_bin, max_bin)]
    yticks_confidences = [f"{i * 10}-{(i + 1) * 10}" for i in range(int(config['Params']['range'][0] // 10),
                                                                    int(config['Params']['range'][1] // 10))]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.change_mode('eval')
    with torch.no_grad():
        for key, dataloader in dataloaders_dict.items():
            set_path = os.path.join(output_folder, key[:key.find('_')])
            dl_results = []
            if not os.path.isdir(set_path):
                os.makedirs(os.path.join(set_path, "-4"))
                os.makedirs(os.path.join(set_path, "4-10"))
                os.makedirs(os.path.join(set_path, "10+"))
            for i, data in enumerate(tqdm(dataloader, desc=f"{key[:key.find('_')]} loader eval:", unit="batch",
                                          ncols=100)):
                input_pdf, gt = data['pdf'].to(device), data['gt'].to(device)
                predictions, confidences = model(input_pdf.type(torch.FloatTensor).to(device))
                confidences = torch.exp(confidences)[:, :, :, 0].detach().cpu().numpy()
                unnormalized_predictions = (((predictions + 1) / 2) * config['Params']['range'][1]).view(
                    predictions.shape[0], -1).detach().cpu().numpy()
                unnormalized_gt = (((gt + 1) / 2) * config['Params']['range'][1]).view(
                    unnormalized_predictions.shape).detach().cpu().numpy()
                diff = np.abs(unnormalized_gt - unnormalized_predictions)

                prediction_bins = np.floor(unnormalized_predictions / (60 * config['Params']['gap']) -
                                           min_bin).astype(np.int64) + 1
                reference_bins = np.floor(unnormalized_gt / (60 * config['Params']['gap']) -
                                          min_bin).astype(np.int64) + 1
                confidence_bins = np.argmax(confidences, axis=1)
                gt_of_confidence = (unnormalized_gt // 10) - (config['Params']['range'][0] // 10)
                samples_diff = np.mean(diff, axis=1)

                # pool = multiprocessing.Pool()
                processes = []
                for j, sample in enumerate(input_pdf):
                    sample_diff = samples_diff[j]
                    sample = sample.squeeze().detach().cpu().numpy()[min_bin:max_bin, :]
                    prediction = unnormalized_predictions[j]
                    ref = unnormalized_gt[j]
                    pred_bins = prediction_bins[j]
                    ref_bins = reference_bins[j]
                    conf_bin = confidence_bins[j, ...]
                    gt_of_conf = gt_of_confidence[j, ...]

                    process = multiprocessing.Process(target=save_single_sample, args=(data['setup_id'][j],
                                                                                       data['start_t'][j],
                                                                                       sample,
                                                                                       sample_diff,
                                                                                       pred_bins,
                                                                                       ref_bins,
                                                                                       confidences[j, ...],
                                                                                       set_path,
                                                                                       min_bin,
                                                                                       max_bin,
                                                                                       yticks_regression,
                                                                                       yticks_confidences, ))
                    process.start()

                    processes.append(process)
                    # save_single_sample(data_dict=sample_dict)
                    dl_results.append({'setup_id': data['setup_id'][j], 'prediction': prediction, 'gt': ref,
                                       'conf_bin': conf_bin, 'gt_of_conf': gt_of_conf, 'l1_loss': float(sample_diff)})
                for p in processes:
                    p.join()

            plt.close('all')

            # # # # --- plot dataloader results: --- # # #
            good_perf = len(next(os.walk(os.path.join(set_path, "-4")))[2])
            med_perf = len(next(os.walk(os.path.join(set_path, "4-10")))[2])
            bad_perf = len(next(os.walk(os.path.join(set_path, "10+")))[2])
            total_samples = good_perf + med_perf + bad_perf
            print(f"Model performance on *{key}*:")
            print(f"\tgood performance (-4 dist) = {round(good_perf / total_samples, 2)}")
            print(f"\tmedium performance (4-10 dist) = {round(med_perf / total_samples, 2)}")
            print(f"\tbad performance (10+ BPM dist) = {round(bad_perf / total_samples, 2)}")

            dl_df = pd.DataFrame(dl_results)
            dl_df.to_csv(os.path.join(set_path, f"{key}_eval_summary.csv"))

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            fig.set_size_inches(15, 10)
            loss_hist = dl_df['l1_loss'].hist(bins=10, ax=ax1)
            ax1.set_title(f'Error histogram of model performance on {key}')
            ax1.set_xlabel("l1 loss")
            num_hist_bins = int((config['Params']['range'][1] - config['Params']['range'][0]) // 10)
            ax2.hist([np.array(dl_df['prediction'].to_list()).reshape(-1), np.array(dl_df['gt'].to_list()).reshape(-1)],
                     num_hist_bins, histtype='bar', label=["prediction", "reference"], density=True)
            ax2.legend()
            ax2.set_title("Predictions histogram")
            ax2.set_xlabel("")

            sampled_predictions = dl_df.sample(n=10, random_state=1)
            all_predictions = np.round(np.array(sampled_predictions['prediction'].to_list()) /
                                       (60 * config['Params']['gap'])) - min_bin
            preds_mean = np.mean(all_predictions, axis=1).reshape(-1, 1)
            pred_with_zero_mean = (all_predictions - np.tile(preds_mean, (1, all_predictions.shape[1]))).T
            ax3.plot(np.arange(pred_with_zero_mean.shape[0]), pred_with_zero_mean)
            ax3.set_title(f"All regression predictions for {key}")
            ax3.set_xlabel("")
            ax3.set_ylabel("")

            fig.tight_layout()
            fig.savefig(os.path.join(set_path, 'Histograms.jpg'), bbox_inches='tight')

            fig, (ax_cm1, ax_cm2) = plt.subplots(1, 2)
            fig.set_size_inches(15, 10)
            confusion_mat_normed_by_true, confusion_mat_normed_by_pred = \
                compute_confusion_matrix(confidence_bins=np.stack(dl_df['conf_bin'].values),
                                         gt_bins=np.stack(dl_df['gt_of_conf'].values))
            f1score = f1_score(y_true=np.stack(dl_df['gt_of_conf'].values).reshape(-1),
                                y_pred=np.stack(dl_df['conf_bin'].values).reshape(-1),
                                average='weighted')
            sns.heatmap(confusion_mat_normed_by_true, ax=ax_cm1, center=0.5, annot=True)
            ax_cm1.set_xlabel("True label")
            ax_cm1.set_ylabel("Predicted label")
            ax_cm1.set_title(f"Confusion matrix - normalized by true columns,  f1 score={round(f1score, 2)}")
            sns.heatmap(confusion_mat_normed_by_pred, ax=ax_cm2, center=0.5, annot=True)
            ax_cm1.set_xlabel("True label")
            ax_cm1.set_ylabel("Predicted label")
            ax_cm1.set_title(f"Confusion matrix - normalized by prediction columns,  f1 score={round(f1score, 2)}")
            fig.savefig(os.path.join(set_path, 'Confidence_confusion_matrix.jpg'), bbox_inches='tight')

            plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot given model performance')
    parser.add_argument('-model_path', type=str, required=False, help='Path to the model pickle')
    parser.add_argument('-model_checkpoint', type=str, required=False, help='Path to the model checkpoint')
    parser.add_argument('-output_path', type=str, required=False, help='Path to the folder which '
                                                                       'plottings will be stored')
    args = parser.parse_args()
    config['Params']['load_data_in_init'] = False
    with open(args.model_path, "rb") as input_file:
        model = cPickle.load(input_file)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=torch.device('cpu')))
    train_dataloader, val_dataloader, test_dataloader, szmc_dataloader = get_dataloaders(config=config, test=True)
    dataloaders = {'train_dl': train_dataloader, 'val_dl': val_dataloader,
                   'test_dl': test_dataloader, 'szmc_dl': szmc_dataloader}
    output_folder = os.path.join(args.output_path, 'evaluation_results')
    eval_model(model=model, dataloaders_dict=dataloaders, output_folder=output_folder)
