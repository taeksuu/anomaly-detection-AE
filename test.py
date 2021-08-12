from dataset import MVTecADDataset, CATEGORIES, OBJECTS, TEXTILES, Resize, ToTensor
from models.originalAE import OriginalAE
from models.largeAE import LargeAE

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import morphology
from skimage.segmentation import mark_boundaries
from skimage import measure
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_model_directory', type=str)
    parser.add_argument('--model', type=str, choices=['original', 'large'], default='large')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_space_dim', type=int, default=2048)

    parser.add_argument('--dataset_directory', type=str, default='D:/mvtec_anomaly_detection')
    parser.add_argument('--category', type=str, choices=CATEGORIES, default='hazelnut')
    parser.add_argument('--color_mode', type=str, choices=['rgb', 'grayscale'], default='rgb')

    parser.add_argument('--anomaly_map_loss', type=str, choices=['l1', 'l2', 'ssim', 'all'], default='l2')

    parser.add_argument('--save_directory', type=str, default='D:/AE_results')

    return parser.parse_args()


def main():
    args = parse_args()

    transform_list = [Resize(256), ToTensor()]
    test_dataset = MVTecADDataset(dataset_path=args.dataset_directory, mode="test", category=args.category,
                                  color_mode=args.color_mode, transform=transforms.Compose(transform_list))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # build initial model
    if args.model == 'original':
        model = OriginalAE(args.color_mode, args.latent_space_dim).to(device)
    elif args.model == 'large':
        model = LargeAE(args.color_mode, args.latent_space_dim).to(device)

    load = torch.load(args.load_model_directory)
    model.load_state_dict(load['model'])
    print("Successfully loaded model")
    model.eval()

    test_outputs = []

    test_img_list = []
    test_label_list = []
    test_mask_list = []

    # Forward test data through model
    for (image, label, mask) in tqdm(test_dataloader,
                                     '| Forwarding test data through model | {} |'.format(args.category)):
        test_img_list.extend(image.cpu().detach().numpy())
        test_label_list.extend(label.cpu().detach().numpy())
        test_mask_list.extend(mask.cpu().detach().numpy())

        # input test image into model
        with torch.no_grad():
            output = model(image.float().to(device))
        test_outputs.append(output.cpu().detach())
    test_output = torch.cat(test_outputs, 0)

    test_output = np.asarray(test_output)
    test_img_list = np.asarray(test_img_list)
    test_mask_list = np.asarray(test_mask_list)

    l1_residual_maps = []
    l2_residual_maps = []
    ssim_residual_maps = []

    for output, img in zip(test_output, test_img_list):
        l1_residual_map = np.mean(np.abs(img - output), axis=0)
        l2_residual_map = np.linalg.norm(img - output, axis=0)
        ssim_residual_map = \
        ssim(img.transpose(1, 2, 0), output.transpose(1, 2, 0), win_size=11, full=True, multichannel=True)[1]
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
        
        l1_residual_maps.append(gaussian_filter(l1_residual_map, sigma=4))
        l2_residual_maps.append(gaussian_filter(l2_residual_map, sigma=4))
        ssim_residual_maps.append(gaussian_filter(ssim_residual_map, sigma=4))

    l1_residual_maps = np.array(l1_residual_maps)
    l2_residual_maps = np.array(l2_residual_maps)
    ssim_residual_maps = np.array(ssim_residual_maps)
    all_residual_maps = l1_residual_maps + l2_residual_maps + ssim_residual_maps

    normalized_l1 = (l1_residual_maps - l1_residual_maps.min()) / (l1_residual_maps.max() - l1_residual_maps.min())
    normalized_l2 = (l2_residual_maps - l2_residual_maps.min()) / (l2_residual_maps.max() - l2_residual_maps.min())
    normalized_ssim = (ssim_residual_maps - ssim_residual_maps.min()) / (
                ssim_residual_maps.max() - ssim_residual_maps.min())
    normalized_all = (all_residual_maps - all_residual_maps.min()) / (all_residual_maps.max() - all_residual_maps.min())

    if args.anomaly_map_loss == 'l1':
        normalized_scores = normalized_l1
    elif args.anomaly_map_loss == 'l2':
        normalized_scores = normalized_l2
    elif args.anomaly_map_loss == 'ssim':
        normalized_scores = normalized_ssim
    elif args.anomaly_map_loss == 'all':
        normalized_scores = normalized_all

    # figure for graphs
    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    fig_proauc = ax[2]
    fig_prauc = ax[3]

    # image-level ROC
    img_scores = normalized_scores.reshape(normalized_scores.shape[0], -1).max(axis=1)
    test_label_list = np.asarray(test_label_list)
    FPR, TPR, _ = roc_curve(test_label_list, img_scores)
    img_rocauc = roc_auc_score(test_label_list, img_scores)
    print("image-level ROCAUC: {:.2f}".format(img_rocauc))
    fig_img_rocauc.plot(FPR, TPR, label="{} image-level ROCAUC: {:.2f}".format(args.category, img_rocauc))

    # pixel-level ROC
    test_mask_list = np.asarray(test_mask_list)
    FPR, TPR, _ = roc_curve(test_mask_list.astype(np.bool).flatten(), normalized_scores.flatten())
    pixel_rocauc = roc_auc_score(test_mask_list.astype(np.bool).flatten(), normalized_scores.flatten())
    print("pixel-level ROCAUC: {:.2f}".format(pixel_rocauc))
    fig_pixel_rocauc.plot(FPR, TPR, label="{} pixel-level ROCAUC: {:.2f}".format(args.category, pixel_rocauc))

    # PRO
    binary_anomaly_maps = np.zeros_like(normalized_scores, dtype=np.bool)
    max_step = 200
    max_score = normalized_scores.max()
    min_score = normalized_scores.min()
    delta = (max_score - min_score) / max_step

    FPR = []
    PRO = []

    for th in tqdm(np.arange(min_score, max_score, delta), 'Computing PRO score'):
        binary_anomaly_maps[normalized_scores <= th] = 0
        binary_anomaly_maps[normalized_scores > th] = 1

        PROs = []
        for b, mask in zip(binary_anomaly_maps, test_mask_list.squeeze(1)):
            for region in measure.regionprops(measure.label(mask)):
                axes0_idx = region.coords[:, 0]
                axes1_idx = region.coords[:, 1]
                TP_pixels = b[axes0_idx, axes1_idx].sum()
                PROs.append(TP_pixels / region.area)

        inverse_masks = 1 - test_mask_list.squeeze(1)
        FP_pixels = np.logical_and(inverse_masks, binary_anomaly_maps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        pro = np.mean(PROs)

        if fpr <= 0.3:
            FPR.append(fpr)
            PRO.append(pro)
    proauc = auc(FPR, PRO) * 100 / 30
    print("PROAUC: {:.2f}".format(proauc))
    fig_proauc.plot(FPR, PRO, label="{} PROAUC: {:.2f}".format(args.category, proauc))

    # precision-recall
    precision, recall, thresholds = precision_recall_curve(test_mask_list.astype(np.bool).flatten(),
                                                           normalized_scores.flatten())
    prauc = auc(recall, precision)
    print("PRAUC: {:.2f}".format(prauc))
    fig_prauc.plot(recall, precision, label="{} PRAUC: {:.2f}".format(args.category, prauc))

    # save metrics to txt file
    metrics_save_directory = os.path.join(args.save_directory,
                                          "{}_{}_b{}".format(args.model, args.latent_space_dim, args.batch_size),
                                          args.category, "anomaly_map_loss-{}".format(args.anomaly_map_loss))
    os.makedirs(metrics_save_directory, exist_ok=True)
    with open(os.path.join(metrics_save_directory, "metrics.txt"), 'w') as f:
        f.write("image-level ROCAUC: {:.2f}\n".format(img_rocauc))
        f.write("pixel-level ROCAUC: {:.2f}\n".format(pixel_rocauc))
        f.write("PROAUC: {:.2f}\n".format(proauc))
        f.write("PRAUC: {:.2f}\n".format(prauc))

    # calculate optimal threshold for final results
    precision, recall, thresholds = precision_recall_curve(test_mask_list.astype(np.bool).flatten(),
                                                           normalized_scores.flatten())
    f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(2 * precision * recall),
                         where=(precision + recall) != 0)
    threshold = thresholds[np.argmax(f1_score)]

    # final results for category
    os.makedirs(os.path.join(args.save_directory,
                             "{}_{}_b{}".format(args.model, args.latent_space_dim, args.batch_size), args.category,
                             "anomaly_map_loss-{}".format(args.anomaly_map_loss), "results"),
                exist_ok=True)
    plot_fig(list(test_img_list), normalized_scores, test_mask_list, threshold, os.path.join(args.save_directory,
                                                                                             "{}_{}_b{}".format(
                                                                                                 args.model,
                                                                                                 args.latent_space_dim,
                                                                                                 args.batch_size),
                                                                                             args.category,
                                                                                             "anomaly_map_loss-{}".format(
                                                                                                 args.anomaly_map_loss),
                                                                                             "results"),
             args.category)

    fig_img_rocauc.title.set_text("Image-level ROCAUC: {:.2f}".format(img_rocauc))
    fig_img_rocauc.legend(loc="lower right")
    fig_img_rocauc.set_xlabel("False Positive Rate")
    fig_img_rocauc.set_ylabel("True Positive Rate")

    fig_pixel_rocauc.title.set_text("Pixel-level ROCAUC: {:.2f}".format(pixel_rocauc))
    fig_pixel_rocauc.legend(loc="lower right")
    fig_pixel_rocauc.set_xlabel("False Positive Rate")
    fig_pixel_rocauc.set_ylabel("True Positive Rate")

    fig_proauc.title.set_text("PROAUC: {:.2f}".format(proauc))
    fig_proauc.legend(loc="lower right")
    fig_proauc.set_xlabel("False Positive Rate")
    fig_proauc.set_ylabel("Per Region Overlap")

    fig_prauc.title.set_text("PRAUC: {:.2f}".format(prauc))
    fig_prauc.legend(loc="lower right")
    fig_prauc.set_xlabel("Recall")
    fig_prauc.set_ylabel("Precision")

    fig.tight_layout()
    fig.savefig(
        os.path.join(args.save_directory, "{}_{}_b{}".format(args.model, args.latent_space_dim, args.batch_size),
                     args.category, "anomaly_map_loss-{}".format(args.anomaly_map_loss),
                     'metrics_curve.png'), dpi=100)


# https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i] * 255.
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


if __name__ == '__main__':
    main()
