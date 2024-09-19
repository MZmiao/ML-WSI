import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_RowSpacing = 0
        accuracy_WordSpacing = 0
        accuracy_FontSize = 0
        accuracy_FontShape = 0
        accuracy_FontInclination = 0
        accuracy_id = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_RowSpacing, batch_accuracy_WordSpacing, batch_accuracy_FontSize, batch_accuracy_FontShape, batch_accuracy_FontInclination , batch_accuracy_id = calculate_metrics(output, target_labels)
            # batch_accuracy_WordSpacing, batch_accuracy_FontSize, batch_accuracy_FontShape = calculate_metrics(output, target_labels)

            accuracy_RowSpacing += batch_accuracy_RowSpacing
            accuracy_WordSpacing += batch_accuracy_WordSpacing
            accuracy_FontSize += batch_accuracy_FontSize
            accuracy_FontShape += batch_accuracy_FontShape
            accuracy_FontInclination += batch_accuracy_FontInclination
            accuracy_id += batch_accuracy_id

    n_samples = len(dataloader)
    avg_loss /= n_samples

    accuracy_RowSpacing /= n_samples
    accuracy_WordSpacing /= n_samples
    accuracy_FontSize /= n_samples
    accuracy_FontShape /= n_samples
    accuracy_FontInclination /= n_samples
    accuracy_id /= n_samples
    print('-' * 72)

    print("Validation  loss: {:.4f}, 书写人：{:.4f},行间距: {:.4f}, 字间距: {:.4f}, 字体大小: {:.4f},字体外形: {:.4f},字体倾斜：{:.4f}\n".format(
        avg_loss, accuracy_id, accuracy_RowSpacing, accuracy_WordSpacing, accuracy_FontSize, accuracy_FontShape,accuracy_FontInclination))
    # print("Validation  loss: {:.4f}, 字间距: {:.4f}, 字体大小: {:.4f},字体外形: {:.4f}\n".format(
    #     avg_loss,  accuracy_WordSpacing, accuracy_FontSize, accuracy_FontShape))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_id', accuracy_id, iteration)
    logger.add_scalar('val_accuracy_RowSpacing', accuracy_RowSpacing, iteration)
    logger.add_scalar('val_accuracy_WordSpacing', accuracy_WordSpacing, iteration)
    logger.add_scalar('val_accuracy_FontSize', accuracy_FontSize, iteration)
    logger.add_scalar('val_accuracy_FontShape', accuracy_FontShape, iteration)
    logger.add_scalar('val_accuracy_FontShape', accuracy_FontInclination, iteration)
    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=False, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []

    gt_RowSpacing_all = []
    gt_WordSpacing_all = []
    gt_FontSize_all = []
    gt_FontShape_all = []
    gt_FontInclination_all = []
    gt_id_all = []

    predicted_RowSpacing_all = []
    predicted_WordSpacing_all = []
    predicted_FontSize_all = []
    predicted_FontShape_all = []

    accuracy_RowSpacing = 0
    accuracy_WordSpacing = 0
    accuracy_FontSize = 0
    accuracy_FontShape = 0
    accuracy_FontInclination = 0
    accuracy_id = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_RowSpacing = batch['labels']['RowSpacing_labels']
            gt_WordSpacing = batch['labels']['WordSpacing_labels']
            gt_FontSize = batch['labels']['FontSize_labels']
            gt_FontShape = batch['labels']['FontShape_labels']
            gt_FontInclination = batch['labels']['FontInclination_labels']
            gt_id = batch['labels']['id_labels']
            output = model(img.to(device))

            batch_accuracy_RowSpacing, batch_accuracy_WordSpacing, batch_accuracy_FontSize, batch_accuracy_FontShape, batch_accuracy_FontInclination, batch_accuracy_id= calculate_metrics(output, batch['labels'])
            # batch_accuracy_WordSpacing, batch_accuracy_FontSize, batch_accuracy_FontShape = calculate_metrics(output, batch['labels'])

            accuracy_RowSpacing += batch_accuracy_RowSpacing
            accuracy_WordSpacing += batch_accuracy_WordSpacing
            accuracy_FontSize += batch_accuracy_FontSize
            accuracy_FontShape += batch_accuracy_FontShape
            accuracy_FontInclination += batch_accuracy_FontInclination
            accuracy_id += batch_accuracy_id

            # get the most confident prediction for each image
            _, predicted_RowSpacing = output['RowSpacing'].cpu().max(1)
            _, predicted_WordSpacing = output['WordSpacing'].cpu().max(1)
            _, predicted_FontSize = output['FontSize'].cpu().max(1)
            _, predicted_FontShape = output['FontShape'].cpu().max(1)
            _,predicted_FontInclination = output['FontInclination'].cpu().max(1)
            _, predicted_id = output['id'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_RowSpacing = attributes.color_id_to_name[predicted_RowSpacing[i].item()]
                predicted_WordSpacing = attributes.gender_id_to_name[predicted_WordSpacing[i].item()]
                predicted_FontSize = attributes.article_id_to_name[predicted_FontSize[i].item()]
                predicted_FontShape = attributes.article_id_to_name[predicted_FontShape[i].item()]
                predicted_FontInclination = attributes.article_id_to_name[predicted_FontInclination[i].item()]
                predicted_id = attributes.article_id_to_name[predicted_id[i].item()]

                gt_RowSpacin = attributes.color_id_to_name[gt_RowSpacin[i].item()]
                gt_WordSpacing= attributes.gender_id_to_name[gt_WordSpacing[i].item()]
                gt_FontSize= attributes.article_id_to_name[gt_FontSize[i].item()]
                gt_FontShape = attributes.article_id_to_name[gt_FontShape[i].item()]
                gt_FontInclination = attributes.article_id_to_name[gt_FontInclination[i].item()]
                gt_id = attributes.article_id_to_name[gt_id[i].item()]

                gt_RowSpacing_all.append(gt_RowSpacing)
                gt_WordSpacing_all.append(gt_WordSpacing)
                gt_FontSize_all.append(gt_FontSize)
                gt_FontShape_all.append(gt_FontShape)
                gt_FontInclination_all.append(gt_FontInclination)
                gt_id_all.append(gt_id)


                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_RowSpacing, predicted_WordSpacing, predicted_FontSize, predicted_FontShape, predicted_FontInclination, predicted_id))
                gt_labels.append("{}\n{}\n{}".format(gt_RowSpacin, gt_WordSpacing, gt_FontSize, gt_FontShape, gt_FontInclination, gt_id))

                # labels.append("{}\n{}\n{}".format(predicted_WordSpacing, predicted_FontSize,predicted_FontShape))
                # gt_labels.append("{}\n{}\n{}".format(gt_WordSpacing, gt_FontSize, gt_FontShape))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\n 字间距: {:.4f}, 字体大小: {:.4f}，字体外形：: {:.4f}".format(
            accuracy_RowSpacing / n_samples,
            accuracy_WordSpacing / n_samples,
            accuracy_FontSize / n_samples,
            accuracy_FontShape / n_samples,
            accuracy_FontInclination / n_samples,
            accuracy_id / n_samples
            ))

    # 绘制混淆矩阵
    # if show_cn_matrices:
    #     # RowSpacing
    #     cn_matrix = confusion_matrix(
    #         y_true=gt_RowSpacing_all,
    #         y_pred=predicted_RowSpacing_all,
    #         labels=attributes.RowSpacing_labels,
    #         normalize='true')
    #     ConfusionMatrixDisplay(cn_matrix, attributes.RowSpacing_labels).plot(
    #         include_values=False, xticks_rotation='vertical')
    #     plt.title("行间距")
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # gender
    #     cn_matrix = confusion_matrix(
    #         y_true=gt_WordSpacing_all,
    #         y_pred=predicted_WordSpacing_all,
    #         labels=attributes.WordSpacing_labels,
    #         normalize='true')
    #     ConfusionMatrixDisplay(cn_matrix, attributes.WordSpacing_labels).plot(
    #         xticks_rotation='horizontal')
    #     plt.title("字间距")
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # Uncomment code below to see the article confusion matrix (it may be too big to display)
    #     cn_matrix = confusion_matrix(
    #         y_true=gt_WordSpacing_all,
    #         y_pred=predicted_WordSpacing_all,
    #         labels=attributes.WordSpacing_labels,
    #         normalize='true')
    #     plt.rcParams.update({'font.size': 1.8})
    #     plt.rcParams.update({'figure.dpi': 300})
    #     ConfusionMatrixDisplay(cn_matrix, attributes.WordSpacing_labels).plot(
    #         include_values=False, xticks_rotation='vertical')
    #     plt.rcParams.update({'figure.dpi': 100})
    #     plt.rcParams.update({'font.size': 5})
    #     plt.title("字间距")
    #     plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_RowSpacing= output['RowSpacing'].cpu().max(1)
    gt_RowSpacing = target['RowSpacing_labels'].cpu()

    _, predicted_WordSpacing = output['WordSpacing'].cpu().max(1)
    gt_WordSpacing = target['WordSpacing_labels'].cpu()

    _, predicted_FontSize = output['FontSize'].cpu().max(1)
    gt_FontSize = target['FontSize_labels'].cpu()

    _, predicted_FontShape = output['FontShape'].cpu().max(1)
    gt_FontShape = target['FontShape_labels'].cpu()

    _, predicted_FontInclination = output['FontInclination'].cpu().max(1)
    gt_FontInclination = target['FontInclination_labels'].cpu()

    _, predicted_id = output['id'].cpu().max(1)
    gt_id= target['id_labels'].cpu()



    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_RowSpacing = balanced_accuracy_score(y_true=gt_RowSpacing.numpy(), y_pred=predicted_RowSpacing.numpy())
        accuracy_WordSpacing = balanced_accuracy_score(y_true=gt_WordSpacing.numpy(), y_pred=predicted_WordSpacing.numpy())
        accuracy_FontSize = balanced_accuracy_score(y_true=gt_FontSize.numpy(), y_pred=predicted_FontSize.numpy())
        accuracy_FontShape = balanced_accuracy_score(y_true=gt_FontShape.numpy(), y_pred=predicted_FontShape.numpy())
        accuracy_FontInclination = balanced_accuracy_score(y_true=gt_FontInclination.numpy(), y_pred=predicted_FontInclination.numpy())
        accuracy_id = balanced_accuracy_score(y_true=gt_id.numpy(),y_pred=predicted_id.numpy())

    return accuracy_RowSpacing, accuracy_WordSpacing, accuracy_FontSize,accuracy_FontShape,accuracy_FontInclination,accuracy_id
    # return  accuracy_WordSpacing, accuracy_FontSize,accuracy_FontShape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default=r'checkpoints\2021-10-14_15-12\checkpoint-000050.pth', help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./fashion-product-images/styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes变量包含数据集中类别的标签以及字符串名称和ID之间的映射
    attributes = AttributesDataset(args.attributes_file)

    # 在验证过程中，我们只使用张量和归一化变换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_RowSpacing_classes=attributes.num_RowSpacing, n_WordSpacing_classes=attributes.num_WordSpacing,
                             n_FontSize_classes=attributes.num_FontSize,n_FontShape_classes=attributes.num_FontShape,
                             n_FontInclination_classes=attributes.num_FontInclination,
                             n_id_classes=attributes.num_id).to(device)

    # 训练模型的可视化
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
