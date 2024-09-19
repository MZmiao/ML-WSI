import argparse
import os
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from test import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

#保存权重
def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default=r'/mnt/DataSets/styles_new.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # parser.add_argument('--weights', type=str, default=r'.\checkpoints\2023-10-10_10-35\checkpoint-000180.pth',help='initial weights path')

    parser.add_argument('--weights', type=str, default=r' ',help='initial weights path')

    args = parser.parse_args()
    start_epoch = 1
    N_epochs = 200
    batch_size = 8
    num_workers = 2  # 处理数据集加载的进程数
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes变量包含数据集中类别的标签以及字符串名称和ID之间的映射
    attributes = AttributesDataset(args.attributes_file)

    # 指定图像变换以在训练期间进行增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 在验证过程中，我们只使用张量和归一化变换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FashionDataset(r'/mnt/DataSets/train_new.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FashionDataset(r'/mnt/DataSets/val_new.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #创建模型
    model = MultiOutputModel(
                            n_RowSpacing_classes=attributes.num_RowSpacing,
                             n_WordSpacing_classes=attributes.num_WordSpacing,
                             n_FontSize_classes=attributes.num_FontSize,
                             n_FontShape_classes=attributes.num_FontShape,
                             n_FontInclination_classes=attributes.num_FontInclination).to(device)
    #导入权重
    if args.weights != " ":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict, strict=False)

    # for name, para in model.named_parameters():
    #     # 除head, pre_logits外，其他权重全部冻结
    #     if "mlp1"not in name and "mlp2"not in name and 'mlp3'not in name and'mlp4'not in name and'mlp5'not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    #优化器
    # optimizer = torch.optim.Adam(model.parameters())

    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / N_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    #权重保存路径
    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)#训练数据的数量

    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    print("\nAll RowSpacing labels:\n", attributes.RowSpacing_labels)
    print("\nAll WordSpacing labels:\n", attributes.WordSpacing_labels)
    print("\nAll FontShape labels:\n", attributes.FontShape_labels)
    print("\nAll FontSize labels:\n", attributes.FontSize_labels)
    print("\nAll FontInclination labels:\n",attributes.FontInclination_labels)

    print("Starting training ...")



    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_RowSpacing = 0
        accuracy_WordSpacing = 0
        accuracy_FontSize = 0
        accuracy_FontShape = 0
        accuracy_FontInclination = 0
        accuracy_id = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_RowSpacing, batch_accuracy_WordSpacing, batch_accuracy_FontSize, \
            batch_accuracy_FontShape, batch_accuracy_FontInclination, batch_accuracy_id = calculate_metrics(output, target_labels)
            # batch_accuracy_WordSpacing, batch_accuracy_FontSize, batch_accuracy_FontShape = \
            #     calculate_metrics(output, target_labels)

            accuracy_RowSpacing += batch_accuracy_RowSpacing
            accuracy_WordSpacing += batch_accuracy_WordSpacing
            accuracy_FontSize += batch_accuracy_FontSize
            accuracy_FontShape += batch_accuracy_FontShape
            accuracy_FontInclination += batch_accuracy_FontInclination
            accuracy_FontInclination += batch_accuracy_id
            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, 行间距：{:.4f},字间距: {:.4f},  字体大小: {:.4f}，字体外形: {:.4f},字体倾斜：{:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_RowSpacing / n_train_samples,
            accuracy_WordSpacing / n_train_samples,
            accuracy_FontSize / n_train_samples,
            accuracy_FontShape /n_train_samples,
            accuracy_FontInclination/n_train_samples,
            accuracy_id/n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 3 == 0:
            validate(model, val_dataloader, logger, epoch, device)
            checkpoint_save(model, savedir, epoch)
