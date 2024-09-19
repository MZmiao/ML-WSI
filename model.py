
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet50 import resnet50


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOutputModel(nn.Module):
    def __init__(self, n_RowSpacing_classes, n_WordSpacing_classes, n_FontSize_classes,n_FontShape_classes,n_FontInclination_classes):
        super().__init__()

        #resney50模型
        self.base_model = resnet50()
        last_channel = 2048
        #平均池化层，后面好像没用，直接在编码器里平均池化了
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # [batch_size, channels, width, height]

        self.mlp1 = Mlp(last_channel)
        self.mlp2 = Mlp(last_channel)
        self.mlp3 = Mlp(last_channel)
        self.mlp4 = Mlp(last_channel)
        self.mlp5 = Mlp(last_channel)


        # 为输出创建单独的分类器
        self.RowSpacinge = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_RowSpacing_classes))

        self.WordSpacinge = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_WordSpacing_classes) )

        self.FontSize = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_FontSize_classes))

        self.FontShape = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_FontShape_classes))

        self.FontInclination = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_FontInclination_classes))

        self.id = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_FontInclination_classes))



    def forward(self, x):
        # print('输入图像大小',x.shape)
        x = self.base_model(x)          #[1, 1024, 14, 14]
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # print(x.shape)

        # x1 = self.mlp1(x)
        # x2 = self.mlp2(x)
        # x3 = self.mlp3(x)
        # x4 = self.mlp4(x)
        # x5 = self.mlp5(x)



        return {
            'RowSpacing': self.RowSpacinge(x),
            'WordSpacing': self.WordSpacinge(x),
            'FontSize': self.FontSize(x),
            'FontShape':self.FontShape(x),
            'FontInclination':self.FontInclination(x),
            'id': self.id(x)
        }

#计算每个分类器的loss和总loss
    def get_loss(self, net_output, ground_truth):
        RowSpacing_loss = F.cross_entropy(net_output['RowSpacing'], ground_truth['RowSpacing_labels'])
        WordSpacing_loss = F.cross_entropy(net_output['WordSpacing'], ground_truth['WordSpacing_labels'])
        FontSize_loss = F.cross_entropy(net_output['FontSize'], ground_truth['FontSize_labels'])
        FontShape_loss = F.cross_entropy(net_output['FontShape'], ground_truth['FontShape_labels'])
        FontInclination_loss = F.cross_entropy(net_output['FontInclination'],ground_truth['FontInclination_labels'])
        id_loss = F.cross_entropy(net_output['id'], ground_truth['id_labels'])

        loss = RowSpacing_loss + WordSpacing_loss + FontSize_loss + FontShape_loss + FontInclination_loss + id_loss


        return loss, {'RowSpacing': RowSpacing_loss, 'WordSpacing': WordSpacing_loss, 'FontSize': FontSize_loss, 'FontShape':FontShape_loss,'FontInclination':FontInclination_loss,'id':id_loss}

