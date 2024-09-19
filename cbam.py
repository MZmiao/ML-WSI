import torch
from torch import nn



# ��1��ͨ��ע��������
class channel_attention(nn.Module):
    # ��ʼ��, in_channel������������ͼ��ͨ����, ratio�����һ��ȫ���ӵ�ͨ���½�����
    def __init__(self, in_channel, ratio=4):
        # �̳и����ʼ������
        super(channel_attention, self).__init__()

        # ȫ�����ػ� [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # ȫ��ƽ���ػ� [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # ��һ��ȫ���Ӳ�, ͨ�����½�4��
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # �ڶ���ȫ���Ӳ�, �ָ�ͨ����
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu�����
        self.relu = nn.ReLU()
        # sigmoid�����
        self.sigmoid = nn.Sigmoid()

    # ǰ�򴫲�
    def forward(self, inputs):
        # ��ȡ��������ͼ��shape
        b, c, h, w = inputs.shape

        # ����ͼ����ȫ�����ػ� [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # ����ͼ���ȫ��ƽ���ػ� [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # �����ػ������ά�� [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # ��һ��ȫ���Ӳ��½�ͨ���� [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # �����
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # �ڶ���ȫ���Ӳ�ָ�ͨ���� [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # �������ֳػ������� [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid����Ȩֵ��һ��
        x = self.sigmoid(x)
        # ����ά�� [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # ��������ͼ��ͨ��Ȩ����� [b,c,h,w]
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# ��2���ռ�ע��������
class spatial_attention(nn.Module):
    # ��ʼ��������˴�СΪ7*7
    def __init__(self, kernel_size=7):
        # �̳и����ʼ������
        super(spatial_attention, self).__init__()

        # Ϊ�˱��־��ǰ�������ͼshape��ͬ�����ʱ��Ҫpadding
        padding = kernel_size // 2
        # 7*7����ں�ͨ����Ϣ [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid����
        self.sigmoid = nn.Sigmoid()

    # ǰ�򴫲�
    def forward(self, inputs):
        # ��ͨ��ά�������ػ� [b,1,h,w]  keepdim����ԭ�����
        # ����ֵ����ĳά�ȵ����ֵ�Ͷ�Ӧ������
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # ��ͨ��ά����ƽ���ػ� [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # �ػ���Ľ����ͨ��ά���϶ѵ� [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # ����ں�ͨ����Ϣ [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # �ռ�Ȩ�ع�һ��
        x = self.sigmoid(x)
        # ��������ͼ�Ϳռ�Ȩ�����
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# ��3��CBAMע��������
class cbam(nn.Module):
    # ��ʼ����in_channel��ratio=4����ͨ��ע�������Ƶ�����ͨ�����͵�һ��ȫ�����½���ͨ����
    # kernel_size����ռ�ע�������Ƶľ���˴�С
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        # �̳и����ʼ������
        super(cbam, self).__init__()

        # ʵ����ͨ��ע��������
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # ʵ�����ռ�ע��������
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # ǰ�򴫲�
    def forward(self, inputs):
        # �Ƚ�����ͼ�񾭹�ͨ��ע��������
        x = self.channel_attention(inputs)
        # Ȼ�󾭹��ռ�ע��������
        x = self.spatial_attention(x)

        return x

