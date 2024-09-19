import torch
from PIL import Image
import os


import torchvision.transforms as transforms
from torch import device
from torch.autograd import Variable
from dataset import AttributesDataset
from model import MultiOutputModel
from ViT import vit_base_patch16_224
import numpy as np


# transform_BZ = transforms.Normalize(
#     mean=[0.5, 0.5, 0.5],
#     std=[0.5, 0.5, 0.5]
# )

val_tf = transforms.Compose([transforms.Resize(1344),transforms.ToTensor()])

if __name__ == '__main__':
    attributes = AttributesDataset(r'F:\07-Multi-Label-Writer-Identification\1002-vit\styles_new.csv')

    model = MultiOutputModel(
        n_RowSpacing_classes=attributes.num_RowSpacing,
        n_WordSpacing_classes=attributes.num_WordSpacing,
        n_FontSize_classes=attributes.num_FontSize,
        n_FontShape_classes=attributes.num_FontShape,
        n_FontInclination_classes=attributes.num_FontInclination)

    # model = vit_base_patch16_224()

    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    weights = r' '
    if weights != " ":
        assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
        weights_dict = torch.load(weights, map_location=device)
        model.load_state_dict(weights_dict, strict=False)

    img = Image.open(r'F:\07-Multi-Label-Writer-Identification\1002-vit\143-P18.png')
    img_tensor = val_tf(img)
    img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)

    x = model(img_tensor)