import torch

from PIL import Image

import config
from config import device
from data_reprocessing.data_reprocessing import img_transforms
from model.model_1.model import SimpleNet

labels = config.param_for_testing.labels
img = Image.open(config.param_for_testing.test_img)
img = img_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

simplenet = SimpleNet()
simplenet_state_dict = torch.load(config.param_for_testing.trained_model)
simplenet.load_state_dict(simplenet_state_dict)

simplenet = simplenet.to(device)
prediction = simplenet(img)
prediction = prediction.argmax()
if labels[prediction] == 'adult':
    print("The picture shows an adult")
else:
    print("The picture shows a child")
