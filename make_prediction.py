import torch
import os

from PIL import Image
from data_reprocessing.data_reprocessing import img_transforms
from model.model_1.model import SimpleNet

labels = ['adult', 'children']
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

img = Image.open("test_img/child-9-years.jpg")
img = img_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

simplenet = SimpleNet()
simplenet_state_dict = torch.load("trained_model/trained_model")
simplenet.load_state_dict(simplenet_state_dict)

simplenet = simplenet.to(device)
prediction = simplenet(img)
prediction = prediction.argmax()
if labels[prediction] == 'adult':
    print("The picture shows an adult")
else:
    print("The picture shows a child")
print(labels[prediction])
