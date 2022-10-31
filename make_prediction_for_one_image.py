import torch
import config

from PIL import Image
from config import device
from data_preprocessing.data_reprocessing import img_transforms
from model.model_1.model import SimpleNet

simplenet = SimpleNet()
simplenet_state_dict = torch.load(config.param_for_testing.trained_model)
simplenet.load_state_dict(simplenet_state_dict)
simplenet = simplenet.to(config.device)


def make_prediction_for_one_image():
    labels = config.param_for_testing.labels
    img = Image.open(config.param_for_testing.test_img)
    img = img_transforms(img).to(device)
    img = torch.unsqueeze(img, 0)
    prediction = simplenet(img)
    prediction = prediction.argmax()
    if labels[prediction] == 'adult':
        return "the picture shows an adult"
    else:
        return "the picture shows a child"
