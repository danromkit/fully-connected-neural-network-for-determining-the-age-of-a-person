import torch
import torch.optim as optim
import torchvision

import config
from data_preprocessing.data_reprocessing import img_transforms
from make_prediction_for_one_image import make_prediction_for_one_image
from make_prediction_for_test_images import make_prediction_for_test_images
from model.model_1.model import simplenet
from training.training import train

train_data = torchvision.datasets.ImageFolder(root=config.data.train_data_path, transform=img_transforms)

test_data = torchvision.datasets.ImageFolder(root=config.data.test_data_path, transform=img_transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.data.batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.data.batch_size, shuffle=True)

simplenet.to(config.device)
optimizer = optim.Adam(simplenet.parameters(), lr=config.param_for_learning.learning_rate)

# train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader,
#       epochs=config.param_for_learning.number_of_epochs, device=config.device)

# torch.save(simplenet.state_dict(), "trained_model/trained_model")

# prediction_for_one_image = make_prediction_for_one_image()
# print("Prediction_for_one_image:", prediction_for_one_image)

make_prediction_for_test_images(torch.nn.CrossEntropyLoss(), test_data_loader, device=config.device)