import torch, torchvision
import torch.optim as optim

from PIL import Image
from data_reprocessing.data_reprocessing import img_transforms
from model.model_1.model import simplenet
from training.training import train

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_data_path = "data_1/train"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms)

test_data_path = "data_1/test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms)

batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

simplenet.to(device)
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)  # обновление весов

train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, epochs=15, device=device)

torch.save(simplenet.state_dict(), "trained_model/trained_model")

labels = ['adult', 'children']

img = Image.open("test_img/Sam.jpg")
img = img_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])
