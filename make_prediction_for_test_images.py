import torch
from torch.utils.data.dataloader import DataLoader

import config
from model.model_1.model import SimpleNet

simplenet = SimpleNet()
simplenet_state_dict = torch.load(config.param_for_testing.trained_model)
simplenet.load_state_dict(simplenet_state_dict)
simplenet = simplenet.to(config.device)


def make_prediction_for_test_images(loss_fn, test_data_loader: DataLoader, device="cpu"):
    test_loss = 0
    correct = 0
    for inputs, targets in test_data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = simplenet(inputs)
        loss = loss_fn(output, targets)

        test_loss += loss_fn(output, targets).data[0]
        pred = output.data.max(1)[1]  # получаем индекс максимального значения
        correct += pred.eq(targets.data).sum()

    test_loss /= len(test_data_loader.dataset)
    print('nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))
