from typing import NamedTuple
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Data(NamedTuple):
    train_data_path: str = "data_1/train"
    test_data_path: str = "data_1/test"
    batch_size: int = 64


class Parameters_for_learning(NamedTuple):
    learning_rate: float = 0.001
    number_of_epochs: int = 15


class Parameters_for_testing(NamedTuple):
    labels: list = ['adult', 'children']
    test_img: str = "test_img/Depp_1.jpg"
    trained_model: str = "trained_model/trained_model"


data = Data()
param_for_learning = Parameters_for_learning()
param_for_testing = Parameters_for_testing()
