import pytest
from tests import _PATH_DATA
import os.path
import torch
from torch.utils.data import  TensorDataset, DataLoader

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_traindata_len():

    images = torch.load(_PATH_DATA+"/processed/images.pt")
    labels = torch.load(_PATH_DATA+"/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)

    assert len(images) == len(labels) == 25000, "Dataset lenght mismatch"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_train_data_shape():

    images = torch.load(_PATH_DATA+"/processed/images.pt")
    labels = torch.load(_PATH_DATA+"/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)

    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
        )

    for images_batch, _ in trainloader:
        for image in images_batch:
            assert list(image.shape) == [1,28,28], "Data not formatted in the right shape"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_label_representation():
    images = torch.load(_PATH_DATA+"/processed/images.pt")
    labels = torch.load(_PATH_DATA+"/processed/labels.pt")

    train_dataset = TensorDataset(images, labels)

    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
        )
    for _, labels_batch in trainloader:
        for label in labels_batch:
            assert label>=0 and label<=9, "Data label not in class range"