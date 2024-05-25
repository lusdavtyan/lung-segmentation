import pytest

import torch
from torchvision import transforms

import utils
import paths

@pytest.fixture
def device():
    return utils.get_device()

@pytest.fixture
def volume_sizes():
    return utils.get_volume_sizes(paths.PATH_LIST)

@pytest.fixture
def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])

@pytest.fixture
def datasets(volume_sizes, transform):
    train_dataset = utils.CTScansDataset(volume_sizes=volume_sizes, transform=transform, mode='train')
    valid_dataset = utils.CTScansDataset(volume_sizes=volume_sizes, transform=transform, mode='valid')
    test_dataset = utils.CTScansDataset(volume_sizes=volume_sizes, transform=transform, mode='test')
    return train_dataset, valid_dataset, test_dataset

def test_get_device(device):
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'cpu']

@pytest.mark.parametrize("volumes_list", [
    [110, 301, 93, 45, 42, 66, 301, 270, 301, 256, 39, 213, 42, 200, 39, 45, 249, 200, 290, 418],
])
def test_CTScansDataset(volumes_list, volume_sizes, transform):
    assert len(volume_sizes) == len(volumes_list), "The lengths of the volume sizes is incorrect!"

    def check_dataset_mode(mode, start, end):
        dataset = utils.CTScansDataset(volume_sizes=volume_sizes, transform=transform, mode=mode)
        assert len(dataset) == sum(volume_sizes[start:end]), f"The size of the {mode} dataset is incorrect!"
        for i in range(len(volumes_list)):
            assert volume_sizes[i] == volumes_list[i], f"Volume size at index {i} is incorrect!"
        return dataset

    train_dataset = check_dataset_mode('train', 0, 14)
    valid_dataset = check_dataset_mode('valid', 14, 17)
    test_dataset = check_dataset_mode('test', 17, None)

    img, label = test_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.bool


def test_datasets_disjoint(datasets):
    train_dataset, valid_dataset, test_dataset = datasets

    train_indices = set(train_dataset.data_indices)
    valid_indices = set(valid_dataset.data_indices)
    test_indices = set(test_dataset.data_indices)

    assert train_indices.isdisjoint(valid_indices), "Train and Valid datasets overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and Test datasets overlap!"
    assert valid_indices.isdisjoint(test_indices), "Valid and Test datasets overlap!"

    assert len(train_indices) == len(train_dataset.data_indices), "Duplicate indices in Train dataset!"
    assert len(valid_indices) == len(valid_dataset.data_indices), "Duplicate indices in Valid dataset!"
    assert len(test_indices) == len(test_dataset.data_indices), "Duplicate indices in Test dataset!"

if __name__ == "__main__":
    pytest.main(args=[__file__])
