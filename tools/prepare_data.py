import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from config.config import cfg
import torchvision.transforms as transforms

def prepare_data():
    dataloaders = {}
    train_transform_1 = data_utils.get_transform(True)
    # train_transform_1 = transforms.Compose(
    #     [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)]
    #     + train_transform_1.transforms)
    test_transform_1 = data_utils.get_transform(False)

    train_transform_2= data_utils.get_transform(True)
    # train_transform_2 = transforms.Compose(
    #     [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)]
    #     + train_transform_2.transforms)
    test_transform_2 = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # source dataloader
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source)
    dataloaders[source] = CustomDatasetDataLoader(
                dataset_root=dataroot_S, dataset_type=dataset_type,
                batch_size=batch_size, transform_1=train_transform_1, transform_2 = train_transform_2,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    # target dataloader 
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDatasetWithoutLabel'
    print('Building %s dataloader...' % target)
    dataloaders[target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform_1=train_transform_1, transform_2 = train_transform_2,
                train=True, num_workers=cfg.NUM_WORKERS, 
                classnames=classes)

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_T, dataset_type=dataset_type,
                    batch_size=batch_size, transform_1=test_transform_1, transform_2 = test_transform_2,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)
    return dataloaders
    