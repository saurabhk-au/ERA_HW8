import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.RandomCrop(width=32, height=32),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])

def get_train_transforms(mean):
    return A.Compose([
        A.RandomCrop(width=32, height=32),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=mean, std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def get_test_transforms(mean):
    return A.Compose([
        A.Normalize(mean=mean, std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]) 