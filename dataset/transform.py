import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
             ToTensorV2(p=1.0)
         ],

        )


def get_val_transforms():
    return A.Compose(
        [
             ToTensorV2(p=1.0)
         ],

        )