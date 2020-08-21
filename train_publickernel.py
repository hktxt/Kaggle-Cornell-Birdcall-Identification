from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.sklearns import F1
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from models.loss import AngleLoss, AngleLossWithCE
import pytorch_lightning as pl
from models.base import get_model
from dataset.augment import spec_augment, mixup
import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torchsampler import ImbalancedDatasetSampler
from dataset.dataset import callback_get_label1
from dataset.dataset import Birdcall, SpectrogramDataset
import pandas as pd
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


# https://github.com/williamFalcon/vae_demo/blob/master/vae.py
class CornellBirdCall(LightningModule):
    def __init__(self, df, model, criterion, metrics, hparams):
        super().__init__()
        self.df = df
        self.net = model
        self.criterion = criterion
        self.metrics = metrics
        self.hparams = hparams

    def forward(self, x):
        output = self.net(x)
        return output

    def get_lr(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        return current_lr

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.hparams.specaug:
            imgs, labels = mixup(imgs, labels)
            imgs, labels = spec_augment(imgs, labels)

        x = self(imgs)
        loss = self.criterion(x, labels)

        lr = self.get_lr()
        log = {'train_loss': loss, 'lr': lr}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        x = self(imgs)
        val_loss = self.criterion(x, labels)

        static = {
            'gt': labels.cpu().argmax(dim=1),
            'pred': x.cpu().argmax(dim=1)
        }
        log = {'val_loss': val_loss}
        return {'log': log, 'val_loss': val_loss, 'static': static}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        pred = torch.cat([x['static']['pred'] for x in outputs])
        gt = torch.cat([x['static']['gt'] for x in outputs])
        f1 = self.metrics(pred, gt)
        log = {'avg_val_loss': val_loss, 'f1': f1}
        return {'log': log, 'val_loss': val_loss, 'f1': f1}

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_set = SpectrogramDataset(df=df[df['fold'] != self.hparams.fold])

        if self.hparams.balanceSample:
            print('balance sample, it takes time. take three couples of coffee.')
            sampler = ImbalancedDatasetSampler(train_set, callback_get_label=callback_get_label1)
            train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=self.hparams.batch_size)
        else:
            print('normal sample.')
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_set = SpectrogramDataset(df=df[df['fold'] == self.hparams.fold])
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.hparams.batch_size)

        return val_loader


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--fold', default=0)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--arch', default='resnet50', type=str, help="model arch, ['resnet', 'resnest50', "
                                                                     "'efficientnet b0~b3', 'pyconvhgresnet', "
                                                                     "'resnet_sk2', 'se_resnet50_32x4d']")
    parser.add_argument('--classes', default=264)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--balanceSample', default=False)
    parser.add_argument('--specaug', default=False)  # seems like it's not working with AngleLoss.
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()

    seed_everything(42)

    df = pd.read_csv('data/df_mod.csv')[:1000]  # use first 30 lines for debug.
    print('training public kernel baseline.')
    model = get_model(args.arch, classes=args.classes)
    criterion = nn.BCEWithLogitsLoss()
    Bird = CornellBirdCall(df, model, criterion, metrics=F1(), hparams=args)
    lr_logger = LearningRateLogger()
    trainer = pl.Trainer(
                        gpus=[0],
                        max_epochs=30,
                        benchmark=True,
                        # accumulate_grad_batches=1,
                        # log_gpu_memory='all',
                        weights_save_path='./weights',
                        # callbacks=[lr_logger]
                    )
    trainer.fit(Bird)
