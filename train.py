from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.sklearns import F1
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from models.loss import AngleLoss, AngleLossWithCE
import pytorch_lightning as pl
from models.base import get_model
from dataset.augment import spec_augment, mixup
import torch
import random
from models.loss import TopKLossWithBCE, F1LossWithBCE
import torch.nn as nn
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

    # def get_lr(self):
    #     current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     return current_lr

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.hparams.specaug and random.random() > 0.5:
            phase = random.choice([0, 1, 2])
            if phase == 0:
                # mixup
                imgs, labels = mixup(imgs, labels)
            elif phase == 1:
                # spec_aug
                imgs, labels = spec_augment(imgs, labels)
            else:
                # both
                imgs, labels = mixup(imgs, labels)
                imgs, labels = spec_augment(imgs, labels)

        x = self(imgs)
        loss = self.criterion(x, labels)

        # lr = self.get_lr()
        # log = {'train_loss': loss, 'lr': lr}
        log = {'train_loss': loss}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        x = self(imgs)
        val_loss = self.criterion(x, labels)

        # static = {
        #     'gt': labels.cpu().argmax(dim=1),
        #     'pred': x.cpu().argmax(dim=1)
        # }

        x = x.cpu().float()
        n = x.shape[0]
        for i in range(n):
            t = x[i]
            x[i][t == t.max()] = 1
            x[i][t != t.max()] = 0
        static = {
            'gt': labels.cpu(),
            'pred': x
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

        optimizer = torch.optim.Adam(self.parameters(), lr=(self.hparams.lr or self.learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        if self.hparams.feature == 'kaggle':
            train_set = SpectrogramDataset(df=self.df[self.df['fold'] != self.hparams.fold], train=True)
        elif self.hparams.feature == 'kesci':
            train_set = Birdcall(df=self.df[self.df['fold'] != self.hparams.fold], train=True)
        else:
            NotImplementedError

        if self.hparams.balanceSample:
            print('balance sample, it will take ~21min for train set.')
            sampler = ImbalancedDatasetSampler(train_set, callback_get_label=callback_get_label1)
            train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=self.hparams.batch_size,
                                                       drop_last=True)
        else:
            print('normal sample.')
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True,
                                                       drop_last=True)

        return train_loader

    def val_dataloader(self):
        if self.hparams.feature == 'kaggle':
            val_set = SpectrogramDataset(df=self.df[self.df['fold'] == self.hparams.fold], train=False)
        elif self.hparams.feature == 'kesci':
            val_set = Birdcall(df=self.df[self.df['fold'] == self.hparams.fold], train=False)
        else:
            NotImplementedError
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.hparams.batch_size)

        return val_loader


def train(args):
    seed_everything(args.seed)

    df = pd.read_csv('data/df_mod.csv')[:300]  # use first 30 lines for debug.
    print(args)
    model = get_model(args.arch, classes=args.classes, vgg=args.vgg, feature=args.feature)
    if args.f1loss:
        criterion = F1LossWithBCE(classes=args.classes, weights=[1, 1])
    else:
        if args.topK > 0:
            assert args.topK < 1, 'args.topK Err.'
            criterion = TopKLossWithBCE(args.topK)
        else:
            criterion = nn.BCEWithLogitsLoss()
    Bird = CornellBirdCall(df, model, criterion, metrics=F1(average='samples'), hparams=args)
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False
    )
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=args.epochs,
        benchmark=True,
        # accumulate_grad_batches=1,
        # log_gpu_memory='all',
        weights_save_path=f'./weights/{args.arch}',
        amp_level='O1',
        precision=args.precision,
        # auto_lr_find=True,
        callbacks=[lr_logger],
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(Bird)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--arch', default='resnet50', type=str, help="model arch, ['resnet50', 'resnest50', "
                                                                      "'efficientnet-b0~3', 'pyconvhgresnet', "
                                                                      "'resnet_sk2', 'se_resnet50_32x4d']")
    parser.add_argument('--classes', default=264, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--topK', default=-1, type=float, help='topK loss, range 0~1. <0 not use.')
    parser.add_argument('--f1loss', default=False, type=bool, help='F1 loss')
    parser.add_argument('--balanceSample', default=False, type=bool)
    parser.add_argument('--precision', default=16, type=int)
    parser.add_argument('--specaug', default=True, type=bool)  # seems like it's not working with AngleLoss.
    parser.add_argument('--feature', default='kaggle', type=str, help='type of feature, option: kaggle, kesci')
    parser.add_argument('--vgg', default=True, type=bool, help='modification based on VoxCelb paper.')
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()

    train(args)
