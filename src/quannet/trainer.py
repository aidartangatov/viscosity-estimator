from pathlib import Path

import lightning.pytorch as pl
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from torch import optim

from quannet.config import get_config
from quannet.utils import DEFAULT_CONFIG, LOGGER, ROOT


class CustomLogger(Logger):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.logger.info(f'Step: {step}, {key}: {value}')

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(f'Hyperparameters: {params}')

    @property
    def experiment(self):
        return self.logger

    @property
    def name(self):
        return self.logger.name

    @property
    def version(self):
        return self.logger.version


class RegressionTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def _eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.huber_loss(y_hat, y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.model.args.lr)
        return optimizer


class Trainer:
    def __init__(self, model, train_loader, val_loader, config=DEFAULT_CONFIG, overrides=None):
        self.args = get_config(config, overrides)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        pl.seed_everything(42, workers=True)

        dirpath = Path(self.args.dirpath)
        if dirpath.is_absolute():
            self.dirpath = str(dirpath)
        else:
            self.dirpath = str(ROOT / dirpath)

        self._set_pl_trainer()

    def _set_pl_trainer(self):
        self.trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            callbacks=[self._early_stop_callback(), self._checkpoint_callback(self.dirpath)],
            deterministic=self.args.deterministic,
            logger=CustomLogger(LOGGER),
            fast_dev_run=True,
        )

    @staticmethod
    def _early_stop_callback():
        return EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')

    @staticmethod
    def _checkpoint_callback(dirpath):
        return ModelCheckpoint(
            dirpath=dirpath,
            filename='{epoch:02d}-{val_loss:.2f}',
            save_weights_only=True,
            monitor='val_loss',
            every_n_train_steps=10,
            mode='min',
            save_last=True,
        )

    def train(self):
        lit_model = RegressionTask(model=self.model)
        self.trainer.fit(lit_model, self.train_loader, self.val_loader)
