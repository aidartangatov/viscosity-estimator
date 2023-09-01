from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from quannet.tasks import LitModel, QuanModel
from quannet.utils import LOGGER, SETTINGS, DEFAULT_CONFIG, check_dataset, increment_path
from quannet.config import get_config
from quannet.dataset import build_input


class QuanTrainer:
    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        self.args = get_config(config, overrides)
        self.model = None
        self.best = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None

        project = self.args.project or Path(SETTINGS['runs_dir'])
        name = self.args.name or f'{self.args.mode}'
        if self.args.precomputed_input:
            self.save_dir = Path(project) / name
        else:
            self.save_dir = Path(increment_path(Path(project) / name, exist_ok=self.args.exist_ok))
        self.artefacts_dir = self.save_dir / SETTINGS['artefacts_dir_name']
        self._set_pl_trainer()

    def _set_pl_trainer(self):
        self.checkpoint_callback = self._checkpoint_callback(self.save_dir)
        self.trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            log_every_n_steps=1,
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            callbacks=[self._early_stop_callback(), self.checkpoint_callback],
            deterministic=self.args.deterministic,
            logger=CSVLogger(save_dir=self.save_dir, name='logs'),
        )

    @staticmethod
    def _early_stop_callback():
        return EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')

    @staticmethod
    def _checkpoint_callback(dirpath):
        return ModelCheckpoint(
            dirpath=dirpath,
            filename='{epoch:02d}-{val_loss:.2f}',
            save_weights_only=False,
            monitor='val_loss',
            every_n_train_steps=1,
            save_top_k=1,
            mode='min',
            save_last=True,
        )

    def _prepare_input(self):
        self.dataset = check_dataset(
            self.args.dataset,
            path_csv_col=self.args.path_csv_col,
            target_csv_col=self.args.target_csv_col,
            min_structures=1,
        )

        LOGGER.info('Started generating model input')
        train_loader, val_loader = build_input(
            self.dataset['paths'],
            self.dataset['targets'],
            artefacts_dir=self.artefacts_dir,
            precomputed=self.args.precomputed_input,
            train_val_split=True,
            args=self.args,
        )
        LOGGER.info('Finished generating model input')

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        pl.seed_everything(42, workers=True)
        self._prepare_input()
        lit_model = LitModel(model=self.model, args=self.args)
        self.trainer.fit(lit_model, self.train_loader, self.val_loader)
        best_model_path = str(self.save_dir / 'best_model.ckpt')
        self.trainer.save_checkpoint(best_model_path)
        self.best = best_model_path

    def get_model(self, model: QuanModel, weights=None, verbose=True):
        if weights:
            model.load(weights, verbose=verbose)
        return model
