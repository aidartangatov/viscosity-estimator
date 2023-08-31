from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from quannet.tasks import LitModel, QuanModel
from quannet.utils import LOGGER, SETTINGS, DEFAULT_CONFIG, check_dataset, increment_path
from quannet.config import get_config
from quannet.dataset import build_input


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


class QuanTrainer:
    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        self.args = get_config(config, overrides)
        self.model = None
        self.best = None

        project = self.args.project or Path(SETTINGS['runs_dir'])
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = Path(increment_path(Path(project) / name, exist_ok=self.args.exist_ok))
        self.artefacts_dir = self.save_dir / SETTINGS['arefacts_dir_name']
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
            train_val_split=True,
            args=self.args,
        )
        LOGGER.info('Finished generating model input')

        self.train_loader = train_loader
        self.val_loader = val_loader
        pl.seed_everything(42, workers=True)

        self._set_pl_trainer()

    def _set_pl_trainer(self):
        self.checkpoint_callback = self._checkpoint_callback(self.save_dir)
        self.trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            callbacks=[self._early_stop_callback(), self.checkpoint_callback],
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
        lit_model = LitModel(model=self.model, args=self.args)
        self.trainer.fit(lit_model, self.train_loader, self.val_loader)
        best_model_path = str(self.save_dir / 'best_model.ckpt')
        self.trainer.save_checkpoint(best_model_path)
        self.best = best_model_path

    def get_model(self, model: QuanModel, weights=None, verbose=True):
        if weights:
            model.load(weights, verbose=verbose)
        return model

    # def save_model(self):
    #     """Save model checkpoints based on various conditions."""
    #     ckpt = {
    #         'epoch': self.epoch,
    #         'best_fitness': self.best_fitness,
    #         'model': deepcopy(de_parallel(self.model)).half(),
    #         'updates': self.ema.updates,
    #         'optimizer': self.optimizer.state_dict(),
    #         'train_args': vars(self.args),  # save as dict
    #         'date': datetime.now().isoformat(),
    #         'version': __version__}
    #
    #     # Save last, best and delete
    #     torch.save(ckpt, self.last)
    #     if self.best_fitness == self.fitness:
    #         torch.save(ckpt, self.best)
    #     if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
    #         torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    #     del ckpt
