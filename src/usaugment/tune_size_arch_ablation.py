import logging
import os

import hydra
import lightning as L
import numpy as np
import optuna
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from usaugment.augmentation import get_data_loader_transforms
from usaugment.data import get_data_loader

SLURMEnvironment.detect = lambda: False

console_logger = logging.getLogger("lightning")


def objective(trial: optuna.Trial, config: DictConfig) -> float:
    # `suggest_float` is ignored when using GridSampler.
    # See https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html
    config.lr = trial.suggest_float("lr", 0, 1)
    config.weight_decay = trial.suggest_float("weight_decay", 0, 1)

    log_message = "Trial {:} started with parameters: lr = {:.3e}, weight_decay = {:.3e}".format(
        trial.number,
        config.lr,
        config.weight_decay,
    )
    console_logger.info(log_message)

    # Seed everything
    L.seed_everything(config.seed)

    # Configure transforms
    train_transform, test_transform = get_data_loader_transforms(config)

    # Configure data loaders
    train_loader = get_data_loader(
        config, "train", train_transform, shuffle=True
    )
    val_loader = get_data_loader(
        config, "validation", test_transform, shuffle=False
    )

    # Instantiate training module
    model = instantiate(config.model)

    # Setup the trainer
    checkpoint_callback = ModelCheckpoint(
        monitor=f"val/{config.key_metric}", mode="max", save_top_k=1
    )
    logger = CometLogger(
        project_name="usaugment-experiments",
        log_code=False,
        log_graph=False,
        auto_log_co2=False,
        log_env_details=False,
        offline=config.offline,
        save_dir=os.getcwd(),  # Hydra sets the working directory to the output directory
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    trainer = Trainer(
        max_epochs=config.epochs,
        deterministic=False,
        fast_dev_run=config.fast_dev_run,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        enable_progress_bar=config.enable_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        log_every_n_steps=len(train_loader),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    console_logger.info(
        f"Training complete. Best model score: {checkpoint_callback.best_model_score.item():.3f}"
    )

    return checkpoint_callback.best_model_score.item()


@hydra.main(version_base="1.3", config_path="config", config_name="tune")
def main(config: DictConfig) -> None:
    # Check config for missing keys
    if OmegaConf.missing_keys(config):
        raise RuntimeError(
            f"Got missing keys in config:\n{OmegaConf.missing_keys(config)}"
        )

    # Define the search space following https://arxiv.org/abs/1805.08974
    search_space = {
        "lr": np.logspace(-5, -1, num=7).tolist(),
        "weight_decay": np.logspace(-6, -3, num=7).tolist(),
    }

    # Run a single trial
    storage = JournalStorage(JournalFileBackend(config.optuna_log_path))
    study = optuna.create_study(
        storage=storage,
        study_name=config.optuna_study_name,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    study.optimize(lambda trial: objective(trial, config),
                   n_trials=config.trials_per_job)

    console_logger.info("Trial ended.")


if __name__ == "__main__":
    main()
