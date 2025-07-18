"""Load the best model checkpoints for each seed, evaluate them on the
test set, and report the results.
"""
import os

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from usaugment.augmentation import get_data_loader_transforms
from usaugment.data import get_data_loader

SLURMEnvironment.detect = lambda: False


@hydra.main(version_base="1.3", config_path="config", config_name="test")
def main(config: DictConfig) -> None:
    # Check config for missing keys
    if OmegaConf.missing_keys(config):
        raise RuntimeError(
            f"Got missing keys in config:\n{OmegaConf.missing_keys(config)}"
        )

    # Configure transforms
    _, test_transform = get_data_loader_transforms(config)

    # Configure the data loader
    test_loader = get_data_loader(
        config, "test", test_transform, shuffle=False)

    # Evaluate the model
    model = instantiate(config.model)

    logger = CometLogger(
        project_name="usaugment-experiments",
        log_code=False,
        log_graph=False,
        auto_log_co2=False,
        log_env_details=False,
        offline=config.offline,
        save_dir=os.getcwd(),  # Hydra sets the working directory to the output directory,
        display_summary_level=0,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    trainer = Trainer(
        max_epochs=config.epochs,
        deterministic=False,
        fast_dev_run=config.fast_dev_run,
        enable_progress_bar=config.enable_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        logger=logger,
    )

    trainer.test(model, ckpt_path=config.checkpoint,
                 dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    main()
