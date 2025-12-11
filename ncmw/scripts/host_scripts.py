import hydra
from omegaconf import DictConfig, OmegaConf

import logging
import socket
import time
import random
import numpy as np
import os

from ncmw.host.workflow import run_host_workflow


def run_host(cfg: DictConfig) -> None:
    """Host script called by Hydra or by the full `ncmw` workflow."""
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Hostname: {socket.gethostname()}")

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    log.info(f"Random seed: {seed}")

    start_time = time.time()
    run_host_workflow(cfg)
    runtime = time.time() - start_time
    log.info(f"Finished host workflow in {runtime:.2f} seconds")


@hydra.main(
    config_path=os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/data/hydra",
    config_name="config.yaml",      # <--- IMPORTANT: use config.yaml
    version_base=None,
)
def run_host_hydra(cfg: DictConfig) -> None:
    """Entry point for the dedicated `ncmw_host` console script."""
    run_host(cfg)
