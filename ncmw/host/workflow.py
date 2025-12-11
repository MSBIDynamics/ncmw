from __future__ import annotations

import logging
import os
import time

import cobra
from omegaconf import DictConfig, OmegaConf

from ncmw.utils.utils_io import get_result_path, SEPERATOR
from .host_model import HostBuildConfig, build_host_lumen_model

log = logging.getLogger(__name__)


def _get_output_dir(cfg: DictConfig) -> str:
    project_name = cfg.name
    base = get_result_path(project_name)
    host_subdir = cfg.host.output_subdir if "output_subdir" in cfg.host else "host"
    path = base + SEPERATOR + host_subdir
    os.makedirs(path, exist_ok=True)
    return path


def run_host_workflow(cfg: DictConfig) -> None:
    log.info("=== NCMW Host workflow ===")
    log.info(OmegaConf.to_yaml(cfg))

    start_time = time.time()

    host_cfg = cfg.host
    hl_cfg = host_cfg.host_lumen_transport

    build_cfg = HostBuildConfig(
        model_file=host_cfg.host_model_file,
        add_lumen_compartment=host_cfg.add_lumen_compartment,
        host_compartment_id=host_cfg.host_compartment_id,
        lumen_compartment_id=host_cfg.lumen_compartment_id,
        external_compartment_id=host_cfg.external_compartment_id,
        host_lumen_transport_enabled=hl_cfg.enabled,
        host_lumen_default_bound=float(hl_cfg.default_bound),
    )

    model = build_host_lumen_model(build_cfg)

    biomass_rxn_id = host_cfg.biomass_reaction_id
    if biomass_rxn_id is not None and biomass_rxn_id in model.reactions:
        model.objective = model.reactions.get_by_id(biomass_rxn_id)
    else:
        for rxn in model.reactions:
            if "biomass" in rxn.id.lower():
                model.objective = rxn
                break

    solution = None
    if host_cfg.run_fba:
        solution = model.optimize()
        log.info(f"Host FBA status: {solution.status}, objective: {solution.objective_value}")

    out_dir = _get_output_dir(cfg)

    if host_cfg.write_sbml:
        out_path = os.path.join(out_dir, "host_lumen_model.xml")
        cobra.io.write_sbml_model(model, out_path)
        log.info(f"Wrote host+lumen model to {out_path}")

    if host_cfg.write_summary:
        summary = {
            "status": str(solution.status) if solution is not None else None,
            "objective_value": float(solution.objective_value) if solution is not None else None,
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
        }
        try:
            import yaml
        except ImportError:
            yaml = None

        if yaml is not None:
            with open(os.path.join(out_dir, "host_summary.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(summary, f)

    runtime = time.time() - start_time
    log.info(f"Finished host workflow in {runtime:.2f} seconds")
