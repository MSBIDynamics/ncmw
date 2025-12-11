from __future__ import annotations

import logging
import os
import time

import cobra
from omegaconf import DictConfig, OmegaConf

from ncmw.utils.utils_io import get_result_path, SEPERATOR
from .host_model import (
    HostBuildConfig,
    build_host_lumen_model,
    merge_host_and_community_models,
)

log = logging.getLogger(__name__)


def _get_output_dir(cfg: DictConfig) -> str:
    """
    Compute the output folder for host results:
        results/<project_name>/<host.output_subdir or 'host'>
    """
    project_name = cfg.name
    base = get_result_path(project_name)
    host_subdir = cfg.host.output_subdir if "output_subdir" in cfg.host else "host"
    path = base + SEPERATOR + host_subdir
    os.makedirs(path, exist_ok=True)
    return path


def _load_community_model_from_results(project_name: str, rel_path: str) -> cobra.Model:
    """
    Load a community model stored under results/<project_name>/<rel_path>.

    Example rel_path (from host.yaml):
      community/community_models/ShuttleCommunityModel.xml
    """
    base = get_result_path(project_name)
    full_path = os.path.join(base, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Community model not found at '{full_path}'. "
            f"Make sure you ran the community workflow first."
        )
    return cobra.io.read_sbml_model(full_path)


def _set_objective(model: cobra.Model, host_cfg: DictConfig) -> None:
    """
    Set a reasonable objective for the host workflow:

      - If host_cfg.biomass_reaction_id is given and exists, use that.
      - Otherwise: sum of all reactions whose ID contains 'biomass'.
    """
    biomass_rxn_id = host_cfg.biomass_reaction_id

    if biomass_rxn_id is not None and biomass_rxn_id in model.reactions:
        model.objective = model.reactions.get_by_id(biomass_rxn_id)
        return

    # otherwise: sum of all biomass-like reactions
    biomass_rxns = [r for r in model.reactions if "biomass" in r.id.lower()]
    if biomass_rxns:
        model.objective = {rxn: 1.0 for rxn in biomass_rxns}


def run_host_workflow(cfg: DictConfig) -> None:
    """
    High-level entry point for the host/lumen ("husk") workflow.

    Two modes:

      1) Host-only:
         - build HostLumenModel (host + lumen, with host<->lumen transport)
         - run FBA
         - write SBML + summary

      2) Host+Community:
         - load community model from results/<project>/<community_model_file>
         - build HostLumenModel
         - merge them into a single COBRA model with shared lumen compartment
         - set objective (sum of biomass reactions by default)
         - run FBA
         - write SBML + summary

    The mode is controlled by cfg.host.couple_to_community (boolean) and
    cfg.host.community_model_file (relative path under results/<project>).
    """
    log.info("=== NCMW Host workflow ===")
    log.info(OmegaConf.to_yaml(cfg))

    start_time = time.time()

    project_name = cfg.name
    host_cfg = cfg.host
    hl_cfg = host_cfg.host_lumen_transport

    # BuildConfig for the host+lumen model
    build_cfg = HostBuildConfig(
        model_file=host_cfg.host_model_file,
        add_lumen_compartment=host_cfg.add_lumen_compartment,
        host_compartment_id=host_cfg.host_compartment_id,
        lumen_compartment_id=host_cfg.lumen_compartment_id,
        external_compartment_id=host_cfg.external_compartment_id,
        host_lumen_transport_enabled=hl_cfg.enabled,
        host_lumen_default_bound=float(hl_cfg.default_bound),
    )

    couple_to_community = bool(host_cfg.get("couple_to_community", False))
    community_rel_path = host_cfg.get("community_model_file", None)

    # -------- Build host (+ optional community) model --------
    if couple_to_community and community_rel_path:
        try:
            # 1) host+lumen
            host_model = build_host_lumen_model(build_cfg)
            # 2) community from results/<project>/<rel_path>
            community_model = _load_community_model_from_results(
                project_name, community_rel_path
            )
            # 3) merge into single HostCommunityModel
            community_ext = host_cfg.get("community_external_compartment_id", "external")

            model = merge_host_and_community_models(
                host_model,
                community_model,
                lumen_compartment_id=host_cfg.lumen_compartment_id,
                community_external_compartment_id=community_ext,
            )

            log.info(
                "Built combined Host+Community model using "
                f"community file '{community_rel_path}'."
            )
        except FileNotFoundError as e:
            # Safe fallback: if community model is not available, run host-only
            log.warning(
                f"{e} Falling back to host-only model. "
                "Set host.couple_to_community=false to avoid this warning."
            )
            model = build_host_lumen_model(build_cfg)
    else:
        # Host-only mode
        model = build_host_lumen_model(build_cfg)
        if couple_to_community:
            log.warning(
                "host.couple_to_community=true but no community_model_file was given. "
                "Running host-only model."
            )

    # -------- Objective --------
    _set_objective(model, host_cfg)

    # -------- Solve --------
    solution = None
    if host_cfg.run_fba:
        solution = model.optimize()
        log.info(
            f"Host workflow FBA status: {solution.status}, "
            f"objective: {solution.objective_value}"
        )

    # -------- Outputs --------
    out_dir = _get_output_dir(cfg)

    if host_cfg.write_sbml:
        out_path = os.path.join(out_dir, "host_lumen_model.xml")
        cobra.io.write_sbml_model(model, out_path)
        log.info(f"Wrote host (or Host+Community) model to {out_path}")

    if host_cfg.write_summary:
        summary = {
            "status": str(solution.status) if solution is not None else None,
            "objective_value": float(solution.objective_value) if solution is not None else None,
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
            "coupled_to_community": bool(couple_to_community and community_rel_path),
        }
        try:
            import yaml
        except ImportError:
            yaml = None

        if yaml is not None:
            with open(
                os.path.join(out_dir, "host_summary.yaml"),
                "w",
                encoding="utf-8",
            ) as f:
                yaml.safe_dump(summary, f)

    runtime = time.time() - start_time
    log.info(f"Finished host workflow in {runtime:.2f} seconds")
