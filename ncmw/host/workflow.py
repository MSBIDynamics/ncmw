from __future__ import annotations

import logging
import os
import time
from dataclasses import fields as dataclass_fields

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
      - Otherwise: sum of all reactions whose ID contains 'biomass' or 'growth'.
    """
    biomass_rxn_id = host_cfg.get("biomass_reaction_id", None)

    if biomass_rxn_id is not None and biomass_rxn_id in model.reactions:
        model.objective = model.reactions.get_by_id(biomass_rxn_id)
        return

    biomass_rxns = [
        r
        for r in model.reactions
        if ("biomass" in r.id.lower()) or ("growth" in r.id.lower())
    ]
    if biomass_rxns:
        model.objective = {rxn: 1.0 for rxn in biomass_rxns}


# -------------------- Exchange routing (important) --------------------


def _is_simple_exchange(rxn: cobra.Reaction) -> bool:
    """
    Heuristic to detect exchange/boundary reactions.
    """
    try:
        if getattr(rxn, "boundary", False):
            return True
    except Exception:
        pass
    return rxn.id.startswith("EX_") and len(rxn.metabolites) == 1


def _make_lumen_met_id(met_id: str, lumen_compartment_id: str) -> str:
    """
    Convert a typical external metabolite id to a lumen id:
      glc__D_e -> glc__D_lu  (if lumen_compartment_id="lu")
    """
    if met_id.endswith("_e"):
        return met_id[:-2] + f"_{lumen_compartment_id}"
    return f"{met_id}_{lumen_compartment_id}"


def _make_lumen_ex_id(ex_id: str, lumen_compartment_id: str) -> str:
    """
    Convert a typical exchange id to a lumen exchange id:
      EX_glc__D_e -> EX_glc__D_lu
      HOST_EX_glc__D_e -> HOST_EX_glc__D_lu
    """
    if ex_id.endswith("_e"):
        return ex_id[:-2] + f"_{lumen_compartment_id}"
    return f"{ex_id}_{lumen_compartment_id}"


def _ensure_lumen_compartment(model: cobra.Model, lumen_compartment_id: str) -> None:
    if lumen_compartment_id not in model.compartments:
        model.compartments[lumen_compartment_id] = "lumen"


def _ensure_lumen_metabolite(
    model: cobra.Model, met: cobra.Metabolite, lumen_compartment_id: str
) -> cobra.Metabolite:
    lumen_met_id = _make_lumen_met_id(met.id, lumen_compartment_id)
    if lumen_met_id in model.metabolites:
        return model.metabolites.get_by_id(lumen_met_id)

    lumen_met = cobra.Metabolite(
        lumen_met_id,
        name=f"{met.name} (lumen)",
        compartment=lumen_compartment_id,
        formula=getattr(met, "formula", None),
        charge=getattr(met, "charge", None),
    )
    model.add_metabolites([lumen_met])
    return lumen_met


def _route_exchanges_through_lumen(
    model: cobra.Model,
    *,
    source_compartment_id: str,
    lumen_compartment_id: str,
) -> int:
    """
    Force all environmental exchange to happen via lumen.

    For every EX-like boundary reaction exchanging a metabolite in `source_compartment_id`:
      - create lumen metabolite (id converted to *_<lumen>)
      - create equivalent exchange reaction on the lumen metabolite with same bounds
      - CLOSE the old exchange (lb=ub=0)

    Returns: number of exchanges rerouted.
    """
    _ensure_lumen_compartment(model, lumen_compartment_id)

    rerouted = 0
    ex_rxns = [r for r in list(model.reactions) if _is_simple_exchange(r)]

    for ex in ex_rxns:
        if len(ex.metabolites) != 1:
            continue

        met = next(iter(ex.metabolites.keys()))
        if met.compartment != source_compartment_id:
            continue

        lumen_met = _ensure_lumen_metabolite(model, met, lumen_compartment_id)

        new_ex_id = _make_lumen_ex_id(ex.id, lumen_compartment_id)
        if new_ex_id not in model.reactions:
            coeff = ex.metabolites[met]  # typically -1.0
            new_ex = cobra.Reaction(new_ex_id)
            new_ex.name = (ex.name or ex.id) + " (lumen)"
            new_ex.lower_bound = ex.lower_bound
            new_ex.upper_bound = ex.upper_bound
            new_ex.add_metabolites({lumen_met: coeff})
            model.add_reactions([new_ex])

        # close the old exchange (prevents bypassing lumen)
        ex.lower_bound = 0.0
        ex.upper_bound = 0.0
        rerouted += 1

    return rerouted


def _ensure_community_lumen_transport(
    model: cobra.Model,
    *,
    community_external_compartment_id: str,
    lumen_compartment_id: str,
    default_bound: float,
    rxn_prefix: str = "T_COMM_LUMEN_",
) -> int:
    """
    Add mixing reactions between community pool (e.g. compartment 'external')
    and lumen (compartment 'lu').

    For each metabolite in community_external_compartment_id, ensure:
        T_COMM_LUMEN_<met.id>:  met_external <-> met_lu
    Bounds: [-default_bound, default_bound]
    """
    _ensure_lumen_compartment(model, lumen_compartment_id)

    added = 0
    mets = [m for m in model.metabolites if m.compartment == community_external_compartment_id]

    for met in mets:
        lumen_met = _ensure_lumen_metabolite(model, met, lumen_compartment_id)
        rxn_id = f"{rxn_prefix}{met.id}"
        if rxn_id in model.reactions:
            continue

        rxn = cobra.Reaction(rxn_id)
        rxn.name = f"community pool <-> lumen mixing for {met.id}"
        rxn.lower_bound = -float(default_bound)
        rxn.upper_bound = float(default_bound)
        rxn.add_metabolites({met: -1.0, lumen_met: 1.0})
        model.add_reactions([rxn])
        added += 1

    return added


# -------------------- Host workflow --------------------


def run_host_workflow(cfg: DictConfig) -> None:
    log.info("=== NCMW Host workflow ===")
    log.info(OmegaConf.to_yaml(cfg))

    start_time = time.time()

    project_name = cfg.name
    host_cfg = cfg.host

    # ---- Build HostBuildConfig robustly (works even if dataclass changed) ----
    field_names = {f.name for f in dataclass_fields(HostBuildConfig)}

    hl_cfg = host_cfg.get("host_lumen_transport", {})
    params: dict = {
        "model_file": host_cfg.get("host_model_file"),
        "add_lumen_compartment": host_cfg.get("add_lumen_compartment", True),
        "host_compartment_id": host_cfg.get("host_compartment_id", "h"),
        "lumen_compartment_id": host_cfg.get("lumen_compartment_id", "lu"),
        "external_compartment_id": host_cfg.get("external_compartment_id", "e"),
        # legacy names (prototype)
        "host_lumen_transport_enabled": hl_cfg.get("enabled", True),
        "host_lumen_default_bound": float(hl_cfg.get("default_bound", 1000.0)),
    }

    if "host_extracellular_compartment_id" in field_names:
        params["host_extracellular_compartment_id"] = host_cfg.get(
            "host_extracellular_compartment_id", "he"
        )

    # bounded host secretion/uptake (only if supported by HostBuildConfig)
    if "host_lumen_max_secretion" in field_names:
        params["host_lumen_max_secretion"] = float(hl_cfg.get("max_secretion", 10.0))
    if "host_lumen_max_uptake" in field_names:
        params["host_lumen_max_uptake"] = float(hl_cfg.get("max_uptake", 0.0))
    if "max_secretion" in field_names:
        params["max_secretion"] = float(hl_cfg.get("max_secretion", 10.0))
    if "max_uptake" in field_names:
        params["max_uptake"] = float(hl_cfg.get("max_uptake", 0.0))

    build_cfg_kwargs = {k: v for k, v in params.items() if k in field_names}
    build_cfg = HostBuildConfig(**build_cfg_kwargs)

    couple_to_community = bool(host_cfg.get("couple_to_community", False))
    community_rel_path = host_cfg.get("community_model_file", None)

    # -------- Build host (+ optional community) model --------
    if couple_to_community and community_rel_path:
        try:
            host_model = build_host_lumen_model(build_cfg)
            community_model = _load_community_model_from_results(project_name, community_rel_path)

            community_ext = host_cfg.get("community_external_compartment_id", "external")
            lumen_id = host_cfg.get("lumen_compartment_id", "lu")

            model = merge_host_and_community_models(
                host_model,
                community_model,
                lumen_compartment_id=lumen_id,
                community_external_compartment_id=community_ext,
            )

            # 1) IMPORTANT: enforce environment exchange through lumen (no bypass)
            rerouted_comm = _route_exchanges_through_lumen(
                model,
                source_compartment_id=community_ext,
                lumen_compartment_id=lumen_id,
            )

            # 2) IMPORTANT: ensure community pool <-> lumen mixing exists
            cl_cfg = host_cfg.get("community_lumen_transport", {})
            cl_enabled = bool(cl_cfg.get("enabled", False))
            cl_bound = float(cl_cfg.get("default_bound", 1000.0))
            added_mix = 0
            if cl_enabled:
                added_mix = _ensure_community_lumen_transport(
                    model,
                    community_external_compartment_id=community_ext,
                    lumen_compartment_id=lumen_id,
                    default_bound=cl_bound,
                )

            log.info(
                "Built combined Host+Community model using "
                f"community file '{community_rel_path}'. "
                f"Rerouted exchanges via lumen: community={rerouted_comm}. "
                f"Added community<->lumen mixing reactions: {added_mix}."
            )
        except FileNotFoundError as e:
            log.warning(
                f"{e} Falling back to host-only model. "
                "Set host.couple_to_community=false to avoid this warning."
            )
            model = build_host_lumen_model(build_cfg)
    else:
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
    if bool(host_cfg.get("run_fba", True)):
        solution = model.optimize()
        log.info(
            f"Host workflow FBA status: {solution.status}, "
            f"objective: {solution.objective_value}"
        )

    # -------- Outputs --------
    out_dir = _get_output_dir(cfg)

    if bool(host_cfg.get("write_sbml", True)):
        out_path = os.path.join(out_dir, "host_lumen_model.xml")
        cobra.io.write_sbml_model(model, out_path)
        log.info(f"Wrote host (or Host+Community) model to {out_path}")

    if bool(host_cfg.get("write_summary", True)):
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
