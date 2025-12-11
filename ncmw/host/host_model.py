from __future__ import annotations

from dataclasses import dataclass
import os

import cobra

from ncmw.utils.utils_io import DATA_PATH


@dataclass
class HostBuildConfig:
    model_file: str
    add_lumen_compartment: bool = True
    host_compartment_id: str = "h"
    lumen_compartment_id: str = "lu"
    external_compartment_id: str = "e"
    host_lumen_transport_enabled: bool = True
    host_lumen_default_bound: float = 1000.0


def _resolve_model_path(model_file: str) -> str:
    if os.path.isabs(model_file):
        return model_file
    models_root = os.path.join(DATA_PATH, "models")
    return os.path.join(models_root, model_file)


def load_host_model(cfg: HostBuildConfig) -> cobra.Model:
    path = _resolve_model_path(cfg.model_file)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        return cobra.io.read_sbml_model(path)
    elif ext == ".json":
        return cobra.io.load_json_model(path)
    elif ext in (".yml", ".yaml"):
        return cobra.io.load_yaml_model(path)
    elif ext in (".mat", ".matlab"):
        return cobra.io.load_matlab_model(path)
    else:
        raise ValueError(f"Unsupported host model extension: {ext} (path={path})")


def add_lumen_compartment(model: cobra.Model, cfg: HostBuildConfig) -> None:
    if not cfg.add_lumen_compartment:
        return
    if cfg.lumen_compartment_id not in model.compartments:
        model.compartments[cfg.lumen_compartment_id] = "lumen"


def add_host_lumen_transport(model: cobra.Model, cfg: HostBuildConfig) -> None:
    if not cfg.host_lumen_transport_enabled:
        return

    for met in list(model.metabolites):
        if met.compartment != cfg.external_compartment_id:
            continue

        base_id = met.id
        ext = cfg.external_compartment_id
        if base_id.endswith(f"_{ext}"):
            lumen_id = f"{base_id[:-len(ext)]}{cfg.lumen_compartment_id}"
        else:
            lumen_id = f"{base_id}_{cfg.lumen_compartment_id}"

        if lumen_id in model.metabolites:
            lumen_met = model.metabolites.get_by_id(lumen_id)
        else:
            lumen_met = cobra.Metabolite(
                lumen_id,
                name=f"{met.name} (lumen)",
                compartment=cfg.lumen_compartment_id,
            )
            model.add_metabolites([lumen_met])

        rxn_id = f"T_HOST_LUMEN_{met.id}"
        if rxn_id in model.reactions:
            continue

        rxn = cobra.Reaction(rxn_id)
        rxn.name = f"host <-> lumen transport for {met.id}"
        rxn.lower_bound = -cfg.host_lumen_default_bound
        rxn.upper_bound = cfg.host_lumen_default_bound
        rxn.add_metabolites({met: -1.0, lumen_met: 1.0})
        model.add_reactions([rxn])


def build_host_lumen_model(cfg: HostBuildConfig) -> cobra.Model:
    model = load_host_model(cfg)
    add_lumen_compartment(model, cfg)
    add_host_lumen_transport(model, cfg)
    return model
