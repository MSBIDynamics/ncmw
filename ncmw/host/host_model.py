from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List

import cobra

from ncmw.utils.utils_io import DATA_PATH


@dataclass
class HostBuildConfig:
    """
    Configuration for building the host + lumen ("husk") model.
    """
    model_file: str  # relative to data/models or an absolute path

    add_lumen_compartment: bool = True
    host_compartment_id: str = "h"
    lumen_compartment_id: str = "lu"
    external_compartment_id: str = "e"

    host_lumen_transport_enabled: bool = True
    host_lumen_default_bound: float = 1000.0


# --------- Host-only helpers ---------


def _resolve_model_path(model_file: str) -> str:
    """Resolve host model path (relative to data/models or absolute)."""
    if os.path.isabs(model_file):
        return model_file
    models_root = os.path.join(DATA_PATH, "models")
    return os.path.join(models_root, model_file)


def load_host_model(cfg: HostBuildConfig) -> cobra.Model:
    """Load the base host model from SBML/JSON/YAML/MAT."""
    path = _resolve_model_path(cfg.model_file)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        model = cobra.io.read_sbml_model(path)
    elif ext == ".json":
        model = cobra.io.load_json_model(path)
    elif ext in (".yml", ".yaml"):
        model = cobra.io.load_yaml_model(path)
    elif ext in (".mat", ".matlab"):
        model = cobra.io.load_matlab_model(path)
    else:
        raise ValueError(f"Unsupported host model extension: {ext} (path={path})")

    return model


def add_lumen_compartment(model: cobra.Model, cfg: HostBuildConfig) -> None:
    """Create a lumen compartment if requested."""
    if not cfg.add_lumen_compartment:
        return

    if cfg.lumen_compartment_id not in model.compartments:
        model.compartments[cfg.lumen_compartment_id] = "lumen"


def add_host_lumen_transport(model: cobra.Model, cfg: HostBuildConfig) -> None:
    """
    Prototype: for every external metabolite, add reversible host<->lumen transport.
    """
    if not cfg.host_lumen_transport_enabled:
        return

    for met in list(model.metabolites):
        if met.compartment != cfg.external_compartment_id:
            continue

        base_id = met.id
        ext = cfg.external_compartment_id

        # derive lumen metabolite id
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
    """Load base host model and decorate it with lumen + transport."""
    model = load_host_model(cfg)
    add_lumen_compartment(model, cfg)
    add_host_lumen_transport(model, cfg)
    return model


# --------- Host + community merging helpers ---------


def merge_host_and_community_models(
    host_model: cobra.Model,
    community_model: cobra.Model,
    lumen_compartment_id: str,
    community_external_compartment_id: str = "e",
) -> cobra.Model:
    """
    Create a combined COBRA model with:
      - community reactions/metabolites
      - host reactions/metabolites (already with lumen)
      - community external metabolites mapped into the lumen compartment

    We assume the community model uses 'e' as external; we move them to 'lu'
    to represent a shared lumen environment.
    """

    combined = cobra.Model("HostCommunityModel")

    # 1) Normalize community external compartment to lumen
    for met in community_model.metabolites:
        if met.compartment == community_external_compartment_id:
            met.compartment = lumen_compartment_id

    # 2) Add community metabolites
    for met in community_model.metabolites:
        if met.id not in combined.metabolites:
            combined.add_metabolites([met.copy()])

    # 3) Add community reactions (prefix if collision)
    for rxn in community_model.reactions:
        rxn_to_add = rxn.copy()
        if rxn_to_add.id in combined.reactions:
            rxn_to_add.id = f"COMM_{rxn_to_add.id}"
        combined.add_reactions([rxn_to_add])

    # 4) Add host metabolites
    for met in host_model.metabolites:
        if met.id not in combined.metabolites:
            combined.add_metabolites([met.copy()])

    # 5) Add host reactions (prefix if collision)
    for rxn in host_model.reactions:
        rxn_to_add = rxn.copy()
        if rxn_to_add.id in combined.reactions:
            rxn_to_add.id = f"HOST_{rxn_to_add.id}"
        combined.add_reactions([rxn_to_add])

    return combined
