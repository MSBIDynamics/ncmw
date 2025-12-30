from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, List, Optional

import cobra

from ncmw.utils.utils_io import DATA_PATH


@dataclass
class HostBuildConfig:
    """
    Configuration for building the host + lumen model (paper-like topology).

    We assume the host SBML has some "external" compartment (often 'e').
    We relabel/move that to a dedicated host-extracellular compartment (default 'he'),
    then connect host-extracellular <-> lumen ('lu') with bounded shuttle reactions.
    """
    model_file: str  # relative to data/models or an absolute path

    # compartments
    host_compartment_id: str = "h"                # host cytosol (not heavily used here)
    external_compartment_id: str = "e"            # host SBML "external" compartment id
    host_extracellular_compartment_id: str = "he" # NEW: host extracellular
    lumen_compartment_id: str = "lu"              # NEW: lumen

    add_lumen_compartment: bool = True
    add_host_extracellular_compartment: bool = True

    # host<->lumen transport
    host_lumen_transport_enabled: bool = True

    # Backwards compat: if max_* are not provided, we fall back to +/- default_bound.
    host_lumen_default_bound: float = 1000.0

    # NEW biology: host is a constrained supplier
    host_lumen_max_secretion: Optional[float] = None  # he -> lu (positive flux)
    host_lumen_max_uptake: Optional[float] = None     # lu -> he (negative flux)

    # Optional: restrict which metabolites are allowed to shuttle
    allowed_metabolites: Optional[List[str]] = None   # list of metabolite IDs in host extracellular (old ids are OK)


# --------- helpers ---------


def _resolve_model_path(model_file: str) -> str:
    """Resolve host model path (relative to data/models or absolute)."""
    if os.path.isabs(model_file):
        return model_file
    models_root = os.path.join(DATA_PATH, "models")
    return os.path.join(models_root, model_file)


def _mapped_met_id(met_id: str, from_comp: str, to_comp: str) -> str:
    """
    Convert metabolite id suffix from _{from_comp} to _{to_comp} if possible,
    else append _{to_comp}.
    Mirrors your existing style where ids are usually like glc__D_e or glc__D_external.
    """
    suffix = f"_{from_comp}"
    if met_id.endswith(suffix):
        # remove only 'from_comp' length (keeps underscore) like your previous implementation
        return f"{met_id[:-len(from_comp)]}{to_comp}"
    return f"{met_id}_{to_comp}"


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


def add_host_extracellular_compartment(model: cobra.Model, cfg: HostBuildConfig) -> None:
    """
    Create host extracellular compartment (he) and move metabolites from the host SBML
    external compartment (e) into (he). This preserves reaction connectivity.
    """
    if not cfg.add_host_extracellular_compartment:
        return

    if cfg.host_extracellular_compartment_id not in model.compartments:
        model.compartments[cfg.host_extracellular_compartment_id] = "host extracellular"

    # Move all metabolites that are currently in the host SBML external compartment to host extracellular.
    for met in model.metabolites:
        if met.compartment == cfg.external_compartment_id:
            met.compartment = cfg.host_extracellular_compartment_id

    # Optional: keep the old compartment name in the map (doesn't hurt), but it's now unused.
    if cfg.external_compartment_id not in model.compartments:
        model.compartments[cfg.external_compartment_id] = "host external (legacy)"


def add_lumen_compartment(model: cobra.Model, cfg: HostBuildConfig) -> None:
    """Create a lumen compartment if requested."""
    if not cfg.add_lumen_compartment:
        return

    if cfg.lumen_compartment_id not in model.compartments:
        model.compartments[cfg.lumen_compartment_id] = "lumen"


def add_host_lumen_transport(model: cobra.Model, cfg: HostBuildConfig) -> None:
    """
    Add shuttle reactions between host extracellular (he) and lumen (lu).

    IMPORTANT: This is NOT infinite anymore:
      - If cfg.host_lumen_max_secretion/max_uptake are provided:
            bounds = [-max_uptake, +max_secretion]
      - Else fallback to legacy +/- cfg.host_lumen_default_bound
    """
    if not cfg.host_lumen_transport_enabled:
        return

    allowed = set(cfg.allowed_metabolites) if cfg.allowed_metabolites else None

    # Determine bounds
    if cfg.host_lumen_max_secretion is None and cfg.host_lumen_max_uptake is None:
        lb = -float(cfg.host_lumen_default_bound)
        ub = float(cfg.host_lumen_default_bound)
    else:
        max_sec = float(cfg.host_lumen_max_secretion or 0.0)
        max_upt = float(cfg.host_lumen_max_uptake or 0.0)
        lb = -max_upt
        ub = max_sec

    for met in list(model.metabolites):
        if met.compartment != cfg.host_extracellular_compartment_id:
            continue
        if allowed is not None and met.id not in allowed:
            continue

        lumen_id = _mapped_met_id(
            met.id,
            from_comp=cfg.external_compartment_id,   # use original suffix convention if present
            to_comp=cfg.lumen_compartment_id,
        )

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
            # update bounds to new logic (in case of reruns)
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = lb
            rxn.upper_bound = ub
            continue

        rxn = cobra.Reaction(rxn_id)
        rxn.name = f"host extracellular <-> lumen shuttle for {met.id}"
        rxn.lower_bound = lb
        rxn.upper_bound = ub
        # Positive flux: host extracellular -> lumen
        rxn.add_metabolites({met: -1.0, lumen_met: 1.0})
        model.add_reactions([rxn])


def build_host_lumen_model(cfg: HostBuildConfig) -> cobra.Model:
    """Load base host model and decorate it with host-extracellular + lumen + shuttles."""
    model = load_host_model(cfg)
    add_host_extracellular_compartment(model, cfg)
    add_lumen_compartment(model, cfg)
    add_host_lumen_transport(model, cfg)
    return model


# --------- Host + community merging helpers ---------


def add_community_lumen_transport(
    combined: cobra.Model,
    lumen_compartment_id: str,
    community_external_compartment_id: str = "external",
    default_bound: float = 1000.0,
) -> None:
    """
    Add reversible "mixing" shuttles between community external (external) and lumen (lu).
    This preserves compartments like in the paper, but makes them effectively well-mixed.
    """
    # Ensure lumen compartment exists in combined map
    if lumen_compartment_id not in combined.compartments:
        combined.compartments[lumen_compartment_id] = "lumen"

    # Iterate over community-external metabolites present in combined
    comm_ext_mets = [m for m in combined.metabolites if m.compartment == community_external_compartment_id]

    for comm_met in comm_ext_mets:
        lumen_id = _mapped_met_id(
            comm_met.id,
            from_comp=community_external_compartment_id,
            to_comp=lumen_compartment_id,
        )

        if lumen_id in combined.metabolites:
            lumen_met = combined.metabolites.get_by_id(lumen_id)
        else:
            lumen_met = cobra.Metabolite(
                lumen_id,
                name=f"{comm_met.name} (lumen)",
                compartment=lumen_compartment_id,
            )
            combined.add_metabolites([lumen_met])

        rxn_id = f"T_COMM_LUMEN_{comm_met.id}"
        if rxn_id in combined.reactions:
            continue

        rxn = cobra.Reaction(rxn_id)
        rxn.name = f"community external <-> lumen mixing for {comm_met.id}"
        rxn.lower_bound = -float(default_bound)
        rxn.upper_bound = float(default_bound)
        rxn.add_metabolites({comm_met: -1.0, lumen_met: 1.0})
        combined.add_reactions([rxn])


def merge_host_and_community_models(
    host_model: cobra.Model,
    community_model: cobra.Model,
    lumen_compartment_id: str,
    community_external_compartment_id: str = "external",
    add_comm_lumen_transport: bool = True,
    community_lumen_default_bound: float = 1000.0,
) -> cobra.Model:
    """
    Create a combined COBRA model where compartments remain distinct:
      - host extracellular (he)
      - lumen (lu)
      - community external (external)

    We DO NOT rename community external to lumen anymore.
    Instead we add mixing reactions external <-> lumen.
    """
    combined = cobra.Model("HostCommunityModel")

    # Carry over compartment maps (helpful for SBML write)
    combined.compartments.update(community_model.compartments)
    combined.compartments.update(host_model.compartments)
    if lumen_compartment_id not in combined.compartments:
        combined.compartments[lumen_compartment_id] = "lumen"

    # 1) Add community metabolites
    for met in community_model.metabolites:
        if met.id not in combined.metabolites:
            combined.add_metabolites([met.copy()])

    # 2) Add community reactions
    for rxn in community_model.reactions:
        rxn_to_add = rxn.copy()
        if rxn_to_add.id in combined.reactions:
            rxn_to_add.id = f"COMM_{rxn_to_add.id}"
        combined.add_reactions([rxn_to_add])

    # 3) Add host metabolites
    for met in host_model.metabolites:
        if met.id not in combined.metabolites:
            combined.add_metabolites([met.copy()])

    # 4) Add host reactions
    for rxn in host_model.reactions:
        rxn_to_add = rxn.copy()
        if rxn_to_add.id in combined.reactions:
            rxn_to_add.id = f"HOST_{rxn_to_add.id}"
        combined.add_reactions([rxn_to_add])

    # 5) Add external <-> lumen mixing for the community environment
    if add_comm_lumen_transport:
        add_community_lumen_transport(
            combined,
            lumen_compartment_id=lumen_compartment_id,
            community_external_compartment_id=community_external_compartment_id,
            default_bound=community_lumen_default_bound,
        )

    return combined
