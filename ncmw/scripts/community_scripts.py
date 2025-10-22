import hydra
from ncmw import community
from omegaconf import DictConfig, OmegaConf
import cobra

import logging
import socket

import time

import random
import numpy as np
import pandas as pd
from copy import deepcopy

import sys, os
import json, pickle
from ncmw.community.community_models import HierarchicalCommunityModel
from ncmw.community.hierarchical_model_composition import generate_hierarchical_community_model

file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)


from ncmw.utils.utils_io import (
    get_models,
    get_result_path,
    SEPERATOR,
    save_model,
    get_model_paths,
)

from ncmw.visualization import (
    plot_pairwise_growth_relation_per_weight,
    plot_community_interaction,
    plot_posterior_samples_for_observation,
    plot_community_uptake_graph,
    plot_species_interaction,
    plot_community_summary,
    plot_weight_growth_pairplot,
)

from ncmw.utils.utils_io import (
    get_models,
    get_result_path,
    SEPERATOR,
    save_model,
    get_model_paths,
    get_mediums,
)
from ncmw.community import (
    BagOfReactionsModel,
    ShuttleCommunityModel,
    compute_fair_weights,
    compute_community_summary,
    compute_dominant_weights,
    community_weight_posterior,
)

from ncmw.utils import (
    get_models,
    get_result_path,
    check_for_substring_in_folder,
    SEPERATOR,
)


@hydra.main(config_path="../../data/hydra", config_name="config.yaml")
def run_community_hydra(cfg: DictConfig) -> None:
    run_community(cfg)

def create_community_folder_backbone(community_folder:str, project_folder:str) -> None:
    """Creates the backbone folder system used by the script
    
    Args:
        community_folder: Path to the community folder, here contents is placed
        project_folder: Path to the parent directory

    """
    try:
        if not os.path.exists(project_folder):
            os.mkdir(project_folder)
        if not os.path.exists(community_folder):
            os.mkdir(community_folder)
        if not os.path.exists(community_folder + SEPERATOR + "community_models"):
            os.mkdir(community_folder + SEPERATOR + "community_models")
        if not os.path.exists(community_folder + SEPERATOR + "medium"):
            os.mkdir(community_folder + SEPERATOR + "medium")
        if not os.path.exists(community_folder + SEPERATOR + "weight_inference"):
            os.mkdir(community_folder + SEPERATOR + "weight_inference")
        if not os.path.exists(community_folder + SEPERATOR + "experiments_BagOfReactionsModel"):
            os.mkdir(community_folder + SEPERATOR + "experiments_BagOfReactionsModel")
        if not os.path.exists(community_folder + SEPERATOR + "experiments_ShuttleCommunityModel"):
            os.mkdir(community_folder + SEPERATOR + "experiments_ShuttleCommunityModel")
        if not os.path.exists(community_folder + SEPERATOR + "experiments_HierarchicalCommunityModel"):
            os.mkdir(community_folder + SEPERATOR + "experiments_HierarchicalCommunityModel")
    except:
        raise ValueError("Could not generate output directory, maybe we do not have permission's to do so?")
        
def load_old_configs(community_folder:str, cfg:str) -> DictConfig:
    """This will load the old config file to avoid recomputation.
    
    Args:
        community_folder: Folder in which the community results are placed.
        cfg: Current config file to dump (is after this run the new old_config).
    
    Returns:
        DictConfig: Old configurations.
    
    """
    if os.path.exists(community_folder + SEPERATOR + ".configs"):
        with open(community_folder + SEPERATOR + ".configs", "rb") as f:
            old_cfg = pickle.load(f)
    else:
        old_cfg = cfg

    with open(community_folder + SEPERATOR + ".configs", "wb") as f:
        pickle.dump(cfg, f)
    return old_cfg

def _as_flux_df(view):
    """
    Return a DataFrame with a single column 'fluxes' indexed by reaction id,
    regardless of whether `view` has .to_frame() or is dict/FluxView-like.
    """
    if hasattr(view, "to_frame"):
        df = view.to_frame()
        # make sure the expected column name exists
        if "fluxes" not in df.columns and df.shape[1] == 1:
            df.columns = ["fluxes"]
        return df

    # Try mapping-like
    try:
        items = list(view.items())
    except Exception:
        # Some FluxView variants expose .fluxes (mapping)
        items = list(getattr(view, "fluxes").items())

    import pandas as pd  # safe; already imported at top, but ok if present
    df = pd.DataFrame(items, columns=["reaction", "fluxes"]).set_index("reaction")
    return df


def set_solver_disable_functionalities_if_needed(cfg: DictConfig, log: logging.Logger) -> DictConfig:
    """Pick the best available solver and disable unsupported features if needed.

    Preference order: GUROBI → CPLEX → GLPK → GLPK-Exact → SCIPY.
    - Sets cobra.Configuration().solver to the first available option.
    - Stores the chosen solver name in cfg.solver for downstream logging.
    - Disables features known to be unsupported/slow on non-Gurobi/Cplex solvers:
      * compute_community_fva
      * cooperative_tradeoff

    Args:
        cfg: Hydra config object.
        log: Logger.

    Returns:
        DictConfig: Updated config (with cfg.solver set and features possibly disabled).
    """
    cobra_config = cobra.Configuration()
    fallback_order = ["gurobi", "cplex", "glpk", "glpk_exact", "scipy"]

    chosen = None
    last_err = None
    for s in fallback_order:
        try:
            cobra_config.solver = s
            chosen = s
            break
        except Exception as e:
            # Keep the log style you were using earlier
            name = s.upper()
            if s in ("gurobi", "cplex"):
                log.info(f"{name} unavailable ({e}). Trying next...")
            last_err = e

    if chosen is None:
        raise RuntimeError(
            f"No supported solver available. Tried: {', '.join(fallback_order)}. "
            f"Last error: {last_err}"
        )

    # Persist the choice in cfg.solver (handle struct mode safely)
    try:
        cfg.solver = chosen
    except Exception:
        OmegaConf.set_struct(cfg, False)
        cfg.solver = chosen
        OmegaConf.set_struct(cfg, True)

    # Announce choice in the same voice as your current logs
    if chosen == "gurobi":
        log.info("Using GUROBI solver.")
    elif chosen == "cplex":
        log.info("Using CPLEX solver.")
    elif chosen == "glpk_exact":
        log.info("Using GLPK-Exact solver.")
    elif chosen == "glpk":
        log.info("Using GLPK solver.")
    else:
        log.info("Using SCIPY solver.")

    # Only Gurobi/CPLEX keep all features enabled
    if chosen not in ("gurobi", "cplex"):
        if getattr(cfg.community, "compute_community_fva", False):
            log.warning(f"Disabling compute_community_fva for solver {chosen.upper()}.")
            cfg.community.compute_community_fva = False
        if getattr(cfg.community, "cooperative_tradeoff", False):
            log.warning(f"Disabling cooperative_tradeoff for solver {chosen.upper()}.")
            cfg.community.cooperative_tradeoff = False

    return cfg
    
def generate_community_models(models, cfg, old_cfg, log, community_path):
    log.info("Generating community models")
    community_models = []
    if cfg.community.bag_of_reactions_model:
        path = (
            community_path
            + SEPERATOR
            + "community_models"
            + SEPERATOR
            + "BagOfReactionsModel.pkl"
        )
        # Either loading or reinitializing the community model
        if (
            check_for_substring_in_folder(
                community_path + SEPERATOR + "community_models", "BagOfReactionsModel.pkl"
            )
            and old_cfg.community.models_folder == cfg.community.models_folder
        ):
            log.info("Loading BagOfReactions community model")
            m = BagOfReactionsModel.load(path)
            ids = [mod.id for mod in m.models]
            correct = True
            for model in models:
                correct = correct and (model.id in ids)
            if correct:
                community_models += [m]
            else:
                community_models += [BagOfReactionsModel(models)]
        else:
            log.info("Building BagOfReactions Community model")
            community_models += [BagOfReactionsModel(models)]
    if cfg.community.shuttle_reactions_model:
        path = (
            community_path
            + SEPERATOR
            + "community_models"
            + SEPERATOR
            + "ShuttleCommunityModel.pkl"
        )
        kwargs = cfg.community.shuttle_reaction_params
        if not isinstance(kwargs["shared_reactions"], list):
            kwargs["shared_reactions"] = None
            old_cfg.community.shuttle_reaction_params["shared_reactions"] = None
        if (
            check_for_substring_in_folder(
                community_path + SEPERATOR + "community_models", "ShuttleCommunityModel.pkl"
            )
            and old_cfg.community.models_folder == cfg.community.models_folder
            and all(
                [
                    old_cfg.community.shuttle_reaction_params[key]
                    == cfg.community.shuttle_reaction_params[key]
                    for key in cfg.community.shuttle_reaction_params
                ]
            )
        ):
            log.info("Loading shuttle reactions model")
            m = ShuttleCommunityModel.load(path)
            ids = [mod.id for mod in models]
            correct = True
            for model in models:
                correct = correct and (model.id in ids)
            if correct:
                community_models += [m]
            else:
                community_models += [ShuttleCommunityModel(models, **kwargs)]
        else:
            log.info("Building shuttle reactions model")
            community_models += [ShuttleCommunityModel(models, **kwargs)]
    if cfg.community.hierarchical_model:
            log.info("Building hierarchical model")
            kwargs = cfg.community.shuttle_reaction_params
            model_paths = get_model_paths(cfg.setup.models)
            model = HierarchicalCommunityModel(model_paths, **kwargs)
            community_models += [model]

    log.info(f"Set correct weights: {cfg.community.main_weights}")
    for m in community_models:
        if cfg.community.main_weights == "ones":
            weights = np.ones(len(m.models))
        elif cfg.community.main_weights == "fair":
            weights = compute_fair_weights(m)
        else:
            weights = cfg.community.main_weights
            assert len(weights) == len(
                m.models
            ), "The custom main weights must be a iterable of floats for each member of the community!"
        m.weights = weights
    return community_models

def reset(old_community_model, PATH):
    new_model = type(old_community_model).load(PATH + SEPERATOR + "community_models" + SEPERATOR + str(type(old_community_model).__name__) + ".pkl")
    new_model.medium = old_community_model.medium 
    new_model.weights = old_community_model.weights
    return new_model
    

def generate_medias(community_model, cfg:DictConfig, log, community_path, result_path):
    all_medias = {}
    default = set(get_mediums("medium")[cfg.setup.medium].keys())

    if cfg.community.medium["default"]:
        log.info("Computing default for community")
        # Also save default medium
        medium = community_model.medium
        if cfg.community.medium_strict_subset_of_default:
            medium = dict([(k,v) for k,v in medium.items() if k in default])
        all_medias["default"] = medium
        path = community_path + SEPERATOR + "medium" + SEPERATOR + "DEFAULT" + ".json"
        with open(path, "w+") as f:
            json.dump(community_model.medium, f)
        log.info(f"Saving default: {path}")


    if cfg.community.medium["compm"] or cfg.community.medium["coopm"]:
        log.info("Computing COMPM for community")
        medium_prefix = result_path + SEPERATOR + "analysis" + SEPERATOR
        mediums = get_mediums("medium", medium_prefix)
        COMPM = dict()
        for medium in mediums.values():
            for key, val in medium.items():
                COMPM[key] = float(abs(val))
        all_medias["compm"] = COMPM
        if cfg.community.medium_strict_subset_of_default:
            COMPM = dict([(k,v) for k,v in COMPM.items() if k in default])
        if cfg.community.medium["compm"]:
            path = community_path + SEPERATOR + "medium" + SEPERATOR + "COMPM" + ".json"
            with open(path, "w+") as f:
                json.dump(COMPM, f)
            log.info(f"Saving COMPMs: {path}")

    return all_medias

def generate_coopm_medium(community_model, all_medias, cfg, log:logging.Logger, community_path):
    log.info("Computing COOPM medium")
    # COOPM computed using COMPM as base medium!
    community_model.medium = all_medias["compm"]
    mbr = community_model.slim_optimize()
    if cfg.community.coopm_params["enforce_survival"] > 0:
        for i in range(len(community_model.models)):
            if community_model.single_optimize(i) == 0:
                cfg.community.coopm_params["enforce_survival"] = 0
                log.warning(
                    "We set enforce survival = 0, as not all models within the community can grow!"
                )

    log.info(f"Growth on COMPM: {mbr}")
    coopm = community_model.compute_COOPM(mbr, **cfg.community.coopm_params)
    all_medias["coopm"] = coopm
    path = community_path + SEPERATOR + "medium" + SEPERATOR + f"{type(community_model).__name__}_COOPM" + ".json"
    with open(path, "w+") as f:
        json.dump(coopm, f)
    log.info(f"Saving COOPMS: {path}")
    return coopm

def flux_analysis(community_model, models, medium_name, cfg, log, PATH, path_to_save):
    log.info(f"Community FBA/FVA results on medium: {medium_name}")
    df_growth_summary = pd.DataFrame()

    if cfg.community.compute_community_fva:
        # On Windows with Gurobi, parallel FVA (processes>1) can break due to LP serialization.
        try:
            import platform
            iface = getattr(getattr(community_model, "community_model", community_model).solver.interface, "__name__", "").lower()
            if platform.system() == "Windows" and "gurobi" in iface:
                if getattr(cfg.community.community_fva_params, "processes", 1) != 1:
                    log.warning("Forcing FVA to run single-process on Windows + Gurobi to avoid LP read errors.")
                    cfg.community.community_fva_params.processes = 1
        except Exception:
            pass

        log.info("Computing FVA")

        fraction_of_optimal = cfg.community.cooperative_tradeoff_params["alpha"]
        try:
            # This may fail for several numerical reasions -> Infeasible Error from solver.
            df_fva = community_model.flux_variability_analysis(
                **cfg.community.community_fva_params
            )
            df_fva.columns = [
                f"minimum (fraction of optimal: {fraction_of_optimal})",
                f"maximum (fraction of optimal: {fraction_of_optimal})",
            ]
        except:
            log.warning("Flux variability analysis failed, we will disable it.")
            cfg.community.compute_community_fva= False
            df_fva = pd.DataFrame()
            community_model = reset(community_model, PATH)

    else:
        df_fva = pd.DataFrame()
   
    # FBA results
    growth, single_growths, sol = community_model.optimize(**cfg.community.optimize_params)
    log.info(
        f"Achieved community growth:{growth}, with individual growth: {single_growths}"
    )
    df_growth_summary["Weights"] = list(community_model.weights) + [None]
    df_growth_summary["FBA Growth"] = single_growths + [growth]
    df_fba = _as_flux_df(sol)
    # Cooperative tradeoff results (only if solver is QP-capable)
    iface = getattr(
        getattr(community_model, "community_model", community_model).solver.interface,
        "__name__",
        ""
    ).lower()
    qp_ok = ("gurobi" in iface) or ("cplex" in iface)

    if cfg.community.cooperative_tradeoff and not qp_ok:
        log.warning(
            "Skipping cooperative_tradeoff: solver '%s' is not QP-capable (requires Gurobi or CPLEX).",
            iface,
        )
        df_fba_cooperative_tradeoff = None
    elif cfg.community.cooperative_tradeoff:
        (
            growth_tradeoff,
            single_growths_tradeoff,
            sol_tradeoff,
        ) = community_model.cooperative_tradeoff(**cfg.community.cooperative_tradeoff_params)

        log.info(
            "Achieved community growth (cooperative tradeoff): %s, with individual growth: %s",
            growth_tradeoff, single_growths_tradeoff,
        )
        df_growth_summary[f"Cooperative tradeoff (alpha: {cfg.community.cooperative_tradeoff_params.alpha})"] = single_growths_tradeoff + [growth_tradeoff]
        df_fba_cooperative_tradeoff = _as_flux_df(sol_tradeoff)


    df_growth_summary.index = [m.id for m in models] + ["Community growth"]
    df_growth_summary.to_csv(path_to_save + SEPERATOR + f"growth_analysis.csv")

    
    
    df_fva["FBA"] = df_fba["fluxes"]
    if cfg.community.cooperative_tradeoff and df_fba_cooperative_tradeoff is not None:
        df_fva[
            f"Cooperative tradeoff (alpha: {cfg.community.cooperative_tradeoff_params.alpha})"
        ] = df_fba_cooperative_tradeoff["fluxes"]
    # else: tradeoff skipped (non-QP solver) — nothing to add

    log.info("Saving all flux analysis")
    df_fva.to_csv(path_to_save + SEPERATOR + f"flux_analysis.csv")

def pairwise_growth(community_model, cfg, log, path_to_save):
    log.info("Compute pairwise growth relationships for different weights.")
    fig = plot_pairwise_growth_relation_per_weight(
        community_model, cfg.visualization.names, **cfg.community.pairwise_growth_params,
    )
    fig.savefig(
        path_to_save + SEPERATOR + "pairwise_growth_relationship.pdf"
    )

def community_flux_summary(community_model, cfg, path_to_save):
    summary_1 = community_model.summary(**cfg.community.optimize_params)
    summary_1.to_csv(path_to_save + SEPERATOR + f"flux_summary.csv")

    fig = plot_community_summary(community_model, summary_1, cfg.visualization.names)
    fig.savefig(path_to_save + SEPERATOR + f"community_summary.pdf")

    summary_2 = None
    if cfg.community.cooperative_tradeoff:
        # Only attempt QP if the current optlang interface is gurobi or cplex
        iface = getattr(
            getattr(community_model, "community_model", community_model).solver.interface,
            "__name__",
            ""
        ).lower()
        qp_capable = ("gurobi" in iface) or ("cplex" in iface)
        if qp_capable:
            summary_2 = community_model.summary(
                cooperative_tradeoff=cfg.community.cooperative_tradeoff_params["alpha"]
            )
            summary_2.to_csv(path_to_save + SEPERATOR + "flux_summary_cooperative_tradeoff.csv")

            # Only plot if there is at least one data column (everything except the last “Total exchange” column)
            if (summary_2 is not None) and (not summary_2.empty) and (len(summary_2.columns) > 1):
                fig = plot_community_summary(community_model, summary_2, cfg.visualization.names)
                fig.savefig(path_to_save + SEPERATOR + "community_summary_cooperative_tradeoff.pdf")
            else:
                logging.getLogger(__name__).warning(
                    "Skipping cooperative_tradeoff plot: no data to plot (empty/degenerate summary)."
                )
        else:
            logging.getLogger(__name__).warning(
                "Skipping cooperative_tradeoff in community_flux_summary: "
                f"solver '{iface}' is not QP-capable (requires Gurobi or CPLEX)."
            )

    return summary_1, summary_2

def run_community(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    cobra_loger = logging.getLogger()
    cobra_loger.setLevel(logging.ERROR)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Hostname: {socket.gethostname()}")

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    log.info(f"Random seed: {seed}")

    start_time = time.time()
    name = cfg.name


    PATH_res = get_result_path(name)
    PATH = PATH_res + SEPERATOR + "community"

    cfg = set_solver_disable_functionalities_if_needed(cfg, log)

    #Creates folder structure
    log.info(f"Working directory: {PATH}")
    create_community_folder_backbone(PATH, PATH_res)
    log.info("Successfully created folder backbone.")

    # Load configurations over previous runs
    old_cfg = load_old_configs(PATH, cfg)

    log.info("Loading models from setup")
    if cfg.community.models_folder == "setup":
        try:
            models = get_models(
                "snm3_models", prefix=PATH_res + SEPERATOR + "setup" + SEPERATOR
            )
        except:
            raise ValueError(
                "We require snm3 models for the next steps, please run the setup! Alternative specifiy comunity.models_folder with a path to the models you want to use"
            )
    else:
        try:
            models = get_models("", prefix=cfg.community.models_folder)
        except:
            raise ValueError(f"No models found at {cfg.community.models_folder}")

    # Generating community models
    community_models = generate_community_models(models,cfg,old_cfg, log, PATH)
    
    # Saving community models
    log.info(f"Saving models: {cfg.community.save_models}")
    if cfg.community.save_models:
        for m in community_models:
            path = PATH + SEPERATOR + "community_models" + SEPERATOR + type(m).__name__
            try:
                # Hirarchical is not pickable
                m.save(path)
            except:
                pass
            log.info(f"Saving community models: {path}")
    if cfg.community.save_as_sbml:
        for m in community_models:
            path = (
                PATH
                + SEPERATOR
                + "community_models"
                + SEPERATOR
                + type(m).__name__
                + ".xml"
            )
            m.save_as_sbml(path)
            log.info(f"Saving community models as sbml: {path}")

    # Medias
    all_medias = generate_medias(community_models[-1], cfg, log, PATH, PATH_res)

    # Now perform experiments for all community models and so on.
    for m in community_models:
        if cfg.community.medium["coopm"]:
            # These are different for different community models!
            coopm = generate_coopm_medium(m, all_medias, cfg,log, PATH)
            all_medias["coopm"] = coopm
        for medium_name, medium in all_medias.items():
            # Set correct medium
            m.medium = medium
            # Get path to save
            path_to_save = (
                PATH
                + SEPERATOR
                + "experiments_"
                + type(m).__name__
                + SEPERATOR
                + medium_name
            )
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            
            flux_analysis(m, models, medium_name, cfg, log, PATH, path_to_save)

            if cfg.community.pairwise_growth:
                pairwise_growth(m, cfg, log ,path_to_save)
            
            if m._type == "compartmentalized" and isinstance(m, ShuttleCommunityModel):
                log.info("Compute flux summary")
                summary_1, summary_2 = community_flux_summary(m, cfg, path_to_save)

                log.info(f"Summary: {summary_1}")
                log.info(f"Summary cooperative tradeof: {summary_2}")

                log.info(f"Compute community visualization")
                fig = plot_community_interaction(
                    m, summary_1, names=cfg.visualization.names
                )
                fig.savefig(path_to_save + SEPERATOR + "community_exchange.pdf", bbox_inches="tight")
                fig = plot_community_uptake_graph(
                    m, summary_1, names=cfg.visualization.names
                )
                fig.savefig(
                    path_to_save + SEPERATOR + "community_uptake.png", dpi=1000, bbox_inches="tight"
                )

                fig = plot_species_interaction(
                    m, summary_1, names=cfg.visualization.names
                )
                fig.savefig(path_to_save + SEPERATOR + "species_interaction.pdf", bbox_inches="tight")

                if cfg.community.cooperative_tradeoff and (summary_2 is not None):
                    fig = plot_community_interaction(
                        m, summary_2, names=cfg.visualization.names
                    )
                    fig.savefig(
                        path_to_save
                        + SEPERATOR
                        + "community_exchange_cooperative_tradeoff.pdf", bbox_inches="tight"
                    )
                    fig = plot_community_uptake_graph(
                        m, summary_2, names=cfg.visualization.names
                    )
                    fig.savefig(
                        path_to_save
                        + SEPERATOR
                        + "community_uptake_cooperative_tradeoff.png",
                        dpi=1000, bbox_inches="tight"
                    )

                    fig = plot_species_interaction(
                        m, summary_2, names=cfg.visualization.names
                    )
                    fig.savefig(
                        path_to_save
                        + SEPERATOR
                        + "species_interaction_cooperative_tradeoff.pdf", bbox_inches="tight"
                    )

    log.info("Compute weight posteriors")
    if cfg.community.compute_infer_weights:
        for m in community_models:
            m.weights = np.ones(len(m.weights))
            m.medium = all_medias[cfg.community.infer_weights.medium]
            if cfg.community.infer_weights.observed_individual_biomass == "balanced":
                maximal_growths = [m.single_optimize(i) for i in range(len(m.models))]
                balanced_observation = np.ones(len(m.models)) * np.min(maximal_growths)
                observation = balanced_observation
            else:
                observation = cfg.community.infer_weights.observed_individual_biomass
                assert len(
                    cfg.community.infer_weights.observed_individual_biomass
                ) == len(
                    m.models
                ), "You have to specify for each community member an observed growth."
                for o in observation:
                    assert o >= 0, "The observed growth must be greater than zero!"

            alpha = None
            if cfg.community.infer_weights.competitive_tradeoff:
                alpha = cfg.community.infer_weights.competitive_tradeoff_alpha
            posterior, weights, growths = community_weight_posterior(
                m,
                num_simulations=cfg.community.infer_weights.simulations_for_different_weights,
                enforce_survival=cfg.community.infer_weights.enforce_survival,
                cooperative_tradeoff=alpha,
            )

            df1 = pd.DataFrame(weights.numpy(), columns=[f"Weight: {m.id}" for m in models])
            df2 = pd.DataFrame(growths.numpy(), columns=[f"Growth: {m.id}" for m in models])

            simulations = pd.concat([df1,df2], axis=1)
            simulations.to_csv(PATH + SEPERATOR + "weight_inference"
                + SEPERATOR + type(m).__name__ +  f"simulations.csv")

            fig = plot_weight_growth_pairplot(
                m, weights, growths, names=cfg.visualization.names
            )
            fig.savefig(
                PATH
                + SEPERATOR
                + "weight_inference"
                + SEPERATOR
                + type(m).__name__
                + "weight_growth_pairplot.pdf"
            )

            fig = plot_posterior_samples_for_observation(m, posterior, observation)
            fig.savefig(
                PATH
                + SEPERATOR
                + "weight_inference"
                + SEPERATOR
                + type(m).__name__
                + "weight_posterior.pdf"
            )

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f"Finished Workflow in {runtime} seconds")
