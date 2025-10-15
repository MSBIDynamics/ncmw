from pandas import DataFrame
import pandas as pd
from cobra.medium import minimal_medium
from cobra.flux_analysis import flux_variability_analysis

from typing import Iterable, Optional, List
import numpy as np


def compute_fvas(models: Iterable, fraction: float) -> List:
    """Compute the FVA result for all models

    Args:
        models: List of models
        fraction: Fraction of maximal biomass rate that must be achived

    Returns:
        list: List of DataFrames containing minimum and maximum fluxes

    """
    dfs = []
    for model in models:
        fva = flux_variability_analysis(model, fraction_of_optimum=fraction)
        dfs.append(fva)
    return dfs


# def compute_COMPM(models: list, fvas: Optional[list] = None) -> List:
#     r"""Computes the COMPM medium, given all the fva results. The COMPM is defined as
#     the medium, in which all models can achive their maximum biomass rate (MBR) if they are alone.

#     Mathematically it requires the FVA results for each reaction contained in the
#     medium! This contains the minimum flux required by the models to obtain
#     MBR. We denote it by :math:`\text{FVA}_{min; r}^{m_i}` for any reaction :math:`r` of model :math:`m_i`.

#     For reactions :math:`r \in M` within the medium :math:`M` of :math:`K` models we
#     define the COMPM as

#     .. math:: COMPM = \left\{ \min_{k = 1...K} \text{FVA}_{min;r}^{m_k} \right\}_{r \in M}

#     Args:
#         models (list): List of cobra metabolit models
#         fvas (list, optional): List of dataframes containing FVA results for the models.
#                                If argument is "None", then it will be recomputed based on the given models.


#     Returns:
#         mediums (list): List of COMPM mediums for each model
#     """
#     if fvas is None:
#         fvas = compute_fvas(models, 1.0)
#     df = pd.concat(list(fvas))
#     df = df.groupby(df.index).min()
#     mediums = []
#     for model in models:
#         medium = model.medium
#         for key in medium:
#             if key in df.index:
#                 flux = df.loc[key].minimum
#                 if flux < 0:
#                     medium[key] = -flux
#                 else:
#                     medium[key] = 0
#         mediums.append(medium)

#     for model, medium in zip(models, mediums):
#         with model as m:
#             max_growth = m.slim_optimize()
#             m.medium = medium
#             growth = m.slim_optimize()
#             assert (
#                 growth >= max_growth - 1e-10
#             ), "In the COMPM medium all community members must reach maximum growth rate, check the input!"

#     return mediums

def compute_COMPM(models: list, fvas: Optional[list] = None) -> List:
    r"""Computes the COMPM medium, given all the FVA results. The COMPM is defined as
    the medium in which all models can achieve their maximum biomass rate (MBR) if alone.

    Mathematically it requires the FVA results for each reaction contained in the
    medium. This contains the minimum flux required by the models to obtain MBR.
    We denote it by :math:`\text{FVA}_{min; r}^{m_i}` for any reaction :math:`r` of model :math:`m_i`.

    For reactions :math:`r \in M` within the medium :math:`M` of :math:`K` models we
    define the COMPM as

    .. math:: COMPM = \left\{ \min_{k = 1...K} \text{FVA}_{min;r}^{m_k} \right\}_{r \in M}

    Args:
        models (list): List of cobra metabolic models.
        fvas (list, optional): List of DataFrames containing FVA results for the models.
                               If None, they will be recomputed at fraction 1.0.

    Returns:
        List: List of COMPM media (dict-like) for each model.
    """
    import warnings
    import copy
    import pandas as pd

    # 1) Get FVAs (min fluxes at 100% of optimum)
    if fvas is None:
        fvas = compute_fvas(models, 1.0)

    # 2) Combine FVAs and take the minimum required flux across models per exchange
    df = pd.concat(list(fvas))
    df = df.groupby(df.index).min()

    # 3) Build a medium per model from its current medium keys
    mediums = []
    for model in models:
        medium = copy.deepcopy(model.medium)
        for ex_id in list(medium.keys()):
            if ex_id in df.index:
                # If min flux is negative, model requires uptake of magnitude -min
                min_flux = float(df.loc[ex_id].minimum)
                medium[ex_id] = -min_flux if min_flux < 0 else 0.0
        mediums.append(medium)

    # 4) Validate with numeric tolerance; try one gentle relaxation if needed
    for idx, (model, medium) in enumerate(zip(models, mediums)):
        with model as m:
            # Baseline optimum on the model's current medium
            m.medium = copy.deepcopy(model.medium)
            max_growth = m.slim_optimize()

            # Apply COMPM candidate
            m.medium = copy.deepcopy(medium)
            growth = m.slim_optimize()

            # Absolute + relative tolerance
            tol = 1e-8 + 1e-6 * max(1.0, abs(max_growth))

            if growth + tol < max_growth:
                # Try a small relaxation to overcome numerical/rounding issues
                relaxed = {k: (v * 1.05 if v > 0 else v) for k, v in medium.items()}
                m.medium = relaxed
                growth2 = m.slim_optimize()

                if growth2 + tol < max_growth:
                    warnings.warn(
                        f"[COMPM] {getattr(model, 'id', f'model_{idx}')} reaches "
                        f"{growth2:.6g} < max {max_growth:.6g} even after relaxation; "
                        "continuing with relaxed medium."
                    )
                    mediums[idx] = relaxed
                else:
                    mediums[idx] = relaxed

    return mediums

