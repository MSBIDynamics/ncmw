from ncmw.community.community_models import CommunityModel
from typing import List, Tuple
import numpy as np
from numpy.typing import ArrayLike as Array
import pandas as pd
from cobra.exceptions import OptimizationError

import networkx as nx

import torch
import warnings
try:
    from sbi.inference import SNLE
except:
    pass


def compute_species_interaction_weights(
    model: CommunityModel, df: pd.DataFrame, alpha: float = 1.0
) -> Array:
    """Compute interaction between two species. The values lie between -1 (negative
    interaction) and + 1 postitive interaction. Be :math:`p_{ij}` the number of
    metabolites produced by i and consumed by j. Be :math:`c_{ij}` the number of
    metabolites consumed by both i and j. Then we compute the interaction as

    .. math:: w_{ij} = w_i* ( p_{ij}/\sum_k p_{ik} - c_{ij}/\sum_k c_{ik})

    Args:
        model: Communiy model
        df: Summary table
        alpha: Parameter for postive interaction, larger implies that postive
               interaction is wheighted more.

    Returns:
        Array: N x N matrix of interaction weights between species.

    """
    N = len(model.models)
    weights = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w_i = model.weights[i]
            summary_i = df[df.columns[i]]
            summary_j = df[df.columns[j]]
            consumed_j = summary_j < 0
            produced_i = summary_i > 0
            consumed_i = summary_i < 0
            positive_normalizer = 0
            negative_normalizer = 0
            for m in range(N):
                summary_m = df[df.columns[m]]
                produced_m = summary_m > 0
                consumed_m = summary_m < 0
                positive_normalizer += model.weights[m] * produced_m[consumed_j].sum()
                negative_normalizer += model.weights[m] * consumed_m[consumed_j].sum()
            if positive_normalizer != 0:
                weights[i, j] = w_i * produced_i[consumed_j].sum() / positive_normalizer
            else:
                weights[i, j] = 0
            weights[i, j] *= alpha
            if negative_normalizer != 0:
                weights[i, j] -= (
                    w_i * consumed_i[consumed_j].sum() / negative_normalizer
                )
            else:
                weights[i, j] -= 0
    return weights


def compute_community_interaction_graph(model, df):
    """Compute a graph that displays the community interaction. We say that model i
    interacts with model j if i produces something that j needs, but is not provided by
    the medium! We consider each pair of external metabolites and connect them by this
    criterium.



    Args:
        model: Cobra model
        df: A summary file i.e. that returned by model.summary()
    Returns:
        Graph: Interaction graph
        df: Relevant summary

    """
    df_help = df[df.columns[:-1]]
    medium_col = np.zeros(len(df_help))
    for i, ex in enumerate(df_help.index):
        if ex in model.medium:
            medium_col[i] = model.medium[ex]
        else:
            medium_col[i] = 0.0
    # Set output to zero
    help_array = df_help.to_numpy()
    help_array[help_array >= 0] = 0
    # Non medium associated inputs -> interactions !
    species_interaction = (help_array.sum(1) + medium_col) < -1e-6
    species_interaction
    df = df[species_interaction]
    df = df.drop(columns=df.columns[-1])

    # Build interaction graph
    G = nx.DiGraph()
    # Build nodes
    for i, col in enumerate(df.columns):
        names = [n[3:-2] + f"_{i}" for n in df[df[col] != 0].index]
        G.add_nodes_from(names)

    for n in G.nodes:
        i = int(n[-1])
        G.nodes[n]["class"] = model.models[i].id.split("_")[0]
    return G, df


def community_weight_posterior(
    model: CommunityModel,
    enforce_survival:float=0.0,
    num_simulations:int=10000,
    cooperative_tradeoff: float=None,
    noise:float=0.05,
    prior:str="dirichlet",
):
    warnings.warn("This function is deprecated as dependencies became incompatible with the version of the package... . You can still use the function in a separate environment.")
    N = len(model.models)

    def simulator(thetas):
        xs = []
        for weight in thetas:
            model.weights = weight.numpy()
            if cooperative_tradeoff is None:
                growths = torch.tensor(
                    model.optimize(enforce_survival=enforce_survival)[1]
                )
            else:
                growths = torch.tensor(
                    model.cooperative_tradeoff(cooperative_tradeoff)[1]
                )
            xs.append(growths)
        xs = torch.vstack(xs).float()
        xs += noise * torch.rand_like(xs)
        return xs

    if prior == "dirichlet":
        prior = torch.distributions.Dirichlet(torch.ones(N))
        prior.set_default_validate_args(False)
    elif prior == "boxuniform":
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(torch.zeros(N), torch.ones(N)), 1
        )
        prior.set_default_validate_args(False)

    thetas = prior.sample((num_simulations,))
    xs = simulator(thetas)

    inf = SNLE(prior)
    _ = inf.append_simulations(thetas, xs).train()
    posterior = inf.build_posterior(
        sample_with="vi",
        vi_parameters={"q": "nsf"},
    )
    return posterior, thetas, xs


def compute_pairwise_growth_relation_per_weight(
    community_model: CommunityModel, idx1: int, idx2: int, h: int = 200
) -> Tuple[Array, Array, Array]:
    """We compare the pairwise growth relation as following: Be :math:`\\alpha \in
    [0,1]` and :math:`G_i, G_j` the growth expression for model i and j. Then we
    maximize the community objective :math:`\\alpha G_i + (1-\\alpha)G_j` and report the
    growth values.

    Args:
        community_model: Community model
        idx1: Index of community member which is tested
        idx2: Index of second community member which is tested
        h: Number of growth evaluations, for different weights.

    Returns:
        np.array: The values of  :math:`\\alpha` on which we the growths.
        np.array: The growth values of the first model.
        np.array: The growth values of the second model.
    """
    weights = community_model.weights
    N = len(weights)
    weight_mask1 = np.zeros(N)
    weight_mask1[idx1] = 1.0
    weight_mask2 = np.zeros(N)
    weight_mask2[idx2] = 1.0
    alpha = np.linspace(0, 1, h)
    alphas = alpha.reshape(-1, 1).repeat(N, 1)
    alpha_weights = alphas * weight_mask1 + (1 - alphas) * weight_mask2

    growth1 = np.zeros(h)
    growth2 = np.zeros(h)
    for i in range(h):
        community_model.weights = alpha_weights[i]
        try:
            _, single_growths, _ = community_model.optimize()
        except OptimizationError:
            # Keep lengths consistent with `h` to avoid plotting mismatches.
            return alpha, np.full(h, np.nan), np.full(h, np.nan)
        growth1[i] = single_growths[idx1]
        growth2[i] = single_growths[idx2]
    return alpha, growth1, growth2


def compute_fair_weights(community_model: CommunityModel) -> np.array:
    """Compute fair weights, fair in the sense that organisms with high single growth
    has a low weights such that all species have the same weighted MBR. More precisely
    if :math:`G_i` is the growth of model i, the corresponding unnormalized weight is defined as

    .. math:: w_i = 1/(G_i +1e-32)

    We then normalize the vector such that  :math:`\sum_i w_i = 1`

    Args:
        community_model: Cobra Model

    Returns:
        array: Numpy array of weights

    """
    N = len(community_model.models)
    weights = np.array([community_model.single_optimize(i) for i in range(N)])
    weights = 1 / (weights + 1e-32)
    weights /= weights.sum()
    return weights


def compute_dominant_weights(
    community_model: CommunityModel, high_val: float = 0.9
) -> List[np.array]:
    """For each community member we compute the weight in which one of the members has
    a dominant weight. For example if :math:`m_i` is the dominant model then :math:`w_i =`high-val and all other :math:`w_j = (1-high-val)/(N-1)`

    Args:
        community_model (CommunityModel): A community model
        high_val (float): The largest weight value

    Returns:
        List[np.array]: List of weigt vector, for any member of the community. In each
        vector one member can be thought as "dominant"

    """
    N = len(community_model.models)
    others_w = (1 - high_val) / (N - 1)
    weights = []
    for i in range(N):
        base_weight = np.ones(N) * others_w
        base_weight[i] = high_val
        weights.append(base_weight)
    return weights


def compute_community_summary(community_model, weights):
    df_total = pd.DataFrame()
    for i in range(len(weights)):
        community_model.weights = weights[i]
        growth, single_growth, sol = community_model.optimize()
        df = pd.DataFrame(
            dict(
                zip([m.id for m in community_model.models], np.round(single_growth, 3))
            ),
            index=[i],
        )
        df["Total"] = np.round(growth, 3)
        df["Weight"] = str(community_model.weights)
        df_total = df_total.append(df)
    return df_total
