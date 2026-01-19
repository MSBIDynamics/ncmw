from copy import deepcopy
import cobra
import numpy as np
import scipy


from abc import ABC, abstractmethod

from warnings import warn
import logging

from .hierarchical_model_composition import generate_hierarchical_community_model


import pickle

import pandas as pd
import tempfile, os
from cobra.io import save_json_model, load_json_model

def _safe_model_clone(m):
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        save_json_model(m, path)
        return load_json_model(path)
    finally:
        try: os.remove(path)
        except OSError: pass

# --- Helper utilities for QP solve without invoking cobra.get_solution() ---
class _FluxView(dict):
    """
    Lightweight, dict-like container for fluxes that supports `sol[rxn_id]`
    and an optional `to_frame()` for legacy call sites.
    """
    def __init__(self, fluxes: dict[str, float], objective_value: float | None = None):
        super().__init__(fluxes)
        self.objective_value = objective_value

    def __getitem__(self, key):
        # Return 0.0 if a reaction id is missing (mirrors common cobra usage)
        return super().get(key, 0.0)

    @property
    def fluxes(self):
        # Provide a pandas-style accessor if someone expects `sol.fluxes[...]`
        return pd.Series(self)

    def to_frame(self):
        # Keep compatibility if any code still calls `sol.to_frame()`
        return pd.DataFrame.from_dict(self, orient="index", columns=["flux"])


def _primal_fluxes(model) -> dict[str, float]:
    """
    Read primal values for each reaction's net flux directly from the solver
    (avoid cobra.get_solution to prevent dual lookups that break on QP).
    Net flux = forward_variable - reverse_variable.
    """
    fluxes: dict[str, float] = {}
    for rxn in model.reactions:
        # Use forward/reverse variables; `rxn.variable` is not guaranteed.
        fwd = getattr(rxn, "forward_variable", None)
        rev = getattr(rxn, "reverse_variable", None)

        val = 0.0
        # Some solver interfaces may return None until optimized; caller must have
        # called model.solver.optimize() right before.
        if fwd is not None:
            try:
                pv = fwd.primal
                if pv is not None:
                    val += float(pv)
            except Exception:
                pass
        if rev is not None:
            try:
                pv = rev.primal
                if pv is not None:
                    val -= float(pv)
            except Exception:
                pass

        fluxes[rxn.id] = val
    return fluxes

def create_stoichiometry_matrix(model):
    """This creates a stoichiometry matrix"""
    metabolites = model.metabolites
    reactions = model.reactions
    S = np.zeros((len(metabolites), len(reactions)))

    met_id = dict()
    rec_id = dict()
    for i, reaction in enumerate(model.reactions):
        rec_id[reaction.id] = i
        for metabolite, stoich in reaction.metabolites.items():
            met_id[metabolite.id] = int(metabolites.index(metabolite))
            S[metabolites.index(metabolite), i] = stoich
    # Compress as sparse matrix
    S = scipy.sparse.csr_matrix(S)
    return S, met_id, rec_id


def get_biomass_reaction(model):
    """Return the biomass reaction of a model"""
    objective_str = str(list(model.objective.variables)[0])
    for rec in model.reactions:
        if rec.id in objective_str:
            return rec


class CommunityModel(ABC):
    """This gives the backbone of any community model."""

    @property
    def medium(self):
        return self._medium

    @property
    def reactions(self):
        return self._reactions

    @property
    def metabolites(self):
        return self._metabolites

    @property
    def exchanges(self):
        return self._exchanges

    @medium.setter
    def medium(self, medium):
        return self._set_medium(medium)

    @abstractmethod
    def _set_medium(self, medium):
        """This method should set the medium properly"""
        pass

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        return self._set_weights(weights)

    @abstractmethod
    def _set_weights(self, weights):
        """This method should set the weights properly"""
        pass

    @abstractmethod
    def slim_optimize(self):
        """This method returns the current objective value"""
        pass

    @abstractmethod
    def single_optimize(self, idx):
        """This method returns the current objective value, if the objective only
        accounts a single model"""
        pass

    @abstractmethod
    def optimize(self):
        """This method return the current objective value and additional information"""
        pass

    def compute_COOPM(self, MBR, fraction=0.1, enforce_survival=True):
        """This method computes the COOPM medium"""
        raise NotImplementedError(
            "This method is not implemented fro your current model"
        )

    def summary(self):
        """This method should report a summary of the model"""
        raise NotImplementedError(
            "This method is not implemented for your current model"
        )

    def save(self, path):
        """This saves the model using pickle. For other format overwrite this function"""
        with open(path + ".pkl", "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class BagOfReactionsModel(CommunityModel):
    """This is a community model, which treats the community as a bag of all reactions that at least one the species has."""

    def __init__(self, models: list):
        """This will create a BagOfReactionsModel form a list of models. This is
        equivalent to create a model that contains the union of all reactions and metabolites
        within at least one model in the list.


                Args:
                    models: List of cobra models


        """
        self.community_model = cobra.Model(
            "Community__" + "".join([model.id + "__" for model in models])
        )
        self._type = "bag"
        self.models = [_safe_model_clone(m) for m in models]
        self.biomass_reactions = [get_biomass_reaction(model) for model in self.models]
        # Rename biomass reaction before merge
        for rec, model in zip(self.biomass_reactions, self.models):
            model.remove_reactions([rec])
            rec.id = rec.id + "__" + model.id
            model.add_reactions([rec])
            model.objective = rec.flux_expression

        cobra_loger = logging.getLogger()
        cobra_loger.setLevel(logging.ERROR)
        for model in self.models:
            tmp = _safe_model_clone(model)
            existing_mets = {met.id for met in self.community_model.metabolites}
            self.community_model.add_metabolites([m for m in tmp.metabolites if m.id not in existing_mets])
            existing_rxns = {rxn.id for rxn in self.community_model.reactions}
            self.community_model.add_reactions([r for r in tmp.reactions if r.id not in existing_rxns])

        cobra_loger.setLevel(logging.WARNING)
        for i in range(len(self.biomass_reactions)):
            self.biomass_reactions[i] = self.community_model.reactions.get_by_id(
                self.biomass_reactions[i].id
            )

        self._weights = np.ones(len(models))
        self._medium = self.community_model.medium
        self._reactions = self.community_model.reactions
        self._metabolites = self.community_model.metabolites
        self._exchanges = self.community_model.exchanges
        self.objective = sum([f.flux_expression for f in self.biomass_reactions])
        self.community_model.objective = self.objective

    def _set_medium(self, medium):
        exchanges = [e.id for e in self.community_model.exchanges]
        present_metabolites = dict()
        missing_metabolites = dict()
        for m, val in medium.items():
            if m in exchanges:
                present_metabolites[m] = val
            else:
                missing_metabolites[m] = val
        self.community_model.medium = present_metabolites
        for m in missing_metabolites:
            warn(m + " not present in the model thus removed from the medium.")
        self._medium = medium

    def _set_weights(self, weights):
        assert len(weights) == len(
            self.models
        ), "You need to specify for each species in the community a weights..."
        self._weights = weights
        self.objective = sum(
            [
                weights[i] * self.biomass_reactions[i].flux_expression
                for i in range(len(self._weights))
            ]
        )
        self.community_model.objective = self.objective
        self.community_model.solver.update()

    def slim_optimize(self, enforce_survival=0) -> float:
        """Return the current community objective

        Returns:
            float: Community objective

        """
        return self.optimize(enforce_survival=enforce_survival)[0]

    def cooperative_tradeoff(self, alpha: float = 0.9):
        """Performs a cooperative tradeoff optimization strategy, which requires two
            steps: First a FBA solution is obtain, we call it MBR. Then we minimize the
            L2 norm of a vector containing all individual growths under the constraint
            that the community objective is greater than :math:`\alpha \cdot MBR`

        Args:
            alpha: Tradeoff value, must be between 0 and 1.

        Returns:
            community_growth : Value of the community objective
            individual_growth: List of individual growths
            solution: Cobra solution object, containing fluxes for all reactions

        """
        MBR = self.slim_optimize()
        iface = self.community_model.solver.interface.__name__.lower()
        if not any(k in iface for k in ("gurobi", "cplex")):
            raise AssertionError(
                "Cooperative trade-off requires a QP-capable solver (Gurobi or CPLEX). "
                f"Current solver: {self.community_model.solver.interface.__name__}"
            )
        assert alpha <= 1 and alpha > 0, "This hyperparameter has to be between 0 and 1"

        # We make a copy because switching between LP and QP problems can become expensive.
        model = deepcopy(self.community_model)
        biomass_reactions = [model.reactions.get_by_id(i.id) for i in self.biomass_reactions]
        min_community_growth = alpha * MBR
        constraint_growth = [
            model.problem.Constraint(
                sum(
                    [
                        self._weights[i] * r.flux_expression
                        for r in biomass_reactions
                    ]
                ),
                lb=min_community_growth,
            )
            for i, f in enumerate(biomass_reactions)
        ]
        model.objective = -sum(
            [r.flux_expression ** 2 for r in biomass_reactions]
        )
        model.add_cons_vars(constraint_growth)
        model.solver.update()
        # Solve QP with the raw solver (avoid COBRA's get_solution to skip dual lookups)
        model.solver.optimize()

        # Pull primal fluxes directly from solver variables
        fluxes = _primal_fluxes(model)
        single_growths = [fluxes[r.id] for r in biomass_reactions]
        community_growth = float(sum(self.weights[i] * fluxes[r.id] for i, r in enumerate(biomass_reactions)))

        # Lightweight, dict-like “solution” so existing call-sites keep working
        sol = _FluxView(fluxes, community_growth)

        del model
        return community_growth, single_growths, sol

    def _add_enforce_survival_constraints(self, percent):
        N = len(self.biomass_reactions)
        constraint_growth = [
            self.community_model.problem.Constraint(
                self._weights[i] * f.flux_expression
                - percent
                / N
                * sum(
                    [
                        self._weights[i] * r.flux_expression
                        for r in self.biomass_reactions
                    ]
                ),
                lb=0,
                ub=1000,
            )
            for i, f in enumerate(self.biomass_reactions)
        ]
        return constraint_growth

    def flux_variability_analysis(
        self, loopless=False, fraction_of_optimum=1.0, pfba_factor=None, processes=None,
    ):
        """Determine the minimum and maximum possible flux value for each reaction.

        Parameters:
        loopless (boolean, optional):
            Whether to return only loopless solutions. This is significantly
            slower. Please also refer to the notes.
        fraction_of_optimum (float, optional) :
            Must be <= 1.0. Requires that the objective value is at least the
            fraction times maximum objective value. A value of 0.85 for instance
            means that the objective has to be at least at 85% percent of its
            maximum.
        pfba_factor (float, optional):
            Add an additional constraint to the model that requires the total sum
            of absolute fluxes must not be larger than this value times the
            smallest possible sum of absolute fluxes, i.e., by setting the value
            to 1.1 the total sum of absolute fluxes must not be more than
            10% larger than the pFBA solution. Since the pFBA solution is the
            one that optimally minimizes the total flux sum, the ``pfba_factor``
            should, if set, be larger than one. Setting this value may lead to
            more realistic predictions of the effective flux bounds.
        processes (int, optional):
            The number of parallel processes to run. If not explicitly passed,
            will be set from the global configuration singleton.

        Returns
        pandas.DataFrame
            A data frame with reaction identifiers as the index and two columns:
            - maximum: indicating the highest possible flux
            - minimum: indicating the lowest possible flux
        """
        return cobra.flux_analysis.flux_variability_analysis(
            self.community_model,
            loopless=loopless,
            fraction_of_optimum=fraction_of_optimum,
            pfba_factor=pfba_factor,
            processes=processes,
        )

    def optimize(self, enforce_survival=0):
        """Performs FBA on the community model

        Args:
            enforce_survival: Must be between 0 and 1. It constrains the community such
            that all members must have at least X % of the community growth.
        Returns:
            community_growth : Value of the community objective
            individual_growth: List of individual growths
            solution: dict-like object with reaction fluxes (id -> value)
        """
        model = self.community_model

        # If enforcing survival, add constraints and solve directly with the solver.
        if enforce_survival > 0:
            assert 0 < enforce_survival <= 1, "Minimal percentage must be between 0 and 1."
            constraint_growth = self._add_enforce_survival_constraints(enforce_survival)
            model.add_cons_vars(constraint_growth)
            model.solver.update()
            model.solver.optimize()

            fluxes_dict = _primal_fluxes(model)
            sol = _FluxView(fluxes_dict, objective_value=float(model.solver.objective.value))
            total_growth = float(model.solver.objective.value)  # LP objective value

            # Clean up the temporary constraints
            model.remove_cons_vars(constraint_growth)
            model.solver.update()

        else:
            # No extra constraints: solve once and build a flux view.
            model.solver.update()
            model.solver.optimize()
            fluxes_dict = _primal_fluxes(model)
            sol = _FluxView(fluxes_dict, objective_value=float(model.solver.objective.value))
            # keep slim_optimize() for compatibility if you prefer; objective value is same
            total_growth = model.slim_optimize()

        single_growths = [sol[r.id] for r in self.biomass_reactions]
        return total_growth, single_growths, sol

    def single_optimize(self, idx):
        weights = np.zeros(len(self.models))
        weights[idx] = 1.0
        old_weights = self._weights
        self._set_weights(weights)
        growth = self.slim_optimize()
        self._set_weights(old_weights)

        return growth

    def summary(self, enforce_survival=0, cooperative_tradeoff=None):
        raise NotImplementedError(
            "Within the bag of reaction model, we cannot determine which metabolites is owned by which species, thus cannot report a community report. Use optimize obtain fluxes!"
        )

    def compute_COOPM(self, MBR, fraction=0.1, enforce_survival=0.1):
        """Computes the COOPM medium"""
        minMBR = fraction * MBR
        medium = list(self.medium.keys())
        # biomass = [f.flux_expression for f in self.biomass_reactions]

        # Binary variables: Theta_i
        thetas = []
        for i in range(len(medium)):
            thetas.append(
                self.community_model.problem.Variable("theta_" + str(i), type="binary")
            )

        # Constraints for exchanges, which are turned of for theta_i = 1
        theta_constraints = []
        for i, id in enumerate(medium):
            reaction = self.community_model.reactions.get_by_id(id)
            min_bound = -20.0
            cons = self.community_model.problem.Constraint(
                (reaction.flux_expression + min_bound * thetas[i]),
                lb=min_bound,
                ub=1000,
            )
            theta_constraints.append(cons)

        # Constraints for growth rates, which must be at least 10% MBR
        if enforce_survival:
            constraint_growth = self._add_enforce_survival_constraints(enforce_survival)
            constraint_growth += [
                self.community_model.problem.Constraint(
                    sum(
                        [
                            self._weights[i] * self.biomass_reactions[i].flux_expression
                            for i in range(len(self.biomass_reactions))
                        ]
                    ),
                    lb=minMBR,
                    ub=1000,
                )
            ]
        else:
            constraint_growth = self.community_model.problem.Constraint(
                sum(
                    [
                        self._weights[i] * self.biomass_reactions[i].flux_expression
                        for i in range(len(self.biomass_reactions))
                    ]
                ),
                lb=minMBR,
                ub=1000,
            )

        # Adding new variables and constraints.
        self.community_model.add_cons_vars(thetas)
        self.community_model.add_cons_vars(theta_constraints)
        self.community_model.add_cons_vars(constraint_growth)

        # Objevtive is maximising turned of exchanges, that is sum of theta_is
        objective = self.community_model.problem.Objective(sum(thetas), direction="max")
        self.community_model.objective = objective
        self.community_model.solver.update()

        sol = self.community_model.optimize()
        COOPM = dict()
        for id in medium:
            if sol.fluxes[id] < 0:
                COOPM[id] = abs(sol.fluxes[id])

        self.community_model.remove_cons_vars(thetas)
        self.community_model.remove_cons_vars(theta_constraints)
        self.community_model.remove_cons_vars(constraint_growth)

        self._set_weights(self._weights)
        self.community_model.solver.update()

        return COOPM

    def compute_convex_combination(self, alphas, maxbiomass=0.1):
        assert sum(alphas) == 1, "The weights must sum to one!"
        assert len(alphas) == len(self.models), "Scpecify a weight for each model..."
        model = _safe_model_clone(self.community_model)
        biomass = [
            model.reactions.get_by_id(f.id).flux_expression
            for f in self.biomass_reactions
        ]
        constraint_growth = [
            model.problem.Constraint(
                self.weights[i] * f,
                lb=alphas[i] * maxbiomass,
                ub=alphas[i] * maxbiomass,
            )
            for i, f in enumerate(biomass)
        ]
        model.add_cons_vars(constraint_growth)
        model.solver.update()

        sol = model.optimize()
        return model.summary()

    def save(self, path):
        """This saves the model using pickle. For other format overwrite this
        function"""
        self.objective = None

        with open(path + ".pkl", "wb+") as f:
            pickle.dump(self, f)

        self.objective = sum(
            [
                self.community_model.reactions.get_by_id(f.id).flux_expression
                for f in self.biomass_reactions
            ]
        )
        self.community_model.objective = self.objective
        self.community_model.solver.update()

    def save_as_sbml(self, path):
        """This will save the community model as sbml"""
        return cobra.io.sbml.write_sbml_model(self.community_model, path)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        model.objective = sum(
            [
                model.community_model.reactions.get_by_id(f.id).flux_expression
                for f in model.biomass_reactions
            ]
        )
        model.community_model.objective = model.objective
        return model


class ShuttleCommunityModel(BagOfReactionsModel):
    def __init__(self, models, shared_exchanges=None, **kwargs):
        """Constructs a shuttle community model. Each community member obtains it's own
        compartment. The interaction between compartments and the external environment
        is mediate by so called "shuttle reactions".

        Args:
            models: List of models that will form a community
            shared_exchanges: A list of exchange reactions which can be shared between
                members. By default all exchanges can be shared.
            shuttle_reaction_lower_bound: The lower bound of the shuttle reactions
                (default: -50).
            shuttle_reaction_upper_bound: The upper bound of the shuttle reactions
                (default: 1000).
            shuttle_regularization: A boolean value if we enable this regularization.
            This ensures that members that has zero growth (are dead) cannot exchange metabolites (default: True).
            shuttle_regularization_factor: A factor the regulates the strength of
                regularization. The larger the value the smaller the regularization
                (default: 1000).
        """
        self.models = models
        self.model_ids = [m.id for m in self.models]
        # For this reactions we impose shuttle reactions if possible.
        if shared_exchanges is None:
            self.shared_exchanges = []
            for model in models:
                for ex in model.exchanges:
                    if ex.id not in self.shared_exchanges:
                        self.shared_exchanges.append(ex.id)
        self._type = "compartmentalized"

        self.shuttle_reaction_lower_bound = kwargs.get(
            "shuttle_reaction_lower_bound", -50
        )
        self.shuttle_reaction_upper_bound = kwargs.get(
            "shuttle_reaction_upper_bound", 1000
        )
        self.shuttle_regularization = kwargs.get("shuttle_regularization", True)
        self.shuttle_regularization_factor = kwargs.get(
            "shuttle_regularization_factor", 1000
        )

        self.build_community()

        self._weights = np.ones(len(self.models))
        self.biomass_reactions = []
        self.biomass_id = []
        self._metabolites = self.community_model.metabolites
        self._reactions = self.community_model.reactions
        self._exchanges = self.community_model.exchanges

        for m in self.models:
            rec = get_biomass_reaction(m)
            id = rec.id + "__" + m.id
            self.biomass_id.append(id)
            self.biomass_reactions.append(self.community_model.reactions.get_by_id(id))
        self._set_weights(self._weights)

        if self.shuttle_regularization:
            self._add_shuttle_regularization()

    def _set_weights(self, weights):
        self._weights = weights
        objective = []
        for i, r in enumerate(self.biomass_reactions):
            objective.append(self._weights[i] * r.flux_expression)

        self.community_model.objective = sum(objective)
        self.community_model.solver.update()

    def _generate_shuttle_id_from_met_id(self, met, model_id):
        return "SH_" + met + "__" + model_id

    def _get_shuttle_id_shift(self, model_id):
        return 3

    def summary(self, enforce_survival: float = 0, cooperative_tradeoff: float = None):
        """Creates a community summary. We list all exchanges between members of the
            community and between the environment.



        Args:
            enforce_survival: Enforce a minimium percentage of growth for each member.
            cooperative_tradeoff: Perform cooperative tradeoff.

        Returns:
            pd.DataFrame: Table of fluxes.

        """
        if cooperative_tradeoff is None:
            total_growth, single_growths, sol = self.optimize(
                enforce_survival=enforce_survival
            )
        else:
            total_growth, single_growths, sol = self.cooperative_tradeoff(
                cooperative_tradeoff
            )
        exchanges = self.community_model.exchanges
        non_zero_ids = []
        non_zero_flux = []
        for ex in exchanges:
            flux = sol[ex.id]
            if flux != 0:
                non_zero_ids.append(ex.id)
                non_zero_flux.append(flux)

        vals = np.zeros((len(non_zero_ids), len(self.models) + 1))
        shuttle_reactions_ids = [s.id for s in self.shuttle_reactions]
        for i in range(len(non_zero_ids)):
            met = non_zero_ids[i][self._get_shuttle_id_shift(self.model_ids[0]) :]
            for j in range(len(self.models)):
                shuttle_id = self._generate_shuttle_id_from_met_id(
                    met, self.model_ids[j]
                )
                if shuttle_id in shuttle_reactions_ids:
                    vals[i, j] = sol[shuttle_id]
        vals[:, -1] = np.array(non_zero_flux)

        df = pd.DataFrame(
            vals,
            index=non_zero_ids,
            columns=[m.id for m in self.models] + ["Total exchange"],
        ).sort_values(["Total exchange"])

        print("Objective: ", total_growth)
        for i in range(len(self.models)):
            print(
                self.models[i].id + " : ",
                single_growths[i],
                " with weights ",
                self._weights[i],
            )
        return df

    def build_community(self):
        # Community model we will build
        community_model = cobra.Model("Community")

        # Collect new metabolite contained in each model
        new_metabolites = []
        for model in self.models:
            for old_metabolite in model.metabolites:
                new_metabolite = cobra.Metabolite()
                for key, val in old_metabolite.__dict__.items():
                    if (
                        key != "_id"
                        and key != "_model"
                        and key != "_reaction"
                        and key != "compartment"
                    ):
                        new_metabolite.__dict__[key] = val
                new_metabolite.id = old_metabolite.id + "__" + model.id
                new_metabolite.compartment = (
                    old_metabolite.compartment + "__" + model.id
                )
                new_metabolites.append(new_metabolite)
        # Add new metabolites to community model
        community_model.add_metabolites(new_metabolites)

        # Collect new reactions contained in each model
        new_reactions = []
        for model in self.models:
            for old_reaction in model.reactions:
                # Skip exchange reaction, they are no longer part of the internal models
                if old_reaction.id[:3] == "EX_":
                    continue
                new_reaction = cobra.Reaction()
                for key, val in old_reaction.__dict__.items():
                    if (
                        key != "_id"
                        and key != "_model"
                        and key != "_metabolites"
                        and key != "compartment"
                    ):
                        new_reaction.__dict__[key] = val
                new_reaction.id = old_reaction.id + "__" + model.id
                for met, stoch in old_reaction.metabolites.items():
                    new_met_id = met.id + "__" + model.id
                    new_met = community_model.metabolites.get_by_id(new_met_id)
                    new_reaction.add_metabolites({new_met: stoch})
                new_reactions.append(new_reaction)
        # Add new reactions to community model
        community_model.add_reactions(new_reactions)

        # Add new exchange reactions
        external_metabolites = []
        for model in self.models:
            # The union of external metabolites
            for met in model.metabolites:
                if "_e" == met.id[-2:] and met.id not in external_metabolites:
                    external_metabolites.append(met.id)
        new_external_metabolites = []
        for id in external_metabolites:
            for model in self.models:
                if id in [met.id for met in model.metabolites]:
                    old_metabolite = model.metabolites.get_by_id(id)
                    break
            new_metabolite = cobra.Metabolite()
            for key, val in old_metabolite.__dict__.items():
                if (
                    key != "_id"
                    and key != "_model"
                    and key != "_reaction"
                    and key != "compartment"
                ):
                    new_metabolite.__dict__[key] = val
            new_metabolite.id = old_metabolite.id
            new_metabolite.compartment = "external"
            new_external_metabolites.append(new_metabolite)
        community_model.add_metabolites(new_external_metabolites)
        for met in new_external_metabolites:
            community_model.add_boundary(met)

        # Add shuttle reactions
        shuttle_reactions = []
        for model in self.models:
            for met in new_external_metabolites:
                if (
                    met.id in [met.id for met in model.metabolites]
                    and f"EX_{met.id}"  # in self.shared_exchanges
                ):
                    met2 = community_model.metabolites.get_by_id(
                        met.id + "__" + model.id
                    )
                else:
                    continue
                shuttle_reaction = cobra.Reaction()
                id = met.id
                shuttle_reaction.id = f"SH_{id}__{model.id}"
                shuttle_reaction.name = f"Shuttle reaction for {id}"
                shuttle_reaction.lower_bound = self.shuttle_reaction_lower_bound
                shuttle_reaction.upper_bound = self.shuttle_reaction_upper_bound
                shuttle_reaction.add_metabolites({met: 1, met2: -1})
                shuttle_reactions.append(shuttle_reaction)
                community_model.add_reactions([shuttle_reaction])
        # Add shuttle constraints!
        all_shuttles = [ex.id for ex in community_model.reactions if "SH_" == ex.id[:3]]
        shuttle_constraints = []
        for met in new_external_metabolites:
            present_shuttles = []
            for model in self.models:
                id = f"SH_{met.id}__{model.id}"
                if id in all_shuttles:
                    present_shuttles.append(
                        community_model.reactions.get_by_id(id).flux_expression
                    )
            exchange = community_model.reactions.get_by_id(
                f"EX_{met.id}"
            ).flux_expression
            cons = community_model.problem.Constraint(
                exchange - sum(present_shuttles), lb=0, ub=1000,
            )
            shuttle_constraints.append(cons)
        community_model.add_cons_vars(shuttle_constraints)

        # Set correct medium
        medium = {}
        for model in self.models:
            for key, val in model.medium.items():
                if key not in medium:
                    medium[key] = val

        community_model.solver.update()
        self.community_model = community_model
        self.shuttle_reactions = [
            ex for ex in community_model.reactions if "SH_" == ex.id[:3]
        ]
        self._set_medium(medium)

    def _get_model_id_from_shuttle_id(self, str):
        return str.split("__")[-1]

    def _add_shuttle_regularization(self):
        # Add regularizatuion constraints if required
        cons = []
        C = self.shuttle_regularization_factor
        N = len(self.shuttle_reactions)
        for i in range(N):
            id = self.shuttle_reactions[i].id
            model_id = self._get_model_id_from_shuttle_id(id)
            model_ids = [m.id for m in self.models]
            j = model_ids.index(model_id)
            con1 = self.community_model.problem.Constraint(
                self.shuttle_reactions[i].flux_expression
                - C * self.weights[j] * self.biomass_reactions[j].flux_expression,
                ub=0,
            )
            con2 = self.community_model.problem.Constraint(
                self.shuttle_reactions[i].flux_expression
                + C * self.weights[j] * self.biomass_reactions[j].flux_expression,
                lb=0,
            )
            cons.append(con1)
            cons.append(con2)
        self.community_model.add_cons_vars(cons)

    def save(self, path):
        """This saves the model using pickle. For other format overwrite this
        function"""
        self.objective = None

        with open(path + ".pkl", "wb+") as f:
            pickle.dump(self, f)

        self.objective = sum(
            [
                self.community_model.reactions.get_by_id(f.id).flux_expression
                for f in self.biomass_reactions
            ]
        )
        self.community_model.objective = self.objective
        self.community_model.solver.update()

    @staticmethod
    def load(path):
        """Loads the model"""
        with open(path, "rb") as f:
            model = pickle.load(f)

        model.objective = sum(
            [
                model.community_model.reactions.get_by_id(f.id).flux_expression
                for f in model.biomass_reactions
            ]
        )
        model.community_model.objective = model.objective
        model.community_model.solver.update()
        return model


class HierarchicalCommunityModel(ShuttleCommunityModel):
    def __init__(self, model_paths, *args, **kwargs):
        self.model_paths = model_paths

        self.community_model_writer = generate_hierarchical_community_model(
            self.model_paths
        )
        self.community_model = self.community_model_writer.convert_to_cobra()
        self.models = self.community_model_writer.models
        self.model_ids = self.community_model_writer.getIds()

        self.shared_exchanges = [e.id for e in self.community_model.exchanges]
        self._type = "compartmentalized"

        self.shuttle_reaction_lower_bound = kwargs.get(
            "shuttle_reaction_lower_bound", -50
        )
        self.shuttle_reaction_upper_bound = kwargs.get(
            "shuttle_reaction_upper_bound", 1000
        )
        self.shuttle_regularization = kwargs.get("shuttle_regularization", True)
        self.shuttle_regularization_factor = kwargs.get(
            "shuttle_regularization_factor", 1000
        )

        self._weights = np.ones(len(self.models))
        self.biomass_id = self.community_model_writer.getAllBiomassFunctons()
        self.biomass_reactions = [
            self.community_model.reactions.get_by_id(g) for g in self.biomass_id
        ]
        self.shuttle_reactions = [
            r for r in self.community_model.reactions if r.id[:3] == "SH_"
        ]
        self.shuttle_reactions_id = [s.id for s in self.shuttle_reactions]
        self._metabolites = self.community_model.metabolites
        self._reactions = self.community_model.reactions
        self._exchanges = self.community_model.exchanges
        self._shuttle_reaction_id_shift = 9

        self.medium = self.community_model_writer.getMedium()
        self._set_weights(self._weights)

        if self.shuttle_regularization:
            self._add_shuttle_regularization()

    # def _get_model_id_from_shuttle_id(self, str):
    #     ids = str.split("__")[1]
    #     for id in self.model_ids:
    #         if id in ids:
    #             return id

    # def _get_shuttle_id_shift(self, model_id):
    #     return 3

    # def _generate_shuttle_id_from_met_id(self, met, model_id):
    #     return "SH__" + model_id + "_M_" + met

    def save_as_sbml(self, path, hierarchical=True):
        """This will save the community model as sbml"""
        if hierarchical:
            self.community_model_writer.save(path)
        else:
            self.community_model_writer.save_flatten(path)

