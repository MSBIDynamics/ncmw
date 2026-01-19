import cobra
from cobra.core import Model
from typing import Tuple, List
import pandas as pd

import subprocess
from warnings import warn
import tempfile, os
from cobra.io import save_json_model, load_json_model

from ncmw.utils import (
    get_default_configs,
    get_default_medium,
    get_biomass_reaction,
    DATA_PATH,
)

def _safe_model_clone(m):
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        save_json_model(m, path)
        return load_json_model(path)
    finally:
        try: os.remove(path)
        except OSError: pass
        
def _monkeypatch_model_copy_for_fastcc():
    def _safe_copy(self):
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            save_json_model(self, path)
            return load_json_model(path)
        finally:
            try: os.remove(path)
            except OSError: pass
    # Ensure COBRA internals use a safe copy during fastcc
    Model.copy = _safe_copy


def gapfill_model(
    model: Model, eps: float = 1e-6, fill_model_base: str = "base", **kwargs
):
    """Adds reactions to the model, such that it has growth in the given medium

    Args:
        model (Model): Cobra model
        eps ([type], optional): Minimum growth value. Defaults to 1e-6.
        fill_model_base (str, optional): The base set of reactions to consider .
        Defaults to "ecoli".
        kwargs (dict, optional): Additional kwargs for cobra gapfilling, seed pycobra
        documentation https://cobrapy.readthedocs.io/en/latest/gapfilling.html

    Returns:
        Model: Cobra model that has growth if algorithm succeeds
        list: List of reactions that were added
    """
    model = _safe_model_clone(model)
    growth = model.slim_optimize()
    if growth > eps:
        # Already has growth gapfilling is unnecessary
        return model, []

    if isinstance(fill_model_base, Model):
        fill_model = fill_model_base
    elif fill_model_base == "base":
        test_model = cobra.io.read_sbml_model(
            DATA_PATH + "/gapfill_baseline/" + "model.xml"
        )
        fill_model = cobra.Model("universal_reactions")
        fill_model.add_reactions(test_model.reactions)
    else:
        raise NotImplementedError()

    if "demand_reactions" in kwargs:
        demand_reactions = kwargs.pop("demand_reactions")
    else:
        demand_reactions = False

    try:
        solution = cobra.flux_analysis.gapfill(
            model, fill_model, demand_reactions=demand_reactions, **kwargs
        )[-1]
        filled_model = _safe_model_clone(model)
        filled_model.add_reactions(solution)
    except:
        warn("The model still has no growth... . We try an alternative")
        filled_model = _safe_model_clone(model)
        _, rec = gapfill_medium(model)

        try:
            solution = []
            for r in rec:
                objective = model.reactions.get_by_id(r).flux_expression
                model.objective = objective
                s = cobra.flux_analysis.gapfill(
                    model, fill_model, demand_reactions=demand_reactions, **kwargs
                )[-1]
                solution.extend(s)
            filled_model.add_reactions(solution)
        except:
            warn("The model still has no growth... . We greadily add sink reactions...")
            solution = []
            for r in rec:
                from cobra import Reaction

                reaction = Reaction("SK" + r[2:])
                solution.append("SK" + r[2:])
                reaction.lower_bound = -0.1
                reaction.upper_bound = 1000
                filled_model.add_reaction(reaction)
                reaction.add_metabolites({r[3:]: -1.0})
                if filled_model.slim_optimize() > eps:
                    break

    assert filled_model.slim_optimize() > eps, "Gapfilling failed..."

    return filled_model, solution


def gapfill_medium(model: Model, eps: float = 1e-1) -> Tuple[Model, List]:
    """This will add the minimal set of exchange reactions such that the model
    has more than eps growth.

    Args:
        model (Model): Cobra model which has less than eps growth
        eps (float, optional): Value for which we consider the model to have zero
                               growth . Defaults to 1e-6.

    Returns:
        Model: Cobra model with extended medium
        list: List of extended metabolites
    """
    model_help = _safe_model_clone(model)

    try:
        if model_help.slim_optimize() > eps:
            # Already feasible model.
            return model, []
    except:
        pass

    # Turn on all exchanges
    for ex in model_help.exchanges:
        ex.lower_bound = -10
    # We can gapfill any exchange reaction that currently is not in the medium
    gapfillable = set([ex.id for ex in model_help.exchanges]).difference(
        set(model.medium.keys())
    )

    # print(f"There are {len(gapfillable)} many metabolites to fill the medium")

    biomass = get_biomass_reaction(model_help)
    # Binary variables: Theta_i
    # This is an indicator which is zero if the metabolite should be added to the medium
    thetas = []
    for i in range(len(gapfillable)):
        thetas.append(model_help.problem.Variable("theta_" + str(i), type="binary"))

    # Constraints for exchanges, which are turned of for theta_i = 1
    theta_constraints = []
    for i, id in enumerate(gapfillable):
        reaction = model_help.reactions.get_by_id(id)
        min_bound = -10
        reaction.lower_bound = min_bound
        cons = model_help.problem.Constraint(
            (reaction.flux_expression + min_bound * thetas[i]), lb=min_bound, ub=1000
        )
        theta_constraints.append(cons)

    # Constraints for growth rates, which must be at least 10% MBR
    constraint_growth = model_help.problem.Constraint(
        biomass.flux_expression, lb=eps, ub=1000
    )

    # Adding new variables and constraints.
    model_help.add_cons_vars(thetas)
    model_help.add_cons_vars(theta_constraints)
    model_help.add_cons_vars(constraint_growth)

    # Objevtive is maximising turned of exchanges, that is sum of theta_is
    objective = model_help.problem.Objective(sum(thetas), direction="max")
    model_help.objective = objective
    model_help.solver.update()

    # Model optimization
    sol = model_help.optimize()
    # Change medium and check if it worked
    new_exchanges = [
        ex.id
        for ex in model_help.exchanges
        if ex.flux < 0 and ex.id not in model.medium
    ]
    extended_medium = model.medium
    for id in new_exchanges:
        extended_medium[id] = 10
    model.medium = extended_medium
    # assert model.slim_optimize() > eps, "The medium extension failed for some reason..."

    return model, new_exchanges


def set_default_configs_and_snm3_medium(
    model: Model, configs: str = "default.json", medium="snm3.json"
):
    """This

    Args:
        model (Model): A cobra model
        configs (str): File name of a config file in json format
        medium (str): File name of a medium file in json format

    Returns:
        Model: Cobra model
    """
    # Set bounds
    configs_dict = get_default_configs(configs)
    medium_dict = get_default_medium(medium)
    for key, val in configs_dict.items():
        key = key.split(".")
        if "reactions" in key[0]:
            for reaction in model.reactions:
                reaction_dic = reaction.__dict__
                if reaction_dic["_" + key[1]] != 0:
                    reaction_dic["_" + key[1]] = val
        else:
            reaction = model.reactions.get_by_id(key[0])
            reaction_dic = reaction.__dict__
            if reaction_dic["_" + key[1]] != 0:
                reaction_dic["_" + key[1]] = val

    # Set medium
    exchanges = [ex.id for ex in model.exchanges]
    model_medium = dict()
    for key in medium_dict:
        if key in exchanges:
            model_medium[key] = medium_dict[key]
    model.medium = model_medium
    return model


def score_memote(model_path: str, report_path: str, solver_timout: str = "120") -> None:
    """Generates a memote evaluation report for the model quality.

    NOTE: This typically requires a rather consistent model, otherwise it breaks.
    NOTE: This can take a while on a local computer, especially if the solver_timeout is high

    Args:
        str: Path to the model file: Typically SBML format
        str: Path to the model file: Typically SBML format
        str: Time in seconds until the solver timeouts.
    """
    try:
        p = subprocess.Popen(
            [
                "memote",
                "report",
                "snapshot",
                "--filename",
                report_path,
                "--solver-timeout",
                solver_timout,
                model_path,
            ]
        )
        return p
    except ValueError(
        "It seems that the model cannot be evaluated or the installation of memote is broken..."
    ):
        print("You may consider to score the model online: https://memote.io/")


def create_consistent_model(model: Model) -> Tuple[Model, pd.DataFrame]:
    """This will create a more consistent model, using fastcc
    (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003424) and
    also returns a summary report with the improvments.

        Args:
            model (cobra.core.Model): Cobra metabolic model class
            verbose (bool, optional): Print out results in console . Defaults to True.

        Returns:
            cobra.core.Model: Returns a more consistent cobra model.
            pd.DataFrame: Returns some statistics that may be imprved.
    """
    blocked_reactions = cobra.flux_analysis.find_blocked_reactions(model)
    met_formulas = cobra.manipulation.check_metabolite_compartment_formula(model)
    mass_charge_balance = cobra.manipulation.check_mass_balance(model)

    consistent_model = fastcc(model)
    blocked_reactions_consistent = cobra.flux_analysis.find_blocked_reactions(
        consistent_model
    )
    met_formulas_consistent = cobra.manipulation.check_metabolite_compartment_formula(
        consistent_model
    )
    mass_charge_balance_consistent = cobra.manipulation.check_mass_balance(
        consistent_model
    )

    data = {
        "Blocked reactions": [
            len(blocked_reactions),
            len(blocked_reactions_consistent),
        ],
        "Metabolite formula problems": [
            len(met_formulas),
            len(met_formulas_consistent),
        ],
        "Mass charge balance violations": [
            len(mass_charge_balance),
            len(mass_charge_balance_consistent),
        ],
    }
    df = pd.DataFrame(data, index=["Original Model", "Consistent model"])

    return consistent_model, df


def fastcc(model: Model) -> Model:
    """FastCC algorithm to increase model quality by resolving conflicts and removing
    unnecessary e.g. blocked pathways
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003424

    Returns:
        cobra.core.Model: Returns a more consistent cobra model.
    """
    _monkeypatch_model_copy_for_fastcc()
    consistent_model = cobra.flux_analysis.fastcc(model)
    return consistent_model
