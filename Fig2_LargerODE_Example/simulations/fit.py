""" explore entire model space for FAMos synth model.
 save (only) optimize_result for each model: M_ijk...
 where i = 1 iff ith parameter is estimated"""

from functools import partial
from pathlib import Path

import numpy as np
import petab_select
import pypesto.petab
import pypesto.optimize
import pypesto.select
import pypesto.store as store
from petab_select import ESTIMATE, Criterion, Method
from petab_select.constants import (
    MODEL_ID,
    TYPE_PATH,
)
from pypesto.select.model_problem import ModelProblem
from pypesto.select.postprocessors import multi_postprocessor


n_starts = 20


def model_id_postprocessor(problem: ModelProblem) -> str:
    """
    Returns M_ijk...,
    where ijk... are 1 if 1/2/3/...th parameters are estimated in the model,
    and zero otherwise.
    """
    model_id = "M_"
    for parameter_value in problem.model.parameters.values():
        model_id += "1" if parameter_value == ESTIMATE else "0"
    problem.model.model_id = model_id


def save_postprocessor(
    problem: ModelProblem,
    output_path: TYPE_PATH = ".",
):
    """own method to save only optimization result.
    Naming: M_0101.. where i = 1 iff ith parameter is estimated"""

    store.write_result(
        problem.minimize_result,
        Path(output_path) / (problem.model.model_id + ".hdf5"),
        overwrite=False,
        problem=False,
        optimize=True,
        profile=False,
        sample=False,
    )


def timing_postprocessor(
    problem: ModelProblem,
    output_filepath: TYPE_PATH,
):
    """own method to save only optimization result.
    Naming: M_0101.. where i = 1 iff ith parameter is estimated"""
    #timing_output_path = Path(output_path) / 'timings.tsv'
    start_optimization_times = problem.minimize_result.optimize_result.time
    model_id = problem.model.model_id
    total_optimization_time_str = str(sum(start_optimization_times))
    start_optimization_times_str = '\t'.join(
        str(start_optimization_time)
        for start_optimization_time in start_optimization_times
    )

    # FIXME arbitrary convergence criterion
    n_converged = str((
        np.array(problem.minimize_result.optimize_result.fval)
        < (
            problem.minimize_result.optimize_result.list[0].fval
            + 0.1
        )
    ).sum())

    row = '\t'.join([
        model_id,
        total_optimization_time_str,
        n_converged,
        start_optimization_times_str,
    ])

    with open(output_filepath, 'a+') as f:
        f.write(row + '\n')


# specify directories
base_dir = Path(__file__).parent
output_dir = base_dir / "output"
petab_select_yaml = (
    base_dir / "FAMoS_2019_PEtab" / "FAMoS_2019_petab_select_problem.yaml"
)

# Define iteration-specific paths.
save_output_path = output_dir / "hdf5"
save_output_path.mkdir(parents=True, exist_ok=True)

timing_output_path = output_dir / "timing"
timing_output_path.mkdir(parents=True, exist_ok=True)
timing_output_filepath = timing_output_path / "timings.tsv"

# Setup postprocessor for saving fitting result
model_postprocessor = partial(
    multi_postprocessor,
    postprocessors=[
        model_id_postprocessor,
        partial(save_postprocessor, output_path=save_output_path),
        partial(timing_postprocessor, output_filepath=timing_output_filepath),
    ],
)

# setup select problems
petab_select_problem = petab_select.Problem.from_yaml(petab_select_yaml)

pypesto_select_problem = pypesto.select.Problem(
    petab_select_problem=petab_select_problem,
    model_postprocessor=model_postprocessor,
)

# setup optimizer settings: fides, 100 starts, multithread engine
minimize_options = {
    "n_starts": n_starts,
    "optimizer": pypesto.optimize.FidesOptimizer(),
    # "engine": pypesto.engine.MultiProcessEngine(),
    "filename": None,
}

# fit entire model space
best_models = pypesto_select_problem.select_to_completion(
    method=Method.BRUTE_FORCE,
    criterion=Criterion.AIC,
    startpoint_latest_mle=True,
    minimize_options=minimize_options,
)


# Fixup timings file.
with open(timing_output_filepath, 'r+') as f:
    header = '\t'.join([
        "model_id",
        "total_time",
        "n_converged",
        *[
            f"start_time_{i}"
            for i in range(n_starts)
        ],
    ])
    data = f.read()
    f.seek(0)
    f.write(header + "\n" + data)
