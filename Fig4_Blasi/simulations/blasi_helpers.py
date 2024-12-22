import math
import petab
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
import tqdm

from typing import Dict, Iterable, Union, Tuple, List

import pypesto.select
import pypesto.store as store
from pypesto.select.model_problem import ModelProblem
from pypesto.select.misc import correct_x_guesses

import petab_select
from petab_select import ESTIMATE, Model
import petab_select.ui
from petab.C import ESTIMATE
from petab_select.constants import (
    TYPE_PATH,
    Criterion,
    PETAB_PROBLEM
)

from exhaustive_search.cluster.blasi.example.blasi_pypesto_problem import get_pypesto_problem



CRITERIA = [
    Criterion.NLLH,
    Criterion.AIC,
    Criterion.AICC,
    Criterion.BIC,
]

N_PARAMETERS = 32


### save stuff ####
def model_to_pypesto_problem(
        model: Model,
        x_guesses: Iterable[Dict[str, float]] = None,
) -> pypesto.Problem:
    petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]

    corrected_x_guesses = correct_x_guesses(
        x_guesses=x_guesses,
        model=model,
        petab_problem=petab_problem,
    )

    pypesto_problem = get_pypesto_problem(
        petab_problem=petab_problem,
        x_guesses=corrected_x_guesses,
    )

    """
    importer = PetabImporter(petab_problem)
    if objective is None:
        objective = importer.create_objective()
    pypesto_problem = importer.create_problem(
        objective=objective,
        x_guesses=corrected_x_guesses,
    )
    """
    return pypesto_problem


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
        overwrite=True,
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
    # timing_output_path = Path(output_path) / 'timings.tsv'
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

    # compute all criteria (needed for fitting single model routine)
    for criterion in CRITERIA:
        problem.model.compute_criterion(criterion)

    # create row with information to current model
    row = '\t'.join([
        model_id,
        total_optimization_time_str,
        *[str(problem.model.get_criterion(criterion)) for criterion in CRITERIA],
        n_converged,
        start_optimization_times_str,
    ])

    with open(output_filepath, 'a+') as f:
        f.write(row + '\n')


def prettify_timings(timing_output_filepath: Path, n_starts):
    """add column headings to timings.tsv table"""
    with open(timing_output_filepath, 'r+') as f:
        header = '\t'.join([
            "model_id",
            "total_time",
            *[criterion.name for criterion in CRITERIA],
            "n_converged",
            *[
                f"start_time_{i}"
                for i in range(n_starts)
            ],
        ])
        data = f.read()
        f.seek(0)
        f.write(header + "\n" + data)


def save_fitted_models_to_yaml(models: List[petab_select.model.Model],
                               model_output_path: Path):
    """ for all given models -> convert to dict -> save as yaml
    """
    # save all fitted models
    for model in models:
        model.to_yaml(
            model_output_path / (model.model_id + ".yaml"),
            paths_relative_to=model_output_path,
        )


### helpers local searches ###
def get_nr_parameters(model_id: str) -> int:
    """returns number of parameters set in model"""
    return sum([int(i) for i in list(model_id[2:])])


def is_submodel(supermodel: str, submodel: str):
    """ checks if submodel is true submodel of supermodel"""
    for i in range(len(supermodel) - 2):
        if supermodel[2 + i] == '0' and submodel[2 + i] == '1':
            return False
    return True


def get_submodels(model_id: str, checked_models: np.ndarray, selection_arguments: Dict) -> list:
    """ get submodels of model that are not excluded yet by any other already calibrated model
    a submodel is a model with one parameter less set than given model

    returns: list of submodels, empty if all submodels are excluded by other models"""
    submodels = []

    # get indices of the 1s in model_id
    one_indices = [k + 2 for k in range(len(model_id[2:])) if model_id[k + 2] == '1']
    for index_of_one in one_indices:
        submodel = model_id[:index_of_one] + '0' + model_id[index_of_one + 1:]
        if not model_already_excluded(current_model=submodel,
                                      checked_models=checked_models,
                                      selection_arguments=selection_arguments):
            submodels.append(submodel)

    return submodels


def get_nr_paras_to_delete(current_model: str,
                           selection_arguments: Dict) -> int:
    """ determines number of parameters to delete in order to get a possibly better performing submodel of current_model
    rounds up computed values

     parameters:
     curren_model: index list of model i.e. binary list
     selection_arguments: all parameters and settings for selection process,
            in particular used criterion, sofar best model, number of data points,...

    returns:
    number of parameters to delete
    """
    criterion = selection_arguments['criterion'].name
    best_criterion = selection_arguments['best_crit_value']
    nr_data_points = selection_arguments['nr_data_points']

    # read timing_df
    timing_df = pd.read_csv(selection_arguments['paths']['timing_dir'], sep='\t')
    timing_df.set_index('model_id', inplace=True)

    current_model_criterion = timing_df.loc[current_model][criterion]


    if criterion == 'AIC':
        # number of parameters to delete (rounded up)
        nr_paras_to_remove = math.ceil(1 / 2 * (current_model_criterion - best_criterion))
    elif criterion == 'BIC':
        # number of parameters to delete (rounded up)
        # pypesto.select routine uses log (not log10)
        nr_paras_to_remove = math.ceil(1 / np.log(nr_data_points) * (current_model_criterion - best_criterion))
    elif criterion == 'AICC':  # criterion == 'AICc'
        # get log likelihood of current model
        # Jakobs solution
        # llh_current = - timing_df.loc[current_model]['NLLH']
        # numerator = (nr_data_points - 1) * (llh_current + 0.5 * best_criterion)
        # denominator = nr_data_points + llh_current + best_criterion
        # # number of parameters to delete (rounded up)
        # nr_paras_to_remove = int(numerator / denominator) + 1

        # my solution
        current_nr_paras = get_nr_parameters(current_model)
        current_aic = timing_df.loc[current_model]['AIC']

        numerator = (current_aic-best_criterion) * (nr_data_points - current_nr_paras - 1) + 2 * current_nr_paras * (current_nr_paras + 1)
        denominator = best_criterion-current_aic + 2 * nr_data_points + 2 * current_nr_paras
        nr_paras_to_remove = math.ceil(numerator / denominator)
    else:
        # not implemented yet for other criteria. thus delete 0 parameters
        print(f'nr fo parameters to delete not implemented yet for criterion {criterion}')
        nr_paras_to_remove = 0

    return nr_paras_to_remove


def model_already_excluded(current_model: str, checked_models: np.ndarray, selection_arguments: Dict) -> bool:
    """check if model has been checked or can be excluded from already checked model, else return that not excluded yet

    parameters:
    current_model: M_101101...
    checked_models: array containing arrays containing model ids of all models that have been calibrated
    selection_arguments

    returns: True if model has already been calibrated or can be excluded from already checked models
            False if model is still unknown
    """

    nr_free_parameters = get_nr_parameters(current_model)
    if current_model in checked_models[nr_free_parameters]:
        return True

    # go through bigger models check if current model can be excluded from it
    for i in range(nr_free_parameters + 1, 33):
        for model in checked_models[i]:
            # check if current_model is submodel of model
            if is_submodel(supermodel=model, submodel=current_model):
                # otherwise we can't definitely not exclude current model
                if get_nr_parameters(model) - nr_free_parameters <= get_nr_paras_to_delete(current_model=model,
                                                                                           selection_arguments=
                                                                                                selection_arguments):
                    # current_model is within an area beneath the model where we can say for sure that current_model is
                    # no improvement
                    return True
    # current_model could not be excluded through information on already checked models
    return False


def add_models_to_checkedmodels(checked_models: np.array, models2add: List) -> np.array:
    """add models that have been checked to list of checked models

    parameters:
    checked_models: at ith position array with already checked models with i parameters free for estimation
    candidates: list of model ids

    return:
    updated checked_models list
    """
    for model_id in tqdm.tqdm(models2add):
        i = get_nr_parameters(model_id)
        checked_models[i] = np.append(checked_models[i], model_id)

    return checked_models


def add_initial2file(filepath: Path, text: str, time: bool = True):
    """writes model to file to keep track of progress
    i.e. in the file (_filepath_) is list of all initial models and time at which corresponding local search was started
     """
    if time:
        with open(filepath, 'a+') as f:
            f.write(text + f' {datetime.now()}\n')
    else:
        with open(filepath, 'a+') as f:
            f.write(text + '\n')


### find next initial model ###
def block_of_ones(array: list, i: int) -> bool:
    """ checks if array contains a block of i times 1 and no other 1 i.e. array = 0^* + 1^i + 0^* """
    first_one = False
    second_zero = False

    if array.count(1) != i:
        # block of i times 1 is impossible
        return False

    for index in range(len(array)):
        # go through array and check if all ones occur consecutively
        if array[index] and first_one is False:
            # first time 1
            first_one = True
        elif array[index] and first_one is True and second_zero is False:
            # in first block of ones
            first_one = True
        elif array[index] and first_one is True and second_zero is True:
            # array includes 0^* + 1^+ + 0^+ + 1^+ -> no blck of i times 1 possible
            return False
        elif not array[index] and first_one is True:
            second_zero = True

    return True


def swap_and_align(array: list, a: int, b: int) -> list:
    """ swaps array[a] (=1) and array[b] (=0) and aligns 'ones' left of position a to left border
         makes deep copy of the given array
         returns swapped array """
    array_output = copy.deepcopy(array)
    # swap
    temp_a = array[a]
    temp_b = array[b]
    array_output[a] = temp_b
    array_output[b] = temp_a

    # align ones left of the position where we swapped to left border, e.g. 01100100 -> 10010100
    nr_ones_left = array_output[0:a].count(1)
    array_output[0:nr_ones_left] = [1] * nr_ones_left
    array_output[nr_ones_left:a] = [0] * (a - nr_ones_left)

    return array_output


def get_steplen(model_id: str) -> int:
    """ get position of last one (not index, i.e. index of most right 'one' +1) only in number part of model_id
    returns i when last one is ith element in number part of model id
    """
    # get list of  indices of all 'ones' in model_id (only of number part of model_id)
    one_indices = [k for k in range(len(model_id[2:])) if model_id[k + 2] == '1']

    if len(one_indices) > 0:
        return one_indices[-1] + 1
    else:
        # no 'one' in model_id
        return -1


def one_indices_to_model_id(one_indices) -> str:
    return 'M_' + ''.join([
        '1' if one_index in one_indices else '0'
        for one_index in range(N_PARAMETERS)
    ])


def get_next_possible_initial_model(prev_init_model: str) -> Union[str, None]:
    """ finds next model (in list of possible permutations)
    list of possible permutations: goes from all to no parameters set and in each group of models with same number of
        set parameters it goes sort of in reversed binary order
        e.g. 1111, 1110, 1101, 1011, 0111, 1100, 1010, 0110, 1001, 0101, 0011, 1000, 0100, 0010, 0001, 0000
    returns: - model_id of next possible initial model if existing
             - empty string (`''`) if no model exists after given prev_init_model
    """
    one_indices = [index - 2 for index, value in enumerate(prev_init_model) if value == '1']
    n_free_parameters = len(one_indices)

    if n_free_parameters == 0:
        # At the smallest possible model, so terminate
        return ''

    if prev_init_model.endswith('1'*n_free_parameters):
        # At the last model of this size, so return first model of `n-1` size
        return one_indices_to_model_id(range(n_free_parameters - 1))

    n_free_parameters_combinations = itertools.combinations(range(N_PARAMETERS), r=n_free_parameters)
    # Set index of iterator to position of `prev_init_model` in iterator. `assert` is unnecessary but may as well...
    assert tuple(one_indices) in n_free_parameters_combinations
    return one_indices_to_model_id(next(n_free_parameters_combinations))


def find_initial_model(prev_initial_model: Union[str, None],
                       checked_models: np.array,
                       selection_arguments: Dict) -> Union[str, None]:
    """ finds next initial model that has not been excluded. Starts search with previous initial model
    ( goes through candidate space complex to simple)
    if None is given as input: return M_11...11 (the very first initial model)
    during search: writes every 10000th model that is checked to initial model file as feedback of current state

    returns model_id of new initial model or None if all models have been excluded"""

    if not prev_initial_model:
        # i.e. prev_init_model=None i.e. we want to find the very first initial model
        next_initial_model = 'M_' + '1' * 32
        if next_initial_model in checked_models[get_nr_parameters(next_initial_model)]:
            # we are in restart after crash -> load last initial model from initial model file
            # thus we restart with local search during which things crashed (recalibrate some models)
            with open(selection_arguments['paths']['initial_model_path'], 'r') as f:
                last_line = f.readlines()[-1]
            next_initial_model = last_line.split(' ')[0]
            add_initial2file(selection_arguments['paths']['initial_model_path'], next_initial_model)
            return next_initial_model
        else:
            # we are in first attempt to do exhaustive search
            add_initial2file(selection_arguments['paths']['initial_model_path'], next_initial_model)
            return next_initial_model
    elif prev_initial_model == 'M_' + '1'*32:
        # we can exclude all submodels with up to nr_of_paras_to_delete parameters less
        nr_para_to_delete = get_nr_paras_to_delete(current_model=prev_initial_model,
                                                   selection_arguments=selection_arguments)
        # can jump directly to end of area under complete model that can be excluded due to fitting results
        prev_initial_model = 'M_' + '0'*(nr_para_to_delete-1) + '1'*(32-nr_para_to_delete+1)

    next_initial_model = get_next_possible_initial_model(prev_initial_model)
    if not next_initial_model:
        # reached end of candidate space
        return None

    model_count = 0
    while model_already_excluded(current_model=next_initial_model,
                                 checked_models=checked_models,
                                 selection_arguments=selection_arguments):
        # find next possible initial model
        next_initial_model = get_next_possible_initial_model(next_initial_model)
        model_count +=1
        if model_count == 10000:
            model_count = 0
            add_initial2file(filepath=selection_arguments['paths']['initial_model_path'],
                             model_id=f'searching next initial model. currently checking: {next_initial_model}',)

        if not next_initial_model:
            # reached end of candidate space i.e. no possible new initial models left -> done with global search
            # write to initial model file that search has ended
            add_initial2file(filepath=selection_arguments['paths']['initial_model_path'],
                             model_id='end of global search')
            return None

    # write to initial model output file
    add_initial2file(selection_arguments['paths']['initial_model_path'], next_initial_model)
    # return next less complex model that has not been excluded yet
    return next_initial_model


### local search ###
def local_search(initial_model_id: str,
                 checked_models: np.ndarray,
                 selection_arguments: Dict,
                 after_crash: Union[float, None] = None
                 ) -> Tuple[str, float, np.ndarray]:
    """ perform local search: for each model: get submodels and fit them individually
    if no submodel is performing better than 'parent model': stop local search
    keep track of models checked during local search separately and add them general list of checked models after
        ending local search

    parameters: after_crash: normally= None,
        if loacl search is called after crash, after_crash=criterion value from model where to restart local search,
        initial model is than the corresponding model id

    return: local_best_model: id of best found model
            local_best_criterion_value: criterion value of best performing model
            checked_models: updated list of checked models

    # todo implement smarter way ? : avoid calibration of models from which we know that they are worse than given model ->
        # dont look for direct submodels but for submodels with n parameters less (n=nr_paras_to_delete)
    """
    # fit initial model
    local_best_model_id = initial_model_id
    if not after_crash:
        local_best_crit_value = fit_model(model_id=initial_model_id, selection_arguments=selection_arguments)
    else:
        local_best_crit_value = after_crash
    local_checked_models = [local_best_model_id]

    better_submodel = True
    while better_submodel:
        better_submodel = False
        submodels = get_submodels(model_id=local_best_model_id,
                                  checked_models=checked_models,
                                  selection_arguments=selection_arguments)

        # fit submodels
        for model_id in submodels:
            crit_value = fit_model(model_id=model_id, selection_arguments=selection_arguments)
            # add to checked models
            local_checked_models = local_checked_models + [model_id]

            if local_best_crit_value > crit_value:
                better_submodel = True
                # get best performing submodel
                local_best_crit_value = crit_value
                local_best_model_id = model_id

    # add models checked during local search to general list of checked models
    checked_models = add_models_to_checkedmodels(checked_models=checked_models, models2add=local_checked_models)

    return local_best_model_id, local_best_crit_value, checked_models


def fit_model(model_id: str, selection_arguments: Dict) -> float:
    """ get model_id -> fit model -> save fitting results (use postprocessor)
    return criterion value
    """
    parameter_names = ['a_0ac_k05', 'a_0ac_k08', 'a_0ac_k12', 'a_0ac_k16',
                       'a_k05_k05k08', 'a_k05_k05k12', 'a_k05_k05k16', 'a_k08_k05k08', 'a_k08_k08k12', 'a_k08_k08k16',
                       'a_k12_k05k12', 'a_k12_k08k12', 'a_k12_k12k16', 'a_k16_k05k16', 'a_k16_k08k16', 'a_k16_k12k16',
                       'a_k05k08_k05k08k12', 'a_k05k08_k05k08k16', 'a_k05k12_k05k08k12', 'a_k05k12_k05k12k16',
                       'a_k05k16_k05k08k16', 'a_k05k16_k05k12k16', 'a_k08k12_k05k08k12', 'a_k08k12_k08k12k16',
                       'a_k08k16_k05k08k16', 'a_k08k16_k08k12k16', 'a_k12k16_k05k12k16', 'a_k12k16_k08k12k16',
                       'a_k05k08k12_4ac', 'a_k05k08k16_4ac', 'a_k05k12k16_4ac', 'a_k08k12k16_4ac']

    # prepare parameter dict according to given model_id
    parameter_dict = {}
    for i in range(32):
        if model_id[i + 2] == '1':
            parameter_dict[parameter_names[i]] = 'estimate'
        else:
            parameter_dict[parameter_names[i]] = 1
    petab_problem = petab.Problem.from_yaml(selection_arguments['paths']['petab_yaml'])
    # create model instance
    model = Model(
        petab_yaml=selection_arguments['paths']['petab_yaml'],
        parameters=parameter_dict,
        petab_problem=petab_problem,
    )
    # update model
    model.model_id = model_id

    # calibrate model
    model_problem = ModelProblem(
        model=model,
        criterion=selection_arguments['criterion'],
        minimize_options=selection_arguments['minimize_options'],
        postprocessor=selection_arguments['postprocessor'],
        model_to_pypesto_problem_method=model_to_pypesto_problem,
    )
    # save model to yaml
    save_fitted_models_to_yaml(models=[model], model_output_path=selection_arguments['paths']['model_output_path'])

    return model_problem.model.criteria[selection_arguments['criterion']]


### restart after stop ###
def load_from_timings(selection_arguments: Dict) -> Tuple[np.ndarray, Dict]:
    """ read so far - best performing model id + criterion value
                    - checked models
             from timings.tsv
        return: checked models, updated selection_arguments
    """
    timing_df = pd.read_csv(selection_arguments['paths']['timing_dir'], sep='\t')
    timing_df.set_index('model_id', inplace=True)
    # get id
    selection_arguments['best_model_id'] = timing_df[selection_arguments['criterion'].name].idxmin()
    # get corresponding criterion value
    selection_arguments['best_crit_value'] = timing_df.loc[selection_arguments['best_model_id']][selection_arguments['criterion'].name]

    # create empty array to track checked models
    checked_models = np.empty(33, dtype=np.ndarray)
    for i in range(33):
        checked_models[i] = np.empty(0, str)

    # add checked models to list of checked models
    checked_models = add_models_to_checkedmodels(checked_models, timing_df.index)

    return checked_models, selection_arguments


def exhaustive_after_crash(selection_arguments: Dict):
    """ restores state of exhaustive search and completes search"""
    # load so far checked models and information on so far best model from timings df
    checked_models, selection_arguments = load_from_timings(selection_arguments=selection_arguments)

    # restore state of local search
    # load timings df
    timing_df = pd.read_csv(selection_arguments['paths']['timing_dir'], sep='\t')
    timing_df.set_index('model_id', inplace=True)
    # get last fitted model
    i=-1
    latest_model = timing_df.index[i]
    current_best_model = latest_model
    current_best_crit = timing_df.loc[current_best_model][selection_arguments['criterion'].name]
    latest_nr_paras = get_nr_parameters(latest_model)
    i=-2
    current_model = timing_df.index[i]
    while get_nr_parameters(current_model)==latest_nr_paras:
        # go through all models and find common supermodel and best performing submodel
        current_crit = timing_df.loc[current_model][selection_arguments['criterion'].name]
        if current_crit  < current_best_crit:
            current_best_crit = current_crit
            current_best_model = current_model
        i = i-1
        current_model = timing_df.index[i]
    # current_model is supermodel, resume local search from current_best_model
    local_best_model_id, local_best_crit_value, checked_models = local_search(initial_model_id=current_best_model,
                                                                              checked_models=checked_models,
                                                                              selection_arguments=selection_arguments,
                                                                              after_crash=current_best_crit)

    # check if model found with local search performs better than global best model
    if selection_arguments['best_crit_value'] > local_best_crit_value:
        global_best_model_id = local_best_model_id
        global_best_crit_value = local_best_crit_value
        selection_arguments['best_model_id'] = global_best_model_id
        selection_arguments['best_crit_value'] = global_best_crit_value

    # resume as always
    # find next initial model
    initial_model_id = find_initial_model(prev_initial_model=None,
                                          checked_models=checked_models,
                                          selection_arguments=selection_arguments)

    while initial_model_id:
        # as long as we find a model that has not been checked so far: continue to perform local searches
        local_best_model_id, local_best_crit_value, checked_models = local_search(initial_model_id=initial_model_id,
                                                                                  checked_models=checked_models,
                                                                                  selection_arguments=selection_arguments)

        # check if model found with local search performs better than global best model
        if selection_arguments['best_crit_value'] > local_best_crit_value:
            global_best_model_id = local_best_model_id
            global_best_crit_value = local_best_crit_value
            selection_arguments['best_model_id'] = global_best_model_id
            selection_arguments['best_crit_value'] = global_best_crit_value

        # find next initial model
        initial_model_id = find_initial_model(prev_initial_model=initial_model_id,
                                              checked_models=checked_models,
                                              selection_arguments=selection_arguments)


def get_len_array_of_arays(array: np.ndarray) -> int:
    length = 0
    for i in range(len(array)):
        length += len(array[i])

    return length