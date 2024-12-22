""" for all 3 criteria go through all models with <=7 parameters and check if they are not excluded nor calibrated yet """


from concurrent.futures import ThreadPoolExecutor
import math
from multiprocessing import Pool
import os
import itertools

from petab_select import Criterion, Method

import sys


sys.path.insert(0, '/home/toni/Workspace/MoSeS/')
from exhaustive_search.cluster.blasi.blasi_helpers import *


root_path = Path(__file__).parent \

output_path = root_path / 'test_check_not_excluded_models_size_2'

selection_arguments = {'local_method': Method.BACKWARD,
                       'criterion': Criterion.AICC, # change criterion here
                       'nr_data_points': 251,
                       'paths': {'timing_dir': root_path / Path('timings.tsv'),
                                 'initial_model_path': root_path / Path('initial_models.txt'),
                                 'current_state_path': root_path / Path('current_state.txt'),
                                 'excluded_model_path': root_path / Path('excluded_models.txt')}
                       }

N_PARAMETERS = 32
N_WORKERS = os.cpu_count()

#POOL_CLASS = ThreadPoolExecutor
POOL_CLASS = Pool

# load so far gained information from timings
checked_models, selection_arguments = load_from_timings(selection_arguments=selection_arguments)


def check_model(one_indices,
                checked_models=checked_models,
                selection_arguments=selection_arguments):
    # print('fct call')

    model_id = one_indices_to_model_id(one_indices)
    if not model_already_excluded(model_id, checked_models, selection_arguments):
        # found model with 7 parameters that has not been excluded yet
        add_initial2file(filepath=selection_arguments['paths']['initial_model_path'],
                         model_id=model_id,
                         time=False)
        return model_id

    add_initial2file(filepath=selection_arguments['paths']['excluded_model_path'], model_id=model_id, time=False)
    return None


def check_all_models_of_given_size(n_free_parameters: int):
    one_indices_iterator = itertools.combinations(range(N_PARAMETERS), r=n_free_parameters)
    chunksize = round(1 + math.comb(N_PARAMETERS, n_free_parameters) / N_WORKERS)
    with POOL_CLASS(N_WORKERS) as pool:
       results = pool.map(check_model, one_indices_iterator, chunksize=chunksize)



#### start: code ####

for crit in [Criterion.AICC]:  #, Criterion.AIC, Criterion.BIC]:
    selection_arguments['criterion'] = crit
    add_initial2file(filepath=selection_arguments['paths']['current_state_path'],
                     model_id=f'\n\nchecking for criterion {crit}',
                     time=True)
    # update paths
    crit_res_path = root_path / Path(crit.name)
    crit_res_path.mkdir(parents=True, exist_ok=True)

    for MODEL_SIZE in [0]:
        # get list of models seperated by model size
        selection_arguments['paths']['initial_model_path'] = crit_res_path / Path(f'initial_models_{MODEL_SIZE}.txt')
        selection_arguments['paths']['excluded_model_path'] = crit_res_path / Path(f'excluded_models_{MODEL_SIZE}.txt')

        add_initial2file(filepath=selection_arguments['paths']['current_state_path'],
                         model_id=f'checking models with size {MODEL_SIZE}',
                         time=True)
        check_all_models_of_given_size(MODEL_SIZE)



#
#
# while model_id  and get_nr_parameters(model_id)==4:
#     if not model_already_excluded(model_id, checked_models, selection_arguments):
#         # found model with 7 parameters that has not been excluded yet
#         add_initial2file(filepath=selection_arguments['paths']['initial_model_path'],
#                          model_id=model_id,
#                          time=False)
#     else:
#         add_initial2file(filepath=root_path / Path('excluded.txt'),
#                          model_id=model_id,
#                          time=False)
#
#     # get next model id = subsequent permutation
#     model_id = get_next_possible_initial_model(model_id)
#
#     model_count += 1
#     if model_count == 10000:
#         model_count = 0
#         # write to current state file: feedback of search progress
#         add_initial2file(filepath=selection_arguments['paths']['current_state_path'],
#                          model_id=f'currently checking: {model_id}')
#
# add_initial2file(filepath=selection_arguments['paths']['current_state_path'],
#                  model_id='done with search')

