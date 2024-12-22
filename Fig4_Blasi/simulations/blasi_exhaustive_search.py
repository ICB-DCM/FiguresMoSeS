"""
perform exhaustive search on blasi example
idea:
- initial local search was performed beforehand: report of results stored in timings.tsv at timing_dir-path (
    -> only reads best found model
- starts exhaustive search: iteratively backward selection until all models excluded


in Dictionary selection_arguments all selection and fitting information are passed along:
e.g.:
{'local_method': <Method.BACKWARD: 'backward'>,
 'criterion': <Criterion.AICC: 'AICc'>,
 'nr_data_points': 251,
 'minimize_options': {'n_starts': 10,
      'optimizer': <FidesOptimizer  hessian_update=BFGS verbose=40 options={}>,
      'engine': <pypesto.engine.single_core.SingleCoreEngine at 0x7f076a9c7490>,
      'filename': None},
 'paths': {'petab_yaml': Path(..),
      'timing_dir': Path(..),
      'model_output_path': Path(..),
      'initial_model_path':Path(..),
 'postprocessor': functools.partial(...),
 'best_model_id': 'M_01000100001000010010000000010001', # gets constantly updated if better model is found
 'best_crit_value': -1708.1109869725483}


"""

from functools import partial
from pypesto.select.postprocessors import multi_postprocessor
import logging
import fides
from petab_select import ESTIMATE, Criterion, Method

from exhaustive_search.cluster.blasi.blasi_helpers import *

#####
selection_arguments = {'local_method': Method.BACKWARD,
                       'criterion': Criterion.AICC,
                       'nr_data_points': 251,
                       'minimize_options': {"n_starts": 20,
                                            "optimizer": pypesto.optimize.FidesOptimizer(hessian_update=fides.BFGS(),
                                                                                         verbose=logging.ERROR),
                                            "engine": pypesto.engine.SingleCoreEngine(),
                                            "filename": None,
                                            },
                       'paths': {}
                       }

##########################
##### Initialization #####
##########################

# define output paths
root_path = Path(__file__).parent
output_path = root_path / "results/exhaustive"
output_path.mkdir(parents=True, exist_ok=True)

# Define iteration-specific paths.
save_output_path = output_path / "hdf5"
save_output_path.mkdir(parents=True, exist_ok=True)

timing_output_path = output_path / "timing"
timing_output_path.mkdir(parents=True, exist_ok=True)
timing_output_filepath = timing_output_path / "timings.tsv"
selection_arguments['paths'].update({'timing_dir': timing_output_filepath})

model_output_path = output_path / "models"
model_output_path.mkdir(parents=True, exist_ok=True)
selection_arguments['paths'].update({'model_output_path': model_output_path})

initial_model_filepath = output_path / "initial_models.txt"
# add to selection arguments dict
selection_arguments['paths'].update({'initial_model_path': initial_model_filepath})

# add path to petab problem yaml
selection_arguments['paths'].update({'petab_yaml': Path(root_path / 'example/optimal_problem/output/petab/petab_problem.yaml')})

# Setup postprocessor for saving fitting result
model_postprocessor = partial(
    multi_postprocessor,
    postprocessors=[
        model_id_postprocessor,
        partial(save_postprocessor, output_path=save_output_path),
        partial(timing_postprocessor, output_filepath=timing_output_filepath),
    ],
)

selection_arguments['postprocessor'] = model_postprocessor

# load petab_select problem
petab_select_yaml = root_path / 'example/optimal_problem/output/select/petab_select_problem.yaml'
petab_select_problem = petab_select.Problem.from_yaml(petab_select_yaml)

pypesto_select_problem = pypesto.select.Problem(
    petab_select_problem=petab_select_problem,
    model_postprocessor=selection_arguments['postprocessor'],
)

##########################
### first local search ###
##########################
# use custom FAMoS routine and AICc as the criterion: use report file (i.e. don't reperform local search)
# found true model
# read information from so far calibrated models (initial local search and first backward selection)
checked_models, selection_arguments = load_from_timings(selection_arguments=selection_arguments)

##########################
### local to global ###
##########################
# know: can delete 25 parameters using each criterion -> use last model with 8 parmeters as previous initial model
initial_model_id = find_initial_model(prev_initial_model='M_' + '0'*24+'1'*8,
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

