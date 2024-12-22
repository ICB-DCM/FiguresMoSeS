""" go through remaining models (<=7 parameters) until all models are either excluded or claibrated

selection_arguments: (beispiel)
    {'local_method': <Method.BACKWARD: 'backward'>,
    'criterion': <Criterion.AICC: 'AICc'>,
    'nr_data_points': 251,
    'minimize_options': {'n_starts': 20,
                        'optimizer': <FidesOptimizer  hessian_update=BFGS verbose=40 options={}>,
                        'engine': <pypesto.engine.single_core.SingleCoreEngine object at 0x7f61e82bbfd0>,
                        'filename': None},
    'paths': {'timing_dir': PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/results/AICC/timing/timings_AICC.tsv'),
            'model_output_path': PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/results/AICC/models'),
            'initial_models_dir': PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/../check_for_not_excluded_models/AICC'),
             'current_state_path': PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/results/AICC/current_state.txt'),
             'petab_yaml': PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/../example/optimal_problem/output/petab/petab_problem.yaml')},
    'best_model_id': 'M_01000100001000010010000000010001',
     'best_crit_value': 48.9065945119969,
     'postprocessor': functools.partial(<function multi_postprocessor at 0x7f61c4d11870>, postprocessors=[<function model_id_postprocessor at 0x7f61c49292d0>, functools.partial(<function save_postprocessor at 0x7f61c4929360>, output_path=PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/results/AICC/hdf5')), functools.partial(<function timing_postprocessor at 0x7f61c49293f0>, output_filepath=PosixPath('/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/remaining_global_search/results/AICC/timing/timings_AICC.tsv'))])
     }

"""

import sys
sys.path.insert(0, '/home/toni/Workspace/MoSeS/')
from exhaustive_search.cluster.blasi.blasi_helpers import *

from functools import partial
from pypesto.select.postprocessors import multi_postprocessor
import logging
import fides
from petab_select import Criterion, Method



def read_remaining_models(files_dir: Path) -> np.ndarray:
    """files_dir: path to directory where files with the initial models are"""

    free_models = np.empty(33, dtype=np.ndarray)
    for i in range(33):
        free_models[i] = np.empty(0, str)

    for model_size in range(8):
        # read remaining models
        remaining_models = np.genfromtxt(files_dir / f'initial_models_{model_size}.txt',
                                         dtype=str,
                                         delimiter='\n',
                                         skip_header=1  # ignoriere erste Zeile ('model_id')
                                         )

        free_models[model_size] = remaining_models

    return free_models



##########################
##### Initialization #####
##########################

selection_arguments = {'local_method': Method.BACKWARD,
                       'criterion': Criterion.AICC,
                       'nr_data_points': 251,
                       'minimize_options': {"n_starts": 20,
                                            "optimizer": pypesto.optimize.FidesOptimizer(hessian_update=fides.BFGS(),
                                                                                         verbose=logging.ERROR),
                                            "engine": pypesto.engine.SingleCoreEngine(),
                                            "filename": None,
                                            },
                       'paths': {},
                       'best_model_id': 'M_01000100001000010010000000010001',
                       'best_crit_value': 48.9065945119969  # aicc
                       }

# define output paths
root_path = Path(__file__).parent
output_path = root_path / f"results/{selection_arguments['criterion'].name}"
output_path.mkdir(parents=True, exist_ok=True)

# Define iteration-specific paths.
save_output_path = output_path / "hdf5"
save_output_path.mkdir(parents=True, exist_ok=True)

timing_output_path = output_path / "timing"
timing_output_path.mkdir(parents=True, exist_ok=True)
timing_output_filepath = timing_output_path / f"timings_{selection_arguments['criterion'].name}.tsv"
selection_arguments['paths'].update({'timing_dir': timing_output_filepath})

model_output_path = output_path / "models"
model_output_path.mkdir(parents=True, exist_ok=True)
selection_arguments['paths'].update({'model_output_path': model_output_path})

initial_models_dir = root_path / '../check_for_not_excluded_models/AICC'
# add to selection arguments dict
selection_arguments['paths'].update({'initial_models_dir': initial_models_dir})

selection_arguments['paths'].update({'current_state_path': output_path / Path('current_state.txt')})

# add path to petab problem yaml
selection_arguments['paths'].update(
    {'petab_yaml': Path(root_path / '../example/optimal_problem/output/petab/petab_problem.yaml')})


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

# get list of free models <= 7 parameters
free_models = read_remaining_models(selection_arguments['paths']['initial_models_dir'])

# go through remaining model space (models <= 7 parameters)
for model_size in reversed(range(8)):
    add_initial2file(selection_arguments['paths']['current_state_path'],
                     f'calibrating models of size {model_size}. In total: {len(free_models[model_size])} models',
                     time=True)

    for model in free_models[model_size]:
        crit_value = fit_model(model, selection_arguments)
        add_initial2file(selection_arguments['paths']['current_state_path'],
                         f'\t remaining free models: {get_len_array_of_arays(free_models)}',
                         time=False)

        # if model performs better than so far global model
        if selection_arguments['best_crit_value'] > crit_value:
            add_initial2file(selection_arguments['paths']['current_state_path'], f'updated best model : now {model}')
            global_best_model_id = model
            global_best_crit_value = crit_value
            selection_arguments['best_model_id'] = global_best_model_id
            selection_arguments['best_crit_value'] = global_best_crit_value

        # exclude as many models as possible
        nr_paras_to_delete = get_nr_paras_to_delete(model, selection_arguments)
        for submodel_size in reversed(range(max(0, model_size - nr_paras_to_delete), model_size)):
            for submodel in free_models[submodel_size]:
                if is_submodel(supermodel=model, submodel=submodel):
                    # delete model from list
                    free_models[submodel_size] = np.setdiff1d(
                        ar1=free_models[submodel_size],
                        ar2=[submodel])

