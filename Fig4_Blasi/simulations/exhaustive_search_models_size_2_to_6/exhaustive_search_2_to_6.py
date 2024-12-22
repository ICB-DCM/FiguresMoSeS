"""
consider models size 6 to 2
we have:
    - all models calibrated. (given in timings directory) Caution: check for # of converged starts -> < 5/20 -> check in uncalibrated table
    #todo: not all models calibrated -> which not?

want:
    - remove models that could have been excluded by calibration result of previous (super) models: exclude models!
        -> which models need to be calibrated?
            - auf jeden Fall: all models of size 6
        - results:
            - 1 timings.tsv
            - 1 txt file with list of model ids that were not calibrated

"""
import datetime
from typing import Dict

import numpy as np
import pandas as pd
from pathlib import Path
import math
import tqdm


def get_nr_parameters(model_id: str) -> int:
    """returns number of parameters set in model"""
    return sum([int(i) for i in list(model_id[2:])])


def get_nr_paras_to_delete_AICC(current_model: str,
                                timings_dict: Dict) -> int:
    """ determines number of parameters to delete in order to get a possibly better performing submodel of current_model


     parameters:
     curren_model:

    returns:
    number of parameters to delete
    """
    current_nr_paras = get_nr_parameters(current_model)

    best_criterion = 48.9065945119969  # aicc value of M_01000100001000010010000000010001 (the true model)
    nr_data_points = 251

    # check that at least 5 times converged
    if timings_dict[f'df_{current_nr_paras}'].loc[current_model]['n_converged'] < 5:
        # read from uncalibrated df (where models are calibrated with 200 starts)
        current_aic = timings_dict[f'df_uncalibrated'].loc[current_model]['AIC']
    else:
        current_aic = timings_dict[f'df_{current_nr_paras}'].loc[current_model]['AIC']

    numerator = (current_aic - best_criterion) * (nr_data_points - current_nr_paras - 1) + 2 * current_nr_paras * (
            current_nr_paras + 1)
    denominator = best_criterion - current_aic + 2 * nr_data_points + 2 * current_nr_paras
    return math.ceil(numerator / denominator)


def get_submodels_to_remove(model_id: str, nptd: int) -> set:
    """
    computes a set of all submodels with up to nptd less parameters (recursive)

    parameters:
        model_id: str: M_0101...
        nptd: int: nr of parameters to delete

    returns:
        list of submodel_ids
    """
    submodels = set([])

    if nptd == 0:
        return submodels

    # get indices of the 1s in model_id
    one_indices = [k + 2 for k in range(len(model_id[2:])) if model_id[k + 2] == '1']

    # replace iteratively: each time one '1' by a '0' -> get submodel_id
    for one_index in one_indices:
        submodel_id = model_id[:one_index] + '0' + model_id[one_index + 1:]
        # Union of sets
        submodels = submodels | {submodel_id} | get_submodels_to_remove(submodel_id, nptd - 1)

    return submodels


def write_progress(timings_dict: Dict, add_text: str = None):
    """ either write additional text add_text or remainingsizes of dfs"""

    file_dir = '/home/toni/Workspace/MoSeS/exhaustive_search/cluster/' \
               'blasi/exhaustive_search_models_size_2_to_6/progress.txt'

    if add_text:
        progress = f"{add_text} \t ({datetime.datetime.now()}) \n"
    else:
        progress = f"{datetime.datetime.now()}\n"

        for size in reversed(range(2, 7)):
            progress += f"\t df_{size}: {len(timings_dict[f'df_{size}'])} models remain \n"

    progress += "\n"
    f = open(file_dir, 'a')
    f.write(progress)
    f.close()


def read_all_dfs(df_base_dir: Path) -> Dict:
    # read all timing dfs
    df_6 = pd.read_csv(df_base_dir / '6.tsv', sep='\t').set_index('model_id')
    df_5 = pd.read_csv(df_base_dir / '5.tsv', sep='\t').set_index('model_id')
    df_4 = pd.read_csv(df_base_dir / '4.tsv', sep='\t').set_index('model_id')
    df_3 = pd.read_csv(df_base_dir / '3.tsv', sep='\t').set_index('model_id')
    df_2 = pd.read_csv(df_base_dir / '2.tsv', sep='\t').set_index('model_id')
    # empty dummy DataFrames
    df_1 = pd.DataFrame()
    df_0 = pd.DataFrame()

    # and models which did not converge min. 5  times after 20 starts
    uncalibrated_df = pd.read_csv(df_base_dir / 'uncalibrated.tsv', sep='\t').set_index('model_id')

    # write all together in one dict
    return {'df_6': df_6, 'df_5': df_5, 'df_4': df_4, 'df_3': df_3, 'df_2': df_2, 'df_1': df_1, 'df_0': df_0,
            'df_uncalibrated': uncalibrated_df}


############

df_base_dir = Path(
    '/home/toni/Workspace/MoSeS/exhaustive_search/cluster/blasi/exhaustive_search_models_size_2_to_6/timings')

timings_dict = read_all_dfs(df_base_dir)
write_progress(timings_dict)

pot_failed_drops = np.empty(shape=(1,), dtype=str)

# go through all dfs:
for model_size in reversed(range(2, 7)):
    write_progress(timings_dict, add_text=f"starting with models of size {model_size}")

    for model_id in tqdm.tqdm(timings_dict[f'df_{model_size}'].index, desc=f'model size = {model_size} '):
        # compute nr of parameters to delete for each model in df: remember uncalibrated.tsv
        nr_paras_to_delete = get_nr_paras_to_delete_AICC(model_id, timings_dict)
        # get submodels to remove
        submodels_to_remove = get_submodels_to_remove(model_id, nr_paras_to_delete)

        # remove submodels from respective dfs
        for submodel_id in submodels_to_remove:
            nr_paras = get_nr_parameters(submodel_id)
            # drop submodel from respective df. If it has already been removed: nothing happens
            try:
                timings_dict[f'df_{nr_paras}'].drop(submodel_id, inplace=True)
            except:
                pot_failed_drops = np.append(pot_failed_drops, submodel_id)

    write_progress(timings_dict)

# all models remaining in df_6, ... df_2 needed to be calibrated -> write all  together in one df
# note uncalibrated is not considered
df_remaining = pd.concat(list(timings_dict.values())[:-1])

# save
df_remaining.to_csv(df_base_dir / 'timings_necessary.tsv', sep='\t')

write_progress(timings_dict, add_text='done with exclusion. Determine missing calibrations')

##############
# check which failed pd.drops failed due to missing calibration data
# reread all dfs
timings_dict = read_all_dfs(df_base_dir)

missing_calibrations = np.empty(shape=(0,), dtype=str)

for model_id in set(pot_failed_drops):
    if model_id not in timings_dict[f'df_{get_nr_parameters(model_id)}'].index:
        # no calibration results given
        missing_calibrations = np.append(missing_calibrations, model_id)

        # all other elements in pot_failed_drops: calibration given, but already removed as a submodel of another model

missing_calibrations.tofile(df_base_dir / 'missing_calibrations.txt', sep=',')
