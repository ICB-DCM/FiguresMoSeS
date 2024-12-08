import amici.plotting
import matplotlib.pyplot as plt
import pandas as pd
import petab
import pypesto.petab
import itertools

import numpy as np
import yaml
import ast

from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from apply_exhaustive_search import only_local, timing_df, local_method
from exhaustive_search.Example_2_Famos_synth.famos_synth_exhaustive_search import index2model_id, model_id2index, get_nr_paras_to_remove
from exhaustive_search.Example_2_Famos_synth.Figures.figure_settings import *

def get_comp_time(dir: Path, timing_df:pd.DataFrame) -> float:
    """loads checked models from dir file, gets computation time of those models from timing df.
    return: sum of computation times"""
    # load result
    result_df = pd.read_csv(dir, sep='\t')
    result_df.columns = ['key', 'value']
    result_df.set_index('key', inplace=True)
    # get model ids from result
    checked_models = [index2model_id(index) for index in ast.literal_eval(result_df.loc['checked models']['value'])]
    # get computation time for model fitting
    comp_time = timing_df.loc[checked_models, 'total_time'].sum()

    return comp_time


def computation_time_comparison(criterion: str, save: bool = False, ybottom: int=100):
    """ybottom: lower bound for y-axis"""


    root_dir = Path(__file__).parent

    # get timing df
    timing_df = pd.read_csv(root_dir / "../../modelspace_fitting/output_20220603_no_hdf5/timing/timings.tsv", sep='\t')
    timing_df.set_index('model_id', inplace=True)

    # initialize figure
    fig, ax = plt.subplots()
    fig.set_size_inches([3.5, 5.8])  # default: width: 6.4, height:4.8
    fig.set_dpi(DPI)
    # get x ticks
    x_ticks = np.arange(0.5, 0.3 * (3 + 1), 0.3)
    x_labels = ['brute force', 'forward', 'backward']

    ### get data
    # get bruteforce data:
    time_brute_force = timing_df['total_time'].sum()

    # get forward data
    time_first_forward = get_comp_time(root_dir / f'../../local2global_results/forward_{criterion}.tsv', timing_df)
    time_completion_forward = get_comp_time(root_dir / f'../../local2global_results/dilan_forward_{criterion}.tsv', timing_df)

    # get backward data
    time_first_backward = get_comp_time(root_dir / f'../../local2global_results/backward_{criterion}.tsv', timing_df)
    time_completion_backward = get_comp_time(root_dir / f'../../local2global_results/dilan_backward_{criterion}.tsv', timing_df)

    first_local_searches = [0, time_first_forward, time_first_backward]
    global_searches = [time_brute_force, time_completion_forward, time_completion_backward]

    # print values
    # print(f'{criterion}\n'
    #       f'|brute force | {time_brute_force}|\n'
    #       f'|forward (init + compl)| {time_first_forward} + {time_completion_forward} = {time_first_forward + time_completion_forward}|\n'
    #       f'| &nbsp; &nbsp; -> improvement | {100 - (time_first_forward + time_completion_forward)*100/time_brute_force}%| \n'
    #       f'| backward (init + compl) | {time_first_backward} + {time_completion_backward} = {time_first_backward + time_completion_backward}| \n'
    #       f'| &nbsp; &nbsp; -> improvement | {100 - (time_first_backward + time_completion_backward)*100/time_brute_force}%| \n'
    #       )

    # plot bar plots
    ax.bar(x_ticks, first_local_searches, 0.2,
           color=[COLOR_LIGHT_ORANGE, COLOR_LIGHT_BLUE, COLOR_LIGHT_BLUE],
           # label=[None, 'first forward selection', 'first backward selection']
           )

    ax.bar(x_ticks, global_searches, 0.2,
           color=[COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE],
           bottom=first_local_searches,
           # label=['brute force', 'forward selection to completion', 'backward selection to completion']
           )

    # axis settings
    ax.set_yscale('log')
    # ax.set_yticks([10e0, 10e1, 10e2, 10e3, 10e4, 10e5], fontsize=FONTSIZE_SMALL)
    ax.minorticks_off()
    # ax.set_yticklabels(['', '$10$', '', '$10^3$', '', '$10^5$'])
    ax.set_ylim(bottom=ybottom, top=10e5 + 100)
    # workaround for bug when setting custom ticks in log scale
    # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.set_ylabel('computation time (sec)', fontsize=FONTSIZE_LARGE)

    ax.set_xticks(x_ticks, fontsize=FONTSIZE_SMALL)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # title
    # ax.set_title('computation time\nbrute force method vs our method',
    #              size=FONTSIZE_LARGE, pad=32)

    # no frame above and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # create legend
    labels = ['brute force method', 'initial local search', 'local to global']
    colors = [COLOR_ORANGE, COLOR_LIGHT_BLUE, COLOR_BLUE]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]

    # legend below plot centered in axis
    # ax.legend(handles, labels, fontsize=FONTSIZE_SMALL, loc='center', ncol=1, bbox_to_anchor=(0.5,-0.32))
    # legend centerd in image
    fig.legend(handles, labels, fontsize=FONTSIZE_SMALL, loc='center', ncol=1, bbox_to_anchor=(0.43, -0.09))

    if save:
        plt.savefig(root_dir / f'{criterion}_ybottom={ybottom}.pdf',
                    bbox_inches='tight')
    plt.show()

### computation time comparison
computation_time_comparison('BIC', True, ybottom=10)
computation_time_comparison('AIC', True, ybottom=10)
plt.show()


# todo add decrease arrows: "...% faster"
