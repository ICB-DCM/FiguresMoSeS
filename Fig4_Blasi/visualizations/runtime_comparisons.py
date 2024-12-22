import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

COLOR_ORANGE = '#eab90c'
COLOR_BLUE = '#07529a'
COLOR_GREY = '#909085'
COLOR_BLACK = '#000000'

calibrated_models_dict = {0: 1, 1: 32, 2: 31, 3: 30, 4: 29, 5: 28, 6: 34+709577, 7: 34, 8: 834, 9: 24, 32: 1}

for i in range(33):
    if i not in calibrated_models_dict:
        calibrated_models_dict[i] = 0

calibrated_models = [calibrated_models_dict[i] for i in range(33)]
theoretical_models = [math.comb(32, i) for i in range(0, 33)]
fraction_calibrated = [calibrated_models[i]/theoretical_models[i] for i in range(33)]

model_size = [i for i in range(0, 33)]
# plt.hist(count_calibrated_models, bins=range(0, 17), align='left')

# Create the background bar for theoretical_max
# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 4))
# ax.bar(model_size, calibrated_models, label='Calibrated Models', color=COLOR_ORANGE)
ax.bar(model_size, fraction_calibrated, label='Calibrated Models', color=COLOR_ORANGE)

# Create the stacked bar for alg_performance on top of theoretical_max
# ax.bar(model_size, theoretical_models, bottom=calibrated_models, label='Total number of models', color=COLOR_BLUE)

# Set labels and legend
ax.set_xlabel('\nNumber of parameters', fontsize=16)
ax.set_ylabel('Fraction of\ncalibrated models', fontsize=16)


# Set x-axis ticks to every third value
ticks = range(0, 33, 3)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks, fontsize=14, rotation=90)
plt.xlim((-.6, 32.6))


# set y-axis
plt.semilogy()
plt.yticks([1, 0.1, 0.01, 0.001, 0.0001, 0.00001], ["100 %", "10%", "1%", "0.1%", "0.01%", "0.001%"], fontsize=14)
plt.ylim((0.000005, 1))

# set box
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout to create more space around the plot
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.23)

# save figure
# plt.savefig('Figure_4_E.png', dpi=450, bbox_inches='tight')


# Time saved between approaches.
result_dict = {'forward': result_forward,
               'backward_reduced': result,
               'backward_full': result_backward_full,
               'exhaustive_from_forward': result_exhaustive_from_forward,
               'brute_force': result_brute_force}

for key in result_dict:
    total_time = 0
    for model_id in result_dict[key]:
        if result_dict[key][model_id]:
            total_time += df.loc[model_id]['total_time']

    print(f"{key}: {total_time}")
    # print(total_time)


# do the plots v1: All in one plot, for both...

# initialize figure

n_models = [121, 127, 72, 65536, 72]
times = [1089.124852180481, 3835.9875168800354, 2288.6218342781067, 902035.0687606335, 2288.6218342781067]

for i in [1, 2]:
    qoi = n_models if i == 1 else times

    fig, ax = plt.subplots()
    fig.set_size_inches([3.5, 5.8])  # default: width: 6.4, height:4.8

    # get x ticks
    x_ticks = np.arange(0.5, 0.3 * (5 + 1), 0.3)
    x_labels = ['forward', 'backward', 'eff. backward', 'brute force', 'eff. exhaustive']

    reduction_bwd_eff_bwd = '57.7% fewer models'
    reduction_eff_exh = '99.9% fewer models'

    # plot bar plots
    ax.bar(x_ticks[3:], qoi[3:], 0.2,
           color=[COLOR_BLUE, COLOR_BLUE, COLOR_ORANGE, COLOR_BLUE, COLOR_ORANGE][3:],
           label=x_labels[3:]
           )
    ax.set_yscale('log')
    if i == 1:
        ax.set_yticks([1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e5])
        ax.set_ylim(bottom=1, top=10e4 + 100)
    else:
        ax.set_yticks([1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e5, 10e6])
        ax.set_ylim(bottom=1e2, top=5e6)
    ax.minorticks_off()

    # ax.set_yticklabels(['', '$10$', '', '$10^3$', '', '$10^5$'])
    if 121 in qoi:
        ax.set_ylabel('number of fitted models')
    else:
        ax.set_ylabel('calibration time [s]')

    ax.set_xticks(x_ticks[3:])
    ax.set_xticklabels(x_labels[3:], rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
