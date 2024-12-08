"""Get the fraction of calibrated models for the large scale example."""
import matplotlib.pyplot as plt


# colors
COLOR_ORANGE = '#eab90c'
COLOR_BLUE = '#07529a'

# data in dicts
# n_model_calibrations = {'FAMOS': 1077, 'efficient \n exhaustive': 710654, 'brute \n force': 2**32}
# time_model_calibrations = {'FAMOS': 13559, 'efficient \n exhaustive': 2326697, 'brute \n force$^*$': 14044543057}

n_model_calibrations = {'brute\nforce': 2**32, 'efficient\nexhaustive': 710654}
time_model_calibrations = {'brute\nforce$^*$': 14044543057, 'efficient\nexhaustive': 2326697}



# Plotting the bar graph with blue color
plt.figure(figsize=(4, 7))
plt.bar(n_model_calibrations.keys(), n_model_calibrations.values(), color=[COLOR_BLUE, COLOR_ORANGE])
# plt.bar(time_model_calibrations.keys(), time_model_calibrations.values(), color=[COLOR_BLUE, COLOR_ORANGE])

# Setting y-axis to logarithmic scale
plt.yscale('log')

# Set y-ticks at specified powers of 10
yticks = [10**5, 10**6, 10**7, 10**8, 10**9]
plt.yticks(yticks, [f'$10^{i}$' for i in range(5, 10)], fontsize=16, rotation=90)
# yticks = [10**i for i in [5, 7, 9, 11]]
# plt.yticks(yticks, [f'$10^{{{i}}}$' for i in [5, 7, 9, 11]])

plt.ylabel('calibrated models\n', fontsize=20)
# plt.ylabel('calibration time [s]\n', fontsize=20)

# Removing top and right spines to eliminate box around the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Remove minor ticks
plt.gca().yaxis.set_minor_locator(plt.NullLocator())

# set font sizes
plt.gca().yaxis.set_tick_params(labelsize=12)
plt.gca().xaxis.set_tick_params(labelsize=20, rotation=45)

plt.tight_layout()

plt.savefig('Figure_4_A.png', dpi=450)

# Show the plot
plt.show()

