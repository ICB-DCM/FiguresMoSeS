#

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Define colors
COLOR_BLUE = '#07529a'
COLOR_ORANGE = '#eab90c'
COLOR_GREY = '#909085'

# Define the arrays
data_x = np.array([
    [7, 26, 6, 60],
    [1, 29, 15, 52],
    [11, 56, 8, 20],
    [11, 31, 8, 47],
    [7, 52, 6, 33],
    [11, 55, 9, 22],
    [3, 71, 17, 6],
    [1, 31, 22, 44],
    [2, 54, 18, 22],
    [21, 47, 4, 26],
    [1, 40, 23, 34],
    [11, 66, 9, 12],
    [10, 68, 8, 12]
])

data_y = np.array([78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4])

# Create the DataFrame
df = pd.DataFrame(data_x,
                  columns=['calcium aluminate', 'tricalcium silicate', 'tetracalcium alumino ferrite', 'dicalcium silicate'])
df['heat of hardening'] = data_y

# Display the DataFrame
print(df)




# Create a pair plot
pair_plot = sns.pairplot(df, diag_kws={'color': COLOR_BLUE, "bins": 7}, plot_kws={'color': COLOR_BLUE})

# Define the limits for each axis
lims = [(0, 25), (0, 75), (0, 25), (0, 75), (70, 120)]

# Adjust the axis limits and colors
for i, ax in enumerate(pair_plot.axes.flatten()):

    print(ax.get_xlabel())
    print(i)
    print('-----------------------------------')
    row = i // len(df.columns)
    col = i % len(df.columns)

    if row == 4:
        x_lim = lims[col]
        y_lim = lims[row]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # Change color for scatter plots
        for line in ax.lines:
            line.set_color(COLOR_ORANGE)
        for collection in ax.collections:
            collection.set_facecolor(COLOR_ORANGE)
            collection.set_edgecolor(COLOR_ORANGE)
    elif row == col == 4:
        # Change color for histograms
        for patch in ax.patches:
            patch.set_facecolor(COLOR_ORANGE)  # Orange color
            patch.set_edgecolor(COLOR_ORANGE)  # Orange color
# Display the plot
plt.show()

# Regression Plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i in range(4):
    sns.regplot(x=df.columns[i], y='heat of hardening', data=df, ax=axes[int(i/2), i%2])

# correlation_plot:
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_cmap", [COLOR_BLUE, COLOR_GREY, COLOR_ORANGE])

# df = df.drop(columns='heat of hardening')
# Compute the correlation matrix
# correlation_matrix = df_x.corr()
correlation_matrix = df.corr()
mask = np.tril(4 * [1]) - np.eye(4)

# Create a heatmap with the custom colormap
plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Matrix', fontsize=18)
# Manually set the axis labels to be multiline
heatmap.set_xticklabels(
    labels=['calcium\naluminate',
            'tricalcium\nsilicate',
            'tetracalcium\nalumino\nferrite',
            'dicalcium\nsilicate'],
    rotation=45,
    fontsize=10,
)
heatmap.set_yticklabels(
    labels=['calcium\naluminate',
            'tricalcium\nsilicate',
            'tetracalcium\nalumino\nferrite',
            'dicalcium\nsilicate'],
    rotation=45,
    fontsize=10)


# Adjust layout to create more space around the plot
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.3)

plt.show()

# plt.savefig('Figure_2_B.png', format='png', dpi=600, bbox_inches='tight')



# here some colorbar stuff


# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(2, 8))  # Adjust figsize for better vertical layout
fig.subplots_adjust(right=0.5)

# Create a scalar mappable object with the colormap and the normalization
norm = plt.Normalize(vmin=min_val, vmax=max_val)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create the color bar
cbar = fig.colorbar(sm, orientation='vertical', ax=ax)
# cbar.set('AIC', fontsize=40)
cbar.ax.tick_params(labelsize=40)

# Display the plot
plt.show()

