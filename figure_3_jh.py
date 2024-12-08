import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define colors
COLOR_BLUE = '#07529a'
COLOR_ORANGE = '#eab90c'
COLOR_GREY = '#909085'


prediction_model_opt = \
    np.array([78.43831359,  72.86733683, 106.19096693,  89.40163706,
              95.64375322, 105.30177673, 104.12867267,  75.59187807,
              91.8182253 , 115.54611741,  81.70226809, 112.24438627,
              111.62466783])

prediction_model_0111 = \
    np.array([77.52262406,  74.17699521, 109.20599988,  90.25118597,
        95.55402201, 105.56735484, 104.12164904,  74.65072463,
        93.45903045, 113.96635859,  80.46245907, 110.98022851,
       110.58136774])

prediction_model_1011 = \
    np.array([98.00458482, 102.61212673, 154.45196889, 110.80011893,
       142.70828894, 151.70447913, 175.53853759, 106.98790109,
       146.96223522, 142.76867779, 122.47250669, 169.70283635,
       172.21658798])

# read in y
data_y = np.array(
    [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4]
)


# scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(data_y, prediction_model_opt, color=COLOR_BLUE, s=100)
plt.scatter(data_y, prediction_model_1011, color=COLOR_GREY, s=100)

# axes
plt.xlabel('Data', fontsize=20)
plt.ylabel('Model Prediction', fontsize=20)
plt.xlim((70, 120))
plt.ylim((70, 120))

# spines
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# set font sizes
plt.gca().yaxis.set_tick_params(labelsize=16)
plt.gca().xaxis.set_tick_params(labelsize=16)

plt.plot([70, 120], [70, 120], '--', color=COLOR_ORANGE, linewidth=3)

plt.tight_layout()
plt.show()

# plt.savefig('Figure_3_A_jh.png', dpi=450)
