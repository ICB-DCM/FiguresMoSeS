import pypesto
from pypesto.petab import PetabImporter
import amici
import numpy as np
import matplotlib.pyplot as plt

import petab

COLOR_ORANGE = '#eab90c'
COLOR_BLUE = '#07529a'
COLOR_GREY = '#909085'
COLOR_BLACK = '#000000'

petab_dir = '../simulationes/m_true/xample_modelSelection.yaml'

importer = PetabImporter.from_yaml(petab_dir)
model = importer.create_model()
solver = model.getSolver()

time_points = np.arange(0, 65, 0.1)
model.setTimepoints(time_points)
rdata = amici.runAmiciSimulation(model, solver)

data = [0, 0.1942176, 0.0484032, 0.61288016, 4.07930835, 10.12008893]
data_times = [0, 1, 5, 10, 30, 60]

plt.plot(time_points, rdata['x'], linewidth=3)

# Set labels for the axes
plt.xlabel('Time [s]', fontsize=28)
plt.ylabel('Concentration [M]', fontsize=28)

# Set the x and y-axis limits
plt.xlim(0, 65)
plt.ylim(0, 12)

# Remove the right and upper spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Set the color for the lines
plt.gca().lines[0].set_color(COLOR_ORANGE)  # COLOR_ORANGE
plt.gca().lines[1].set_color(COLOR_BLUE)  # COLOR_BLUE

plt.legend(['$x_A$', '$x_B$'], loc='upper left', fontsize=28)
plt.plot(data_times, data, color=COLOR_BLUE, marker='o', linestyle='None',
         fillstyle='none', markersize=10, markeredgewidth=3)

plt.gca().yaxis.set_tick_params(labelsize=20)
plt.gca().xaxis.set_tick_params(labelsize=20)
plt.tight_layout()

