import matplotlib.pyplot as plt

# Define colors
COLOR_BLUE = '#07529a'
COLOR_ORANGE = '#eab90c'
COLOR_GREY = '#909085'

simulation = {'observable_x_0ac': 0.7377663860156793, 'observable_x_4ac': 0.0031849256240718583, 'observable_x_k12': 0.06709948216417226, 'observable_x_k12k16': 0.008233273926995443, 'observable_x_k16': 0.03590837454905424, 'observable_x_k05': 0.03340788138445759, 'observable_x_k05k12': 0.04794347717860349, 'observable_x_k05k12k16': 0.0023462300667912465, 'observable_x_k05k08': 0.0056643730812503765, 'observable_x_k05k08k12': 0.0064141402306339135, 'observable_x_k05k08k16': 0.0013405018998448042, 'observable_x_k08': 0.03145923764432344, 'observable_x_k08k12': 0.00766479198215186, 'observable_x_k08k12k16': 0.003359246172299153, 'observable_x_k08k16': 0.004310757742773015}
measurement = {'observable_x_0ac': (0.7325317779722221, 0.02640458391346359), 'observable_x_4ac': (0.003414692855555556, 0.0013099116986302409), 'observable_x_k05': (0.034675634866666666, 0.00312871857994503), 'observable_x_k05k08': (0.0060505791, 0.0010135899486253274), 'observable_x_k05k08k12': (0.0064539220944444435, 0.0013320141350918132), 'observable_x_k05k08k16': (0.0013131901444444444, 0.00032625481184646104), 'observable_x_k05k12': (0.04803940603333332, 0.008546933579780392), 'observable_x_k05k12k16': (0.0026089641888888885, 0.0010285122604644403), 'observable_x_k08': (0.03182336765555556, 0.005398168048271999), 'observable_x_k08k12': (0.008036147591666666, 0.0027141749906952352), 'observable_x_k08k12k16': (0.0034945225000000005, 0.0009182565708633161), 'observable_x_k08k16': (0.004707897041666667, 0.002066970850424177), 'observable_x_k12': (0.07386365073888888, 0.00818320075381209), 'observable_x_k12k16': (0.008099179872222221, 0.0012991718032540186), 'observable_x_k16': (0.03378081865555556, 0.0030347916377625947)}

x_ids = sorted(simulation)
fig, ax = plt.subplots(figsize=(6, 6))
for i_x, x_id in enumerate(x_ids):
    ax.scatter(i_x, simulation[x_id], color=COLOR_BLUE)
    ax.errorbar(i_x, measurement[x_id][0], measurement[x_id][1], color=COLOR_GREY, capsize=5)
ax.set_xticks(range(len(x_ids)))
ax.set_xticklabels([x_id[11:] for x_id in x_ids], rotation=90, fontsize=16)
ax.title.set_text('Measurement vs. Simulation')
ax.title.set_fontsize(20)
ax.set_ylabel('fraction acetylated', fontsize=16)

# Remove the top and right border lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim((-0.05, 1))

# Adjust layout to create more space around the plot
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.5)


plt.savefig('Figure_4_D.png', format='png', dpi=600, bbox_inches='tight')
