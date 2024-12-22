import matplotlib.pyplot as plt
import numpy as np

# Data
# n_models = [121, 127, 72, 65536, 72]
# times = [1089.124852180481, 3835.9875168800354, 2288.6218342781067, 902035.0687606335, 2288.6218342781067]
n_models_hald = [10, 8, 6, 15, 6]
AICs = [241.48, 231.58, 231.58, 231.58, 231.58]


# Colors
COLOR_ORANGE = '#eab90c'
COLOR_BLUE = '#07529a'
COLOR_GREY = '#909085'
COLOR_BLACK = '#000000'

# Labels
labels_D = ['forward', 'backward', 'eff. backward']
labels_E = ['brute force', 'eff. exhaustive']
labels_AIC = ['forward', 'backward', 'eff. backward', 'brute force', 'eff. exhaustive']

# Create plots

# Plot histogram hald
n_models_hald = [10, 8, 6, 15, 6]

fig, ax = plt.subplots(figsize=(8, 9))
ax.bar(labels_AIC, n_models_hald, color=COLOR_ORANGE)
# ax.set_yscale('log')
ax.set_ylabel('number of calibrated models', fontsize=20)
ax.set_yticks([0, 5, 10, 15], fontsize=20)
ax.set_yticklabels(['0', '5', '10', '15'], fontsize=16)
ax.set_xticklabels(labels_AIC, fontsize=20, rotation=45, ha='right')
ax.set_ylim((0, 16))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2)
# plt.title('D')
plt.savefig('Figure_3_hist_jh.png', dpi=450, bbox_inches='tight')
plt.show()

# Plot B
fig, ax = plt.subplots(figsize=(8, 9))
ax.bar(labels_AIC, AICs, color=COLOR_ORANGE)
# ax.set_yscale('log')
ax.set_ylabel('AIC', fontsize=20)
ax.set_yticks([230, 235, 240], fontsize=20)
ax.set_yticklabels(['230', '235', '240'], fontsize=16)
ax.set_xticklabels(labels_AIC, fontsize=20, rotation=45, ha='right')
ax.set_ylim((228, 242))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2)
# plt.title('D')
plt.savefig('Figure_3_B.png', dpi=450, bbox_inches='tight')
plt.show()

# Plot C
fig, ax = plt.subplots(figsize=(6, 9))
ax.bar(labels_D, times[:3], color=[COLOR_ORANGE, COLOR_ORANGE, COLOR_ORANGE])
ax.set_yscale('log')
ax.set_ylabel('calibration time [s]', fontsize=20)
ax.set_yticks([10, 100, 1000, 10000], labels=["$10^1$", "$10^2$", "$10^3$", "$10^4$"], fontsize=16)
ax.set_xticklabels(labels_D, fontsize=20, rotation=45, ha='right')
ax.set_ylim((0, 10000))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.18)
# plt.title('D')
plt.savefig('Figure_3_C.png', dpi=450, bbox_inches='tight')
plt.show()

# Plot E
fig, ax = plt.subplots(figsize=(6, 9))
ax.bar(labels_D, n_models[:3], color=[COLOR_ORANGE, COLOR_ORANGE, COLOR_ORANGE])
ax.set_yscale('log')
ax.set_ylabel('number of calibrated models', fontsize=20)
ax.set_yticks([1, 10, 100], labels=["$10^0$", "$10^1$", "$10^2$"], fontsize=16)
ax.set_ylim((0, 150))
ax.set_xticklabels(labels_D, fontsize=20, rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.18)

plt.savefig('Figure_3_E.png', dpi=450, bbox_inches='tight')
plt.show()



# Plot D
fig, ax = plt.subplots(figsize=(4.5, 9))
ax.bar(labels_E, times[3:], color=[COLOR_ORANGE, COLOR_ORANGE])
ax.set_yscale('log')
ax.set_ylabel('calibration time [s]', fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yticks([100, 1000, 10000, 100000, 1000000], labels=["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"], fontsize=16)
ax.set_xticklabels(labels_E, fontsize=20, rotation=45, ha='right')
ax.set_ylim((0, 1000000))
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.23)

# plt.title('D')
plt.savefig('Figure_3_D.png', dpi=450, bbox_inches='tight')
plt.show()




# Plot F
fig, ax = plt.subplots(figsize=(4.5, 9))
ax.bar(labels_E, n_models[3:], color=[COLOR_ORANGE, COLOR_ORANGE])
ax.set_yscale('log')
ax.set_ylabel('number of calibrated models', fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yticks([1, 10, 100, 1000, 10000, 100000], labels=["$10^0$", "$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$"], fontsize=16)
ax.set_xticklabels(labels_E, fontsize=20, rotation=45, ha='right')
ax.set_ylim((0, 100000))
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.23)


plt.savefig('Figure_3_F.png', dpi=450, bbox_inches='tight')
plt.show()
