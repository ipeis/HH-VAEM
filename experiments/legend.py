import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots()

custom_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                Line2D([0], [0], color='tab:orange', lw=4)]

ax.legend(custom_lines,[r'WITHOUT reparameterization', r'WITH reparameterization'])
plt.axis('off')
plt.savefig('legend.pdf')