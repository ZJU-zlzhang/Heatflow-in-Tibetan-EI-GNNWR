import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

# Read the data from a CSV file
df = pd.read_csv('../Data/Combine_GF_05_Plate.csv')

# Define the observables for the x-axis with detailed descriptions
observables = ['Moho depth (km)', 'LAB depth (km)', 'Topography (m)', 'Susceptibility (SI)', 'Tectonic units (categorical)',
               'Gravity mean curvature (1/Gm)', 'Vertical magnetic field (nT)', 'Distance to ridges (km)',
               'Distance to trenches (km)', 'Distance to transform faults (km)', 'Distance to young rifts (km)',
               'Distance to volcanoes (km)']

# Short names for the dataframe columns
short_names = ['Moho', 'LAB', 'Topo', 'Sus', 'Tectonics', 'MeanCurv', 'Bz5',
               'Ridge', 'Trench', 'Transform', 'YoungRift', 'Volcanos']

# Set the number of columns for subplot grid
n_cols = 4
n_rows = int(np.ceil(len(observables) / n_cols))

# Create the larger figure and axes grid, with increased dpi for high resolution
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows), dpi=300)

# Flatten the axes array for easy iteration
axs = axs.flatten()

# Loop over each observable to create its subplot
for index, observable in enumerate(observables):
    ax = axs[index]
    # Scatter plot
    sns.scatterplot(x=df[short_names[index]], y=df['HF'], ax=ax, alpha=0.3, edgecolor=None)

    # Linear regression line with increased line width
    slope, intercept, r_value, p_value, std_err = linregress(df[short_names[index]], df['HF'])
    x_vals = np.linspace(df[short_names[index]].min(), df[short_names[index]].max(), 100)
    ax.plot(x_vals, intercept + slope * x_vals, 'k-', linewidth=2.5, label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')

    # LOWESS (Locally Weighted Scatterplot Smoothing)
    lowess_results = lowess(df['HF'], df[short_names[index]], frac=0.33)
    ax.plot(lowess_results[:, 0], lowess_results[:, 1], 'r--', label='LOWESS fit')

    ax.set_xlabel(observable, fontweight='bold', fontsize=14)

    # Only show the y-axis label on the leftmost plots
    if index % n_cols == 0:
        ax.set_ylabel('Surface Heat Flow(mW/m$^2$)', fontweight='bold', fontsize=14)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    # This is removed because we want the ticks on the left: ax.yaxis.tick_right()

    # Set y-axis ticks to be outside, to the right of the plot
    ax.tick_params(axis='y', direction='in')  # Changed to have the ticks pointing outwards to the right

    # Set the subplot border color to black
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)  # Make the border slightly thicker

    # Add subplot lettering without a bounding box
    ax.text(0.05, 0.95, f'({chr(ord("a") + index)})', transform=ax.transAxes,
            fontsize=20, verticalalignment='top', fontweight='bold')

    # Place the legend in the bottom left corner of each subplot
    ax.legend(loc='lower left', fontsize='small', frameon=True, framealpha=0.7)

# Hide any unused axes if there are any
for ax in axs[len(observables):]:
    ax.set_visible(False)


# Adjust the layout to prevent overlapping, with no space between subplots
plt.subplots_adjust(wspace=0)

# plt.savefig('../Result/features_relationship.png', dpi=300)

# plt.tight_layout()
plt.show()