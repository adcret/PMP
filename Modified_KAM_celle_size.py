
# [The beginning part of the script remains unchanged]

# Import additional required modules
from scipy.stats import weibull_min, gamma

# [All the code up to the distribution fitting and plotting remains unchanged]

# Assuming 'size_from_area' and 'size_from_area1' are defined in the script

# Define a function to fit distributions and plot
def fit_and_plot_distribution(dist, data, ax, label):
    params = dist.fit(data)
    x = np.linspace(min(data), max(data), 100)
    pdf = dist.pdf(x, *params)
    ax.hist(data, bins=50, range=(0, 25), density=True, alpha=0.5)
    ax.plot(x, pdf, 'r-')
    mean_size = np.mean(data)
    ax.annotate(f'Mean: {mean_size:.2f} mu', xy=(0.5, 0.7), xycoords='axes fraction', ha='center', va='center')
    ax.set_xlim(0, 25)
    ax.set_title(label)

# Create a figure and axes for 4 distributions
fig, axs = plt.subplots(4, 2, figsize=(10, 20))  # Adjust the figsize as needed
axs = axs.flatten()

# Fit and plot for each distribution and dataset
fit_and_plot_distribution(chi, size_from_area, axs[0], 'Chi Distribution (Dataset 1)')
fit_and_plot_distribution(chi, size_from_area1, axs[1], 'Chi Distribution (Dataset 2)')
fit_and_plot_distribution(lognorm, size_from_area, axs[2], 'Lognormal Distribution (Dataset 1)')
fit_and_plot_distribution(lognorm, size_from_area1, axs[3], 'Lognormal Distribution (Dataset 2)')
fit_and_plot_distribution(weibull_min, size_from_area, axs[4], 'Weibull Distribution (Dataset 1)')
fit_and_plot_distribution(weibull_min, size_from_area1, axs[5], 'Weibull Distribution (Dataset 2)')
fit_and_plot_distribution(gamma, size_from_area, axs[6], 'Gamma Distribution (Dataset 1)')
fit_and_plot_distribution(gamma, size_from_area1, axs[7], 'Gamma Distribution (Dataset 2)')

# Show the plot
plt.tight_layout()
plt.show()

# [Rest of the script remains unchanged]
