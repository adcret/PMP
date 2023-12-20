import numpy as np
import matplotlib.pyplot as plt
from DFXM.cell_refinement import ImageProcessor
from scipy.stats import norm


# Data location and pixel size information
file_path = 'C:\\Users\\adacre\\Documents\\DTU_Project\\data\\'
pixel_x = 0.6575  # effective pixel size in mu
pixel_y = 0.203   # effective pixel size in mu

# Initialize the ImageProcessor and ImagePlotter
processor = ImageProcessor(file_path, pixel_x, pixel_y)

# Load and process CHI and PHI COM data
Chi, max_chi = processor.load_and_process_image('838um_2x_2_05_com_chi_fit15.edf')
Phi, max_phi = processor.load_and_process_image('838um_2x_2_05_com_phi_fit15.edf')

grain_mask, Chi_Img, Phi_Img = processor.find_grain(Chi, Chi, Phi, max_chi, max_phi)

# Number of bins for the histogram
nbins = 20

# Create histograms and Gaussian fits for Chi_Img
plt.figure()
masked_chi_values = Chi_Img[grain_mask]
sigma_chi = np.std(masked_chi_values)
plt.hist(masked_chi_values, bins=nbins, range=(-0.52, 0.46), density=True, label='Chi', align='left')
y = np.linspace(-3.5, 4.0, 100)
f = norm.pdf(y, 0, sigma_chi)
plt.plot(y, f, linewidth=1.5, label=f'Fit to Chi, std = {sigma_chi:.2f} deg')
plt.title('Chi and Phi distribution histogram')
print(f'Sigma of Chi_Img distribution = {sigma_chi} deg')

# Create histograms and Gaussian fits for Phi_Img
masked_phi_values = Phi_Img[grain_mask]
sigma_phi = np.std(masked_phi_values)
plt.hist(masked_phi_values, bins=nbins, range=(-0.44, 0.44), density=True, label='Phi', align='left')
f = norm.pdf(y, 0, sigma_phi)
plt.plot(y, f, linewidth=1.5, label=f'Fit to Phi, std = {sigma_phi:.2f} deg')
print(f'Sigma of Phi_Img distribution = {sigma_phi} deg')

plt.legend()
plt.xlabel('Degrees')

plt.show()