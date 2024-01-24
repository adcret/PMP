import numpy as np
import os
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
import fabio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.stats import lognorm, rayleigh, chi, weibull_min
from scipy.optimize import curve_fit
import pandas as pd
import math


path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
com_phi = [file for file in files if 'com_phi' in file]
com_chi =[file for file in files if 'com_chi' in file]
fwhm_phi = [file for file in files if 'fwhm_phi' in file]
fwhm_chi = [file for file in files if 'fwhm_chi' in file]

def extract_number(filename):
    u_index = filename.find('u')
    substring = filename[:u_index] if u_index != -1 else filename
    return int(''.join(filter(str.isdigit, substring)))

com_phi = sorted(com_phi, key=extract_number)
com_chi = sorted(com_chi, key=extract_number)
fwhm_chi = sorted(fwhm_chi, key=extract_number)
fwhm_phi = sorted(fwhm_phi, key=extract_number)

# Information about pixel size
pixel_y = 0.203; pixel_x = 0.6575; # effective pixel sizes in mu

# Read in CHI COM data
chi_file = fabio.open(path + com_chi[5])
A = chi_file.data
row_size, col_size = A.shape

# Rotate and mirror in one axis (transpose and flip vertically)
B1 = A.T
B = np.flipud(B1)

# Identify NaN pixels (NaN becomes True, otherwise False)
TF = np.isnan(B)

# Normalize to average orientation for Chi
ave_chi = np.nanmean(A)  # Compute the average, ignoring NaNs
Chi_Img = B - ave_chi
max_chi = np.nanmax(Chi_Img)  # Find max, ignoring NaNs

# Read in PHI COM data
phi_file = fabio.open(path + com_phi[5])
A = phi_file.data

# Rotate and mirror for Phi data
B1 = A.T
B2 = np.flipud(B1)

# Identify NaN pixels for Phi data
TF = np.isnan(B2)

# Normalize to average orientation for Phi
ave_phi = np.nanmean(A)  # Compute the average, ignoring NaNs
Phi_Img = B2 - ave_phi
max_phi = np.nanmax(Phi_Img)  # Find max, ignoring NaNs

# Assuming TF is a binary numpy array
grain1 = ~TF
g = grain1

# Scale Chi_Img and Phi_Img and create the mosaicity image from this data (it is used to defined the cells)
MinChi, MaxChi = np.nanmin(Chi_Img), np.nanmax(Chi_Img)
Chi_scale = (Chi_Img - MinChi) / (MaxChi - MinChi)

MinPhi, MaxPhi = np.nanmin(Phi_Img), np.nanmax(Phi_Img)
Phi_scale = (Phi_Img - MinPhi) / (MaxPhi - MinPhi)

# Create RGB image Mosa_Img
Mosa_Img = np.stack((Chi_scale, Phi_scale, np.ones_like(Chi_scale)), axis=-1)

# Make a copy and handle NaN and value ranges
mosa = Mosa_Img.copy()
mosa[np.isnan(mosa)] = 0  # Set NaNs to 0
mosa[mosa < 0] = 0        # Clamp values below 0 to 0
mosa[mosa > 1] = 1        # Clamp values above 1 to 1

plt.figure()
plt.imshow(grain1, extent=[0, pixel_x * row_size, 0, pixel_y * col_size], cmap='viridis')   # Plot the grain map
plt.colorbar()
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')   
plt.title('Grain map')

# Create a disk-shaped structuring element with radius 3
se = disk(3, strict_radius=True)

# Perform erosion twice
grain1 = binary_erosion(grain1, se)
grain1 = binary_erosion(grain1, se)


# Perform dilation twice
grain_mask = binary_dilation(grain1, se)
grain_mask = binary_dilation(grain_mask, se)

# Set pixels outside the mask to slightly above max values
Chi_Img[~grain_mask] = max_chi * 1.8
Phi_Img[~grain_mask] = max_phi * 1.8

kernelSize = 2
KAM = np.zeros((col_size, row_size))

# Loop over all data points
for ii in range(col_size):
    for jj in range(row_size):
        if grain_mask[ii, jj] == 1:
            # Define the kernel boundaries
            iStart = max(ii - kernelSize, 0)
            iEnd = min(ii + kernelSize, col_size - 1)
            jStart = max(jj - kernelSize, 0)
            jEnd = min(jj + kernelSize, row_size - 1)

            # Calculate the kernel difference
            kernel_diff = np.abs(Chi_Img[iStart:iEnd+1, jStart:jEnd+1] - Chi_Img[ii, jj]) + \
                          np.abs(Phi_Img[iStart:iEnd+1, jStart:jEnd+1] - Phi_Img[ii, jj])
            nr_pixels_ROI = (iEnd - iStart + 1) * (jEnd - jStart + 1)

            # Store the average misorientation angle in the KAM map
            KAM[ii, jj] = np.sum(kernel_diff) / nr_pixels_ROI

# Create KAM filter and calculate area ratio
KAM_threshold = 0.041
KAM_filter = np.zeros_like(KAM, dtype=bool)

# Apply the threshold to create the filter
KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

# Calculate the area ratio
area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
print(f'KAM mask: percentage in walls {area_ratio * 100:.2f}% with KAM threshold: {KAM_threshold}')

# Morphological operations to get KAM_mask
se = disk(1)
KAM2 = binary_erosion(KAM_filter, se)
KAM_mask = binary_dilation(KAM2, se)

# Skeletonize the KAM mask
skel_Img = skeletonize(KAM_mask)

skel_dilated = binary_dilation(skel_Img, disk(1))

plt.figure()
plt.imshow(skel_dilated, extent=[0, pixel_x * row_size, 0, pixel_y * col_size], cmap='viridis')
plt.colorbar()
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')
plt.title('Skeletonized KAM mask')

# Read in FWHM data for Ch
chi_fwhm_file = fabio.open(path + fwhm_chi[5])
A = chi_fwhm_file.data
B1 = A.T
B = np.flipud(B1)
B[~grain_mask] = 2  # Apply grain mask and set outside values to 2
FWHM_Chi = np.minimum(np.abs(B), 2)

# Read in FWHM data for Phi
phi_fwhm_file = fabio.open(path + fwhm_phi[5])
A = phi_fwhm_file.data
B1 = A.T
B = np.flipud(B1)
B[~grain_mask] = 2  # Apply grain mask and set outside values to 2
FWHM_Phi = np.minimum(np.abs(B), 2)

# Calculate the FWHM image
FWHM_Img = FWHM_Chi + FWHM_Phi



plt.figure()
plt.imshow(FWHM_Img, extent=[0, pixel_x * row_size, 0, pixel_y * col_size], cmap='viridis')
plt.colorbar()
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')
plt.title('FWHM values map')

FWHM_Img[skel_Img] = np.nan  # Set the skeletonized pixels to NaN
plt.figure()
plt.imshow(FWHM_Img, extent=[0, pixel_x * row_size, 0, pixel_y * col_size], cmap='viridis', vmin=0, vmax=2.5)
plt.colorbar()
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')
plt.title('FWHM values map with skeleton overlay')

FWHM_Img[~g] = np.nan  # Set the pixels in the mask to NaN

# Plot FWHM_Img
plt.figure()
plt.imshow(FWHM_Img, extent=[0, pixel_x * row_size, 0, pixel_y * col_size], cmap='viridis', vmin=0, vmax=2.5)
plt.colorbar()
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')
plt.title('FWHM values map with grain1 mask')
FWHM_hist = np.concatenate(FWHM_Img)
FWHM_hist = FWHM_hist[~np.isnan(FWHM_hist)]

# Plot histogram of FWHM values
plt.figure()
plt.hist(FWHM_hist, bins=50, density=True, range=(0, 3.5))
plt.xlabel('FWHM values')
plt.ylabel('Frequency')
plt.title('Histogram of FWHM values inside the cells, with the cell walls')


def gaussian(x, height, center, width):
    return height * np.exp(-(x - center)**2 / (2*width**2))
def two_gaussians(x, h1, c1, w1, h2, c2, w2):
    return gaussian(x, h1, c1, w1) + gaussian(x, h2, c2, w2)


# Prepare the histogram data for fitting
hist, bin_edges = np.histogram(FWHM_hist, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial guesses for the fit parameters
initial_guess = [1, 0.15, 0.1, 1, 0.5, 0.1]

# Perform the curve fitting
bounds = ([0, -np.inf, 0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
popt, pcov = curve_fit(two_gaussians, bin_centers, hist, p0=initial_guess, bounds = bounds)

# Convert center and width parameters to radians for plotting and scale the height for density
popt[0] = popt[0]/math.pi*180  # scale the height
popt[1] = popt[1]*math.pi/180  # c1 in radians
popt[2] = popt[2]*math.pi/180  # w1 in radians
popt[3] = popt[3]/math.pi*180  # scale the height
popt[4] = popt[4]*math.pi/180 # c2 in radians
popt[5] = popt[5]*math.pi/180  # w2 in radians

# Generate a fine grid of x-values for a smooth plot
x_fine = np.linspace(0, 3.5*math.pi/180, 400)

# Plotting the fit result
plt.figure(figsize=(10, 6))
plt.hist(np.radians(FWHM_hist), bins=50, density=True, range=(0, 3.5*math.pi/180), label='Data')
plt.plot(x_fine, two_gaussians(x_fine, *popt), color='red', label='Fit')
plt.xlabel('FWHM')
plt.ylabel('PDF')
plt.title('Double Gaussian Fit to FWHM with cell walls')
plt.xlim(0, 2*math.pi/180)
# Adding the fitting parameters, equations, and analytical expression
eq1 = f"Gaussian 1: A={popt[0]:.1f}, mu={popt[1]:.6f}, sigma={popt[2]:.6f}"
eq2 = f"Gaussian 2: A={popt[3]:.2f}, mu={popt[4]:.6f}, sigma={popt[5]:.6f}"
analytical_eq = r"$f(x) = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$"
plt.text(0.4, 0.55, eq1, transform=plt.gca().transAxes, fontsize=10)
plt.text(0.4, 0.5, eq2, transform=plt.gca().transAxes, fontsize=10)
plt.text(0.4, 0.45, analytical_eq, transform=plt.gca().transAxes, fontsize=10)
plt.legend()




FWHM_Img[skel_dilated] = 12  # Set the skeletonized pixels to 4
FWHM_Img[FWHM_Img == 12] = np.nan  # Set the values of 4 to NaN
FWHM_hist = np.concatenate(FWHM_Img)
FWHM_hist = FWHM_hist[~np.isnan(FWHM_hist)]

# Prepare the histogram data for fitting
hist, bin_edges = np.histogram(FWHM_hist, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial guesses for the fit parameters
initial_guess1 = [1, 0.15, 0.1, 1, 0.5, 0.1]

# Perform the curve fitting
popt1, pcov1 = curve_fit(two_gaussians, bin_centers, hist, p0=initial_guess1, bounds = bounds)

# Convert center and width parameters to radians for plotting and scale the height for density
popt1[0] = popt1[0]/math.pi*180  # scale the height
popt1[1] = popt1[1]*math.pi/180  # c1 in radians
popt1[2] = popt1[2]*math.pi/180  # w1 in radians
popt1[3] = popt1[3]/math.pi*180  # scale the height
popt1[4] = popt1[4]*math.pi/180 # c2 in radians
popt1[5] = popt1[5]*math.pi/180  # w2 in radians

# Generate a fine grid of x-values for a smooth plot
x_fine = np.linspace(0, 3.5*math.pi/180, 400)

# Plotting the fit result
plt.figure(figsize=(10, 6))
plt.hist(np.radians(FWHM_hist), bins=50, density=True, range=(0, 3.5*math.pi/180), label='Data')
plt.plot(x_fine, two_gaussians(x_fine, *popt1), color='red', label='Fit')
plt.xlabel('FWHM')
plt.ylabel('PDF')
plt.title('Double Gaussian Fit to FWHM without cell walls')
plt.xlim(0, 2*math.pi/180)
# Adding the fitting parameters, equations, and analytical expression
eq1 = f"Gaussian 1: A={popt1[0]:.1f}, mu={popt1[1]:.6f}, sigma={popt1[2]:.6f}"
eq2 = f"Gaussian 2: A={popt1[3]:.2f}, mu={popt1[4]:.6f}, sigma={popt1[5]:.6f}"
analytical_eq = r"$f(x) = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$"
plt.text(0.4, 0.55, eq1, transform=plt.gca().transAxes, fontsize=10)
plt.text(0.4, 0.5, eq2, transform=plt.gca().transAxes, fontsize=10)
plt.text(0.4, 0.45, analytical_eq, transform=plt.gca().transAxes, fontsize=10)
plt.legend()



# Plot histogram of FWHM values
plt.figure()
plt.hist(FWHM_hist, bins=50, density=True, range=(0, 3.5))
plt.xlabel('FWHM values in radians')
plt.ylabel('PDF')
plt.title('Histogram of FWHM values inside the cells, without the cell walls')


# Average strain for each cell

### Find the cells and filter out the non-cells ### 
# Invert and dilate the skeleton image

BW_img = ~binary_dilation(skel_Img, disk(1))

# Label connected components
labeled_array, num_features = label(BW_img)
nr_cells = num_features  # Adjust for the exterior being labeled as a cell num_features - 1
print(f"Number of cells: {nr_cells}")

# Get region properties
props = regionprops(labeled_array)

min_cell_size = 10  # minimum size in pixel for a cell to be considered

mask = np.all(mosa == [1, 1, 1], axis=-1)

# Dilate the mask to also exclude the neighbors
erroded_mask = binary_erosion(mask, disk(3))
dilated_mask = binary_erosion(erroded_mask, disk(3))
dilated_mask = binary_dilation(dilated_mask, disk(3))
dilated_mask = binary_dilation(dilated_mask, disk(20))


# List to store the properties of regions that do not overlap with the mask
filtered = []
filtered_props = []

# Iterate over each region in props
for region in props:
    # Get the coordinates of the region
    region_coords = region.coords

    # Check if any of the coordinates overlap with the mask
    overlap = np.any(erroded_mask[region_coords[:, 0], region_coords[:, 1]])

    # If there is no overlap and the region meets the size criteria, add it to the list
    if not overlap and region.area >= min_cell_size:
        filtered_props.append(region)

# Iterate over each region in props
for region in filtered:
    # Get the coordinates of the region
    region_coords = region.coords

    # Check if any of the coordinates overlap with the mask
    overlap = np.any(dilated_mask[region_coords[:, 0], region_coords[:, 1]])
   # if not overlap:
       # filtered_props.append(region) # This can be changed ot remove the holes in the same as well as the edges of the grain map



nr_cells1 = len(filtered_props)

print(f"Number of cells: {nr_cells1}")

# Calculate areas and centroids
areas_all = [prop.area * pixel_x * pixel_y for prop in props][0:] 
areas_all1 = [prop.area * pixel_x * pixel_y for prop in filtered_props][0:] # Skip exterior [1:]
size_from_area = np.sqrt(areas_all)
size_from_area1 = np.sqrt(areas_all1)

# Calculate the average FWHM value for each cell
ave_FWHM = {prop.label: np.mean(FWHM_Img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in filtered_props}

fwhm_values = [ave_FWHM[prop.label] for prop in filtered_props]
fwhm_values = np.radians(fwhm_values)

# Plotting
plt.figure()
plt.scatter(size_from_area1, fwhm_values)
plt.xlabel('Cell Size in Microns (sqrt of area)')
plt.ylabel('Average FWHM (radians)')
plt.title('Average FWHM vs Cell Size')

plt.show()