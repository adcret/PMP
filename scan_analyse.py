import numpy as np
import matplotlib.pyplot as plt
from DFXM.cell_refinement import *
from scipy.stats import norm
from DFXM.cell_refinement import load_and_process_image, find_grain


# Data location and pixel size information
file_path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\'
pixel_x = 0.6575  # effective pixel size in mu
pixel_y = 0.203   # effective pixel size in mu

chi_file = '838um_2x_2_05_com_chi_fit15.edf'
phi_file = '838um_2x_2_05_com_phi_fit15.edf'

# Loads the files and prepares the data for further analysis.
# Returns the images, the maximum values of the images and the shape of the images.

Chi_Img, max_chi, row_chi, col_chi = load_and_process_image(file_path, chi_file)
Phi_Img, max_phi, row_phi, col_phi = load_and_process_image(file_path, phi_file)

print(Chi_Img.shape)
print(max_chi)

## define the values that should be plotted in chi and phi
min_value_chi = -0.2
min_value_phi = 0.01
max_value_chi = 0.05
max_value_phi = 0.15

filtered_chi = Chi_Img.copy()
filtered_phi = Phi_Img.copy()
filtered_chi[(Chi_Img <= min_value_chi) | (Chi_Img >= max_value_chi)] = 1.2
filtered_phi[(Phi_Img <= min_value_phi) | (Phi_Img >= max_value_phi)] = 1.2


# Creates a grain mask and applies it to the images.
# Returns the grain mask and the images with the grain mask applied.
grain_mask, Chi_Img, Phi_Img = find_grain(Chi_Img, Chi_Img, Phi_Img, max_chi, max_phi)

grain_mask_filtered, filtered_chi, filtered_phi = find_grain(filtered_chi, filtered_chi, filtered_phi, max_chi, max_phi)

while True:
    # The user can choose to either plot or not plot the histograms.
    # The default is not to plot the histograms.
    _ = input('Plot Phi and Chi distributions? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        while True:
            try:
                nbins = input('Number of bins: ')
                nbins = int(nbins)
                break
            except ValueError:
                print("Please enter a valid integer for the number of bins.")
                # Handle the error or repeat the input prompt
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
        break
    else:
        break


# These images are already scaled between 0-1 for HSV values
Mosa_HSV, Mosa_RGB = convert_to_HSV_RGB(Chi_Img, Phi_Img)

Mosa_HSV_filtered, Mosa_RGB_filtered = convert_to_HSV_RGB(filtered_chi, filtered_phi)

Mosa_HSV_filtered[(Mosa_HSV_filtered[..., 0] == 1) & (Mosa_HSV_filtered[..., 1] == 0) & (Mosa_HSV_filtered[..., 2] == 0)] = [1, 1, 1]
Mosa_RGB_filtered[(Mosa_RGB_filtered[..., 0] == 1) & (Mosa_RGB_filtered[..., 1] == 0) & (Mosa_RGB_filtered[..., 2] == 0)] = [1, 1, 1]
Mosa_HSV_filtered[(Mosa_HSV_filtered[..., 2] == 1)] = [1, 1, 1]
Mosa_RGB_filtered[(Mosa_RGB_filtered[..., 2] == 1)] = [1, 1, 1]

while True:
    # The user can choose to either plot or not HSV or RGB images.
    # The default is not to plot the maps.
    _ = input('Plot filtered mosaicity maps? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        while True:
            c = input('HSV or RBG? (H/R) ')
            if c == 'H' or c == 'h':
                plt.figure()
                plt.imshow(Mosa_HSV_filtered, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map HSV')
                plt.show()
                break
            elif c == 'R' or c == 'r':
                plt.figure()
                plt.imshow(Mosa_RGB_filtered, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map RGB')
                plt.show()
                break
            else:
                plt.figure()
                plt.imshow(Mosa_HSV_filtered, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map HSV')
                plt.show()                
                break
        break
    else:
        break


while True:
    # The user can choose to either plot or not HSV or RGB images.
    # The default is not to plot the maps.
    _ = input('Plot mosaicity maps? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        while True:
            c = input('HSV or RBG? (H/R) ')
            if c == 'H' or c == 'h':
                plt.figure()
                plt.imshow(Mosa_HSV, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map HSV')
                plt.show()
                break
            elif c == 'R' or c == 'r':
                plt.figure()
                plt.imshow(Mosa_RGB, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map RGB')
                plt.show()
                break
            else:
                plt.figure()
                plt.imshow(Mosa_HSV, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('Mosa map HSV')
                plt.show()                
                break
        break
    else:
        break

KAM = calculate_kam(Chi_Img, Phi_Img, grain_mask)

while True:
    # The user can choose to either plot or not the KAM map.
    # The default is not to plot the map.
    _ = input('Plot KAM map? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        plt.figure()
        plt.imshow(KAM, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
        plt.xlabel('x in micrometer')
        plt.ylabel('y in micrometer')
        plt.axis('equal')
        plt.title('KAM map')
        plt.show()
        break
    else:
        break

while True:
    # The user can choose to either plot or not the KAM histogram.
    # The default is not to plot the histogram.
    _ = input('Plot KAM histogram? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        plt.figure()
        masked_KAM_values = KAM[grain_mask]
        # Calculate the weights for each value in your array to normalize the sum of the bins to 1
        weights = np.ones_like(masked_KAM_values) / len(masked_KAM_values)
        # Include the weights in the histogram function
        plt.hist(masked_KAM_values, bins=300, weights=weights, label='KAM Distribution')
        y = np.arange(0, 1.5, 0.02)
        a, b = 0.236, 18.32  # Fitted values
        f = a * np.exp(-b * y)
        plt.plot(y, f, linewidth=1.5, label='Analytical Function')
        plt.legend()
        plt.title('KAM values histogram')
        break
    else:
        break


KAM_filter = create_binary_filter(KAM, grain_mask, threshold=0.043)

KAM_skel = create_skeleton(KAM_filter, radius=1)

while True:
    # The user can choose to either plot or not the KAM skeleton.
    # The default is not to plot the skeleton.
    _ = input('Plot KAM skeleton? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        plt.figure()
        plt.imshow(KAM_skel, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
        plt.xlabel('x in micrometer')
        plt.ylabel('y in micrometer')
        plt.axis('equal')
        plt.title('KAM skeleton')
        plt.show()
        break
    else:
        break

while True:
    # The user can choose to either plot or not the KAM skeleton overlayed on the RGB or HSV image.
    # The default is not to plot the overlay.
    _ = input('Plot KAM skeleton overlayed on RGB or HSV image? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        while True:
            c = input('RGB or HSV? (R/H) ')
            if c == 'R' or c == 'r':
                plt.figure()
                plt.imshow(overlay_skeleton([Mosa_RGB], KAM_skel)[0], extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('KAM skeleton overlayed on RGB image')
                plt.show()
                break
            elif c == 'H' or c == 'h':
                plt.figure()
                plt.imshow(overlay_skeleton([Mosa_HSV], KAM_skel)[0], extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('KAM skeleton overlayed on HSV image')
                plt.show()
                break
            else:
                plt.figure()
                plt.imshow(overlay_skeleton([Mosa_RGB], KAM_skel)[0], extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])
                plt.xlabel('x in micrometer')
                plt.ylabel('y in micrometer')
                plt.axis('equal')
                plt.title('KAM skeleton overlayed on RGB image')
                plt.show()
                break
        break
    else:
        break

BW_img = ~binary_dilation(KAM_skel, disk(1))
plt.figure()
plt.imshow(BW_img, extent=[0, pixel_x * row_chi, 0, pixel_y * row_phi])

filtered_props, num_features = calculate_cell_properties(BW_img, Mosa_HSV)
centroids = np.array([prop.centroid for prop in filtered_props])[0:]
## Still some issue but at least it's counting now...


print(f'Number of cells = {len(filtered_props)}')
while True:
# The user can choose to plot the centroids of the cells on the RBG image
    _ = input('Plot centroids on RGB image? (Y/N) ')
    if _ == 'Y' or _ == 'y':
        plt.figure()
        plt.imshow(KAM_skel, cmap='gray')
        plt.imshow(Mosa_RGB)
        plt.xlabel('x in pixel')
        plt.ylabel('y in pixel')
        plt.axis('equal')
        plt.title('Centroids on RGB image')
        plt.scatter(centroids[:, 1], centroids[:, 0], c='black', s=1)
        plt.show()
        break
    else:
        break

neigbors_dict = create_adjacency_list(filtered_props, num_features)