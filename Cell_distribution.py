import numpy as np
import os
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
import fabio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.stats import lognorm, norm, kstest
import pandas as pd


path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
com_phi = [file for file in files if 'com_phi' in file]
com_chi =[file for file in files if 'com_chi' in file]

def extract_number(filename):
    u_index = filename.find('u')
    substring = filename[:u_index] if u_index != -1 else filename
    return int(''.join(filter(str.isdigit, substring)))

com_phi = sorted(com_phi, key=extract_number)
com_chi = sorted(com_chi, key=extract_number)

names = [extract_number(file) for file in com_phi]
print(names)    

pixel_x = 0.6575
pixel_y = 0.203

print(com_phi)
print(com_chi)

#Define all the necessary lists
av_cell, av_cell1 = [], []

sizes, sizes1 = [], []
X,  X1 = [], []
PDF, PDF1 =  [], []

excel_data = {}

mean_sizes = []
filtered_cell_counts = []

area_sizes, area_sizes1 = [], []
grain_size = []
cell_density = []
v_fraction = []
cell_number = []

major_axes = []
minor_axes = []
orientations = []


# Loop over all data sets
for _ in range(len(com_chi)):
    chi_file = fabio.open(path + com_chi[_])
    A = chi_file.data
    row_size, col_size = A.shape
    B1 = A.T
    B = np.flipud(B1)
    TF = np.isnan(B)
    ave_chi = np.nanmean(A)
    Chi_Img = B - ave_chi
    # Read in PHI COM data
    phi_file = fabio.open(path + com_phi[_])
    A = phi_file.data
    print(com_phi[_])
    print(com_chi[_])

    # Rotate and mirror for Phi data
    B1 = A.T
    B2 = np.flipud(B1)


    # Normalize to average orientation for Phi
    ave_phi = np.nanmean(A)  # Compute the average, ignoring NaNs
    Phi_Img = B2 - ave_phi
    max_phi = np.nanmax(Phi_Img)  # Find max, ignoring NaNs
    max_chi = np.nanmax(Chi_Img)  # Find max, ignoring NaNs

    # Assuming TF is a binary numpy array
    grain1 = ~TF

    # Create a disk-shaped structuring element with radius 3
    se = disk(3, strict_radius=True)

    # Perform erosion twice
    grain1 = binary_erosion(grain1, se)
    grain1 = binary_erosion(grain1, se)


    # Perform dilation twice
    grain_mask = binary_dilation(grain1, se)
    grain_mask = binary_dilation(grain_mask, se)

    # Calculate the area of the grain mask
    grain_mask_area = np.sum(grain_mask) * pixel_x * pixel_y 
    
    grain_size.append(grain_mask_area)
    print(f"Grain mask area: {grain_mask_area:.2f} um^2")

    # Set pixels outside the mask to slightly above max values
    Chi_Img[~grain_mask] = max_chi * 1.8
    Phi_Img[~grain_mask] = max_phi * 1.8


    # Scale Chi_Img and Phi_Img
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
    KAM_list = np.arange(0, 0.085, 0.001).tolist()
    
    sizes = []
    sizes1 = []
    for value in KAM_list:
        KAM_threshold = value
        KAM_filter = np.zeros_like(KAM, dtype=bool)

        # Apply the threshold to create the filter
        KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

        # Calculate the area ratio
        area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
        print(f'KAM mask: percentage in walls {area_ratio * 100:.2f}% with KAM threshold: {KAM_threshold}')
        
        if 0.69 < area_ratio < 0.72 or area_ratio < 0.65:
            if area_ratio < 0.65:
                # Update KAM_threshold and recompute KAM_filter
                KAM_threshold = 0.015
                KAM_filter = np.zeros_like(KAM, dtype=bool)
                KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

            se = disk(1)
            KAM2 = binary_erosion(KAM_filter, se)
            KAM_mask = binary_dilation(KAM2, se)

            # Skeletonize the KAM mask
            skel_Img = skeletonize(KAM_mask)



            Mosa_Img_overlay = np.copy(Mosa_Img)
            Mosa_Img_overlay[skel_Img] = [2.5, 2.5, 2.5]  # Assuming Mosa_Img is 3-channel

            mosa1 = colors.hsv_to_rgb(mosa)  # Convert HSV to RGB
            # Overlay skeleton on Mosa overlay and plot
            Mosa_overlay = np.copy(mosa1)  # Assuming mosa is in RGB format
            Mosa_overlay[skel_Img] = [0, 0, 0]


            # Invert and dilate the skeleton image
            BW_img = ~binary_dilation(skel_Img, disk(1))
            WB_img = ~BW_img


            # Calculate the number of pixels in the skeleton and remove from the grain mask area for area fraction calculation
            mask_pixels = np.sum(WB_img) * pixel_x * pixel_y
            grain_mask_area = grain_mask_area - mask_pixels


            # Label connected components
            labeled_array, num_features = label(BW_img)
            nr_cells = num_features  # Adjust for the exterior being labeled as a cell num_features - 1
            print(f"Number of cells: {nr_cells}")

            # Get region properties
            props = regionprops(labeled_array)

            min_cell_size = 5  # minimum size in pixel for a cell to be considered

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
            #for region in filtered:
             #   # Get the coordinates of the region
              #  region_coords = region.coords
#
 #               # Check if any of the coordinates overlap with the mask
  #              overlap = np.any(dilated_mask[region_coords[:, 0], region_coords[:, 1]])
   #             if not overlap:
    #                filtered_props.append(region)


             # Remove the cells that are too large this is done because in some dataset the scans are incomplete, leading to 
            filtered_props = [prop for prop in filtered_props if prop.area * pixel_x * pixel_y <= 600]
            nr_cells1 = len(filtered_props)
            filtered_cell_counts.append(nr_cells1)

            print(f"Number of cells after filtering: {nr_cells1}")

            nr_cells_per_unit_area = nr_cells / grain_mask_area
            print(f"Number of cells per unit area: {nr_cells_per_unit_area}")

            nr_cells1_per_unit_area = nr_cells1 / grain_mask_area
            print(f"Number of cells after filtering per unit area: {nr_cells1_per_unit_area}")

            cell_density.append(nr_cells1_per_unit_area)

            # Calculate areas and centroids
            areas_all = [prop.area * pixel_x * pixel_y for prop in props][1:] 
            areas_all1 = [prop.area * pixel_x * pixel_y for prop in filtered_props][0:] # Skip exterior [1:]

            #look at volume fraction of cells
            v_fraction.append(np.sum(areas_all1) / grain_mask_area)



            size_from_area = np.sqrt(areas_all)
            size_from_area1 = np.sqrt(areas_all1)
            sizes.append(size_from_area)
            sizes1.append(size_from_area1)
            

            # Get the major and minor axis lengths of the filtered cells to check for anisotropy
            for prop in filtered_props:
                major_axes.append(prop.major_axis_length)
                minor_axes.append(prop.minor_axis_length)

            #get the orientation of the major axis
            for prop in filtered_props:
                orientations.append(prop.orientation)
                
            
            # Calculate the ratio of minor to major axis lengths
            ratios = [minor/major if major != 0 else 0 for minor, major in zip(minor_axes, major_axes)]
            # Plot histogram
            plt.figure()
            plt.hist(ratios, bins=np.linspace(0, 1, 50))
            plt.xlabel('Minor/Major Axis Ratio')
            plt.ylabel('Frequency')
            plt.title('Minor to Major Axis Ratios')
            
            # Plot histogram of orientations
            plt.figure()
            plt.hist(orientations, bins=np.linspace(-np.pi/2, np.pi/2, 50))
            plt.xlabel('Orientation (rad)')
            plt.ylabel('Frequency')
            plt.title('Orientations of Major Axis')

            plt.figure()
            plt.scatter(ratios, orientations)
            plt.xlabel('Minor/Major Axis Ratio')
            plt.ylabel('Orientation (rad)')
            plt.title('Minor to Major Axis Ratios vs Orientations')
            

            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)
            print(f"Mean ratio: {mean_ratio:.2f}, Median ratio: {median_ratio:.2f}")

            break
    column_name = names[_]  # Using the name from the names list
    if column_name not in excel_data:
        excel_data[column_name] = []
    excel_data[column_name].extend(size_from_area1)
    print('A line was added to the excel file.')

    area_sizes.append(sizes)
    area_sizes1.append(sizes1) 

#df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in excel_data.items()]))
#df.to_excel('cell_sizes_kernel2.xlsx', index=False)
#print("Excel file 'cell_sizes_kernel2.xlsx' has been saved.")
    
#calculate the cell density of normalised number of cells
for i in range(len(cell_density)):
    cell_density[i] = cell_density[i] * grain_size[0]


# Calculate the ratio of minor to major axis lengths
ratios = [minor/major if major != 0 else 0 for minor, major in zip(minor_axes, major_axes)]



area_sizes = list(area_sizes)
area_sizes1 = list(area_sizes1)

# Get the strain steps for plot titles
strain = [0, 0.005, 0.008, 0.013, 0.024, 0.035, 0.046, 0.046, 0.046, 0.046]

# Plotting grain sizes 
plt.figure()
plt.plot(strain, grain_size, 'o-')
plt.xlabel('$\epsilon$', fontsize=20)
plt.ylabel('$\mu m^2$')
plt.title('Grain Size')

# Plotting number of cells
plt.figure(figsize=(9, 6))
plt.plot(strain, cell_density, 'o-', label='Number of Cells normalised per area')
plt.xlabel('$\epsilon$', fontsize=20)
plt.ylabel('Number of cells')
plt.title('Normalised Number of Cells')

# Plotting volume fraction of cells
plt.figure(figsize=(9, 6))
plt.plot(strain, v_fraction, 'o-', label='Volume Fraction of Cells')
plt.xlabel('$\epsilon$', fontsize=20)
plt.ylabel('Volume Fraction')
plt.title('Volume Fraction of Cells')


# Define a function to fit distributions and plot
def fit_and_plot_lognorm(data, ax, label):
    # Remove NaNs and infinite values from data
    data = np.array(data)
    data = data[np.isfinite(data)]
    data = data[data < 25]  # remove outliers

    # Handle empty data or data with insufficient values
    if len(data) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(label)
        return

    try:
        # Fit the log-normal distribution to the data
        params = lognorm.fit(data)
        shape, loc, scale = params
        mu = np.log(scale)
        ratio_mu_sigma = mu / shape
        x = np.linspace(min(data), max(data), 100)
        pdf = lognorm.pdf(x, *params)
        ax.hist(data, bins=50, range=(0, 18), density=True, alpha=0.9)
        ax.plot(x, pdf, 'r-', lw = 4)
        mean_size = np.mean(data)
        D, p_value = kstest(data, lambda x: lognorm.cdf(x, *params))
        ax.annotate(f'p-value={p_value:.2e}', xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.annotate(f'Mean: {mean_size:.2f} mu', xy=(0.5, 0.7), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.annotate(f'Mu/Sigma Ratio: {ratio_mu_sigma:.2f}', xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.set_xlabel('Cell Size', fontsize=20)
        ax.set_ylabel('PDF', fontsize=20)
        ax.set_xlim(0, 18)
        #ax.set_title(label, fontsize=20)
    except Exception as e:
        print(f"Error fitting {label}: {e}")
        ax.text(0.5, 0.5, 'Error in fitting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)



# Loop over each dataset in area_sizes1
for i, dataset in enumerate(area_sizes1):
    # Create a new figure for each dataset
    fig, ax = plt.subplots(figsize=(9, 6))  # Single plot for log-normal distribution

    # Fit and plot log-normal distribution for the current dataset
    fit_and_plot_lognorm(dataset, ax, 'Lognormal Distribution')

    # Set the title for the figure as the name of the sample
    strain_step = strain[i]
    fig.suptitle(f'${{\\epsilon}}$ = {strain_step:.3f}', fontsize=20)


def fit_and_plot_normal_log(data, ax, label):
    # Remove NaNs and infinite values from data
    data = np.array(data)
    data = data[np.isfinite(data)]
    data = data[data > 0]  # Ensure data is positive as log of non-positive values is undefined

    # Handle empty data or data with insufficient values
    if len(data) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(label)
        return

    try:
        # Take the log of the data
        log_data = np.log(data)

        # Fit the normal distribution to the log of the data
        mu, std = norm.fit(log_data)
        x = np.linspace(min(log_data), max(log_data), 100)
        pdf = norm.pdf(x, mu, std)
        ax.hist(log_data, bins=25, density=True, alpha=0.9)
        ax.plot(x, pdf, 'r-')
        mean_size = np.mean(log_data)
        D, p_value = kstest(log_data, lambda x: norm.cdf(x, mu, std))
        ax.annotate(f'KS test: D={D:.2f}, p={p_value:.2f}', xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center')
        ax.annotate(f'Mean: {mean_size:.2f}', xy=(0.5, 0.7), xycoords='axes fraction', ha='center', va='center')
        ax.set_title(label)
    except Exception as e:
        print(f"Error fitting {label}: {e}")
        ax.text(0.5, 0.5, 'Error in fitting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)



# Loop over each dataset in area_sizes1
for i, dataset in enumerate(area_sizes1):
    # Create a new figure for each dataset
    fig, ax = plt.subplots(figsize=(8, 6))  # Single plot for normal distribution of log data

    # Fit and plot normal distribution to the log of the current dataset
    fit_and_plot_normal_log(dataset, ax, 'Normal Distribution of Log Data')

    # Set the title for the figure as the name of the sample
    #fig.suptitle('$epsilon$' + strain[i], fontsize=18)

    # Adjust layout and show the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  





for i, dataset in enumerate(area_sizes1):
    # Process the dataset
    # [Your existing dataset processing code]
    
    # Calculate and store the mean size
    mean_size = np.mean(dataset)
    mean_sizes.append(mean_size)
    

# Plotting mean sizes (ignoring first two datasets)
plt.figure(figsize=(9, 6))
plt.plot(strain[2:], mean_sizes[2:], 'o-', label='Mean Sizes')
plt.xlabel('$\epsilon$', fontsize=20)
plt.ylabel('$\mu$m')
plt.title('Mean Size')
plt.legend()


# Plotting number of filtered cells (ignoring first two datasets)
plt.figure(figsize=(10, 5))
plt.plot(names[2:], filtered_cell_counts[2:], 'o-', label='Number of Filtered Cells')
plt.xlabel('whatever')
plt.ylabel('Filtered Cells Count')
plt.title('Number of Filtered Cells per Dataset')
plt.legend()

plt.show()