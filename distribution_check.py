import numpy as np
import os
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
import fabio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.stats import lognorm, rayleigh, chi, weibull_min, gamma, kstest
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

names = [file[:3] for file in com_phi]
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

area_sizes, area_sizes1 = [], []

def process_image(KAM_threshold, KAM_filter, Mosa_Img, mosa, pixel_x, pixel_y):
    # Perform morphological operations
    se = disk(1)
    KAM2 = binary_erosion(KAM_filter, se)
    KAM_mask = binary_dilation(KAM2, se)

    # Skeletonize the KAM mask
    skel_Img = skeletonize(KAM_mask)

    # Overlay on Mosa_Img
    Mosa_Img_overlay = np.copy(Mosa_Img)
    Mosa_Img_overlay[skel_Img] = [2.5, 2.5, 2.5]

    # Convert HSV to RGB and overlay skeleton
    mosa1 = colors.hsv_to_rgb(mosa)
    Mosa_overlay = np.copy(mosa1)
    Mosa_overlay[skel_Img] = [0, 0, 0]

    # Invert and dilate the skeleton image
    BW_img = ~binary_dilation(skel_Img, disk(1))

    # Label connected components
    labeled_array, num_features = label(BW_img)
    nr_cells = num_features
    print(f"Number of cells: {nr_cells}")

    # Get region properties
    props = regionprops(labeled_array)
    min_cell_size = 10

    # Create masks
    mask = np.all(mosa == [1, 1, 1], axis=-1)
    erroded_mask = binary_erosion(mask, disk(3))
    dilated_mask = binary_erosion(erroded_mask, disk(3))
    dilated_mask = binary_dilation(dilated_mask, disk(3))
    dilated_mask = binary_dilation(dilated_mask, disk(20))

    # Filter regions
    filtered_props = []
    for region in props:
        region_coords = region.coords
        overlap = np.any(erroded_mask[region_coords[:, 0], region_coords[:, 1]])
        if not overlap and region.area >= min_cell_size:
            filtered_props.append(region)

    nr_cells1 = len(filtered_props)
    print(f"Number of cells after filtering: {nr_cells1}")

    # Calculate areas and centroids
    areas_all = [prop.area * pixel_x * pixel_y for prop in props]
    areas_all1 = [prop.area * pixel_x * pixel_y for prop in filtered_props]
    size_from_area = np.sqrt(areas_all)
    size_from_area1 = np.sqrt(areas_all1)

    return size_from_area, size_from_area1



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

    # Identify NaN pixels for Phi data
    TF = np.isnan(B2)

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
                KAM_threshold = 0.0015
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

            # Label connected components
            labeled_array, num_features = label(BW_img)
            nr_cells = num_features  # Adjust for the exterior being labeled as a cell num_features - 1
            print(f"Number of cells: {nr_cells}")

            # Get region properties
            props = regionprops(labeled_array)

            min_cell_size = 2  # minimum size in pixel for a cell to be considered

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
                    filtered.append(region)

            # Iterate over each region in props
            for region in filtered:
                # Get the coordinates of the region
                region_coords = region.coords

                # Check if any of the coordinates overlap with the mask
                overlap = np.any(dilated_mask[region_coords[:, 0], region_coords[:, 1]])
                if not overlap:
                    filtered_props.append(region)



            nr_cells1 = len(filtered_props)

            print(f"Number of cells after filtering: {nr_cells1}")

            # Calculate areas and centroids
            areas_all = [prop.area * pixel_x * pixel_y for prop in props][0:] 
            areas_all1 = [prop.area * pixel_x * pixel_y for prop in filtered_props][0:] # Skip exterior [1:]
            size_from_area = np.sqrt(areas_all)
            size_from_area1 = np.sqrt(areas_all1)
            sizes.append(size_from_area)
            sizes1.append(size_from_area1)

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

area_sizes = list(area_sizes)
area_sizes1 = list(area_sizes1)
# print(av_cell)
plt.figure()
plt.plot(av_cell1, 'o')


# Define a function to fit distributions and plot
def fit_and_plot_distribution(dist, data, ax, label):
    # Remove NaNs and infinite values from data as well as values above 25 microns
    data = np.array(data)
    data = data[np.isfinite(data) & (data < 25)]

    # Ensure data is non-negative for certain distributions
    if dist in [lognorm, weibull_min, gamma]:
        data = data[data > 0]

    # Handle empty data or data with insufficient values
    if len(data) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(label)
        return

    try:
        params = dist.fit(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = dist.pdf(x, *params)
        ax.hist(data, bins=50, range=(0, 25), density=True, alpha=0.9)
        ax.plot(x, pdf, 'r-')
        # Perform the Kolmogorov-Smirnov test
        D, p_value = kstest(data, lambda x: dist.cdf(x, *params))

        # Annotate the plot with the KS test result
        ax.text(0.3, 0.75, f'Kolmogorov Smirnov\nD={D:.4e}\nP={p_value:.4e}', transform=ax.transAxes)

        ax.set_xlim(0, 25)
        ax.set_title(label)
        ax.set_xlabel('Cell size (mu)')
        ax.set_ylabel('PDF')
    except Exception as e:
        print(f"Error fitting {label}: {e}")
        ax.text(0.5, 0.5, 'Error in fitting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


# Loop over each dataset in area_sizes1
for i, dataset in enumerate(area_sizes1):
    # Create a new figure for each dataset
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # One row, five columns for each distribution

    # Fit and plot each distribution for the current dataset
    fit_and_plot_distribution(chi, dataset, axs[0], 'Chi Distribution')
    fit_and_plot_distribution(lognorm, dataset, axs[1], 'Lognormal Distribution')
    fit_and_plot_distribution(weibull_min, dataset, axs[2], 'Weibull Distribution')
    fit_and_plot_distribution(gamma, dataset, axs[3], 'Gamma Distribution')
    fit_and_plot_distribution(rayleigh, dataset, axs[4], 'Rayleigh Distribution')

    # Set the title for the figure as the name of the sample
    fig.suptitle(names[i], fontsize=24)

    # Adjust layout and show the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

plt.show()

