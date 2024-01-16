import numpy as np
import os
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
import fabio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.stats import lognorm




path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
com_phi = [file for file in files if 'com_phi' in file]
com_chi =[file for file in files if 'com_chi' in file]

pixel_x = 0.6575
pixel_y = 0.203

print(com_phi)
print(com_chi)

av_cell = []
av_cell1 = []

sizes = []
sizes1 = []
X = []
X1 = []
PDF = []
PDF1 =  []


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

    kernelSize = 3
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
    KAM_list = np.arange(0, 0.085, 0.0005).tolist()
    

    for _ in KAM_list:
        KAM_threshold = _
        KAM_filter = np.zeros_like(KAM, dtype=bool)

        # Apply the threshold to create the filter
        KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

        # Calculate the area ratio
        area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
        print(f'KAM mask: percentage in walls {area_ratio * 100:.2f}% with KAM threshold: {KAM_threshold}')
        
        if 0.69 < area_ratio < 0.72:
            # Morphological operations to get KAM_mask
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


            # Fit a log-normal distribution to the data
            shape, loc, scale = lognorm.fit(size_from_area)
            print(lognorm.fit(size_from_area))
            mu = np.log(scale)
            sigma = shape

            # Fit a log-normal distribution to the data
            shape1, loc1, scale1 = lognorm.fit(size_from_area1)
            # print(lognorm.fit(size_from_area1))
            mu1 = np.log(scale1)
            sigma1 = shape1

            # Generate values from the fitted distribution
            x = np.linspace(min(size_from_area), max(size_from_area), 100)
            X.append(x)
            pdf = lognorm.pdf(x, shape, loc, scale)
            PDF.append(pdf)

            # Generate values from the fitted distribution
            x1 = np.linspace(min(size_from_area1), max(size_from_area1), 100)
            X1.append(x1)
            pdf1 = lognorm.pdf(x1, shape1, loc1, scale1)
            PDF1.append(pdf1)
            

            # Display mean and median size
            mean_size = np.mean(size_from_area)
            median_size = np.median(size_from_area)
            print(f"Mean: {mean_size} mu, Median: {median_size} mu")
            # Display mean and median size
            mean_size1 = np.mean(size_from_area1)
            median_size1 = np.median(size_from_area1)
            print(f"Mean: {mean_size1} mu, Median: {median_size1} mu, with filtered cells")
            av_cell.append(mean_size)
            av_cell1.append(mean_size1)
            break


        elif area_ratio < 0.65:
            KAM_threshold = 0.001
            KAM_filter = np.zeros_like(KAM, dtype=bool)

                # Apply the threshold to create the filter
            KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

                # Calculate the area ratio
            area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
            print(f'KAM mask: percentage in walls {area_ratio * 100:.2f}% with KAM threshold: {KAM_threshold}')

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


            # Fit a log-normal distribution to the data
            shape, loc, scale = lognorm.fit(size_from_area)
            print(lognorm.fit(size_from_area))
            mu = np.log(scale)
            sigma = shape

            # Fit a log-normal distribution to the data
            shape1, loc1, scale1 = lognorm.fit(size_from_area1)
            # print(lognorm.fit(size_from_area1))
            mu1 = np.log(scale1)
            sigma1 = shape1

            # Generate values from the fitted distribution
            x = np.linspace(min(size_from_area), max(size_from_area), 100)
            X.append(x)
            pdf = lognorm.pdf(x, shape, loc, scale)
            PDF.append(pdf)

            # Generate values from the fitted distribution
            x1 = np.linspace(min(size_from_area1), max(size_from_area1), 100)
            X1.append(x1)
            pdf1 = lognorm.pdf(x1, shape1, loc1, scale1)
            PDF1.append(pdf1)
            

            # Display mean and median size
            mean_size = np.mean(size_from_area)
            median_size = np.median(size_from_area)
            print(f"Mean: {mean_size} mu, Median: {median_size} mu")
            # Display mean and median size
            mean_size1 = np.mean(size_from_area1)
            median_size1 = np.median(size_from_area1)
            print(f"Mean: {mean_size1} mu, Median: {median_size1} mu, with filtered cells")
            break



# print(av_cell)
plt.figure()
plt.plot(av_cell1, 'o')


n = len(sizes1)  # Or the length of X or PDF, assuming they are all the same

min_length = min(len(X1), len(PDF1), len(sizes1), len(av_cell1))
# Create a figure and axes
# Use math.ceil to handle odd numbers of plots
rows = n // 2 if n % 2 == 0 else n // 2 + 1
fig, axs = plt.subplots(rows, 2, figsize=(10, 5))  # Adjust the figsize as needed

# Flatten the axs array for easy indexing in case of a grid
axs = axs.flatten()

for i in range(min_length):
    # Plot histogram
    axs[i].hist(sizes1[i], bins=100, range=(0, 25), density=True, alpha=1)  # Adjust alpha for transparency

    # Plot line
    axs[i].plot(X1[i], PDF1[i], 'r-')
    axs[i].annotate(f'Mean: {av_cell1[i]:.2f} mu', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

    axs[i].set_xlim(0, 25)  # Adjust the x limits as needed
    axs[i].set_title(f'data {com_chi[i]}')  


# Show the plot
plt.show()