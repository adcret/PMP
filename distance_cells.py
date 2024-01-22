import numpy as np
import os
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
import fabio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import label
from skimage.measure import regionprops
import math


pixel_x = 0.6575
pixel_y = 0.203

path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
com_phi = [file for file in files if 'com_phi' in file and '838' in file]
com_chi =[file for file in files if 'com_chi' in file and '838' in file]

print(com_phi)
print(com_chi)

chi_file = fabio.open(path + com_chi[0])
A = chi_file.data
row_size, col_size = A.shape
B1 = A.T
B = np.flipud(B1)
TF = np.isnan(B)
ave_chi = np.nanmean(A)
Chi_Img = B - ave_chi
# Read in PHI COM data
phi_file = fabio.open(path + com_phi[0])
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
KAM_threshold = 0.07
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

# Overlay the skeleton on Chi_Img, Phi_Img, and Mosa_Img
Chi_Img_overlay = np.where(skel_Img, 0.99, Chi_Img) # This somehow leads to much larger skel....
Phi_Img_overlay = np.where(skel_Img, 0.99, Phi_Img)
Mosa_Img_overlay = np.copy(Mosa_Img)
Mosa_Img_overlay[skel_Img] = [2.5, 2.5, 2.5]  # Assuming Mosa_Img is 3-channel


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

# Iterate over each region in propss
for region in filtered:
    # Get the coordinates of the region
    region_coords = region.coords

    # Check if any of the coordinates overlap with the mask
    overlap = np.any(dilated_mask[region_coords[:, 0], region_coords[:, 1]])
    if not overlap:
        filtered_props.append(region)


nr_cells1 = len(filtered_props)
print(f"Number of cells: {nr_cells1}")

# Calculate areas and centroids
areas_all = [prop.area * pixel_x * pixel_y for prop in props][0:] 
areas_all1 = [prop.area * pixel_x * pixel_y for prop in filtered_props][0:] # Skip exterior [1:]

centroids = np.array([prop.centroid for prop in props])[0:] 

size_from_area = np.sqrt(areas_all)
size_from_area1 = np.sqrt(areas_all1)


# Display mean and median size
mean_size = np.mean(size_from_area)
median_size = np.median(size_from_area)
print(f"Mean: {mean_size} mu, Median: {median_size} mu")
# Display mean and median size
mean_size1 = np.mean(size_from_area1)
median_size1 = np.median(size_from_area1)
print(f"Mean: {mean_size1} mu, Median: {median_size1} mu")

# Create and plot the Cell_Img
Cell_Img = np.copy(Mosa_Img)
for ii in range(1, nr_cells):  # Skip exterior nr_cells + 1
    cellPixels = props[ii].coords
    cell_ave_Chi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 0])
    cell_ave_Phi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 1])
    for row, col in cellPixels:
        Cell_Img[row, col, 0] = cell_ave_Chi
        Cell_Img[row, col, 1] = cell_ave_Phi
        Cell_Img[row, col, 2] = 0  # Set blue channel to 0

Cell_Img1 = np.ones_like(Mosa_Img)
for ii in range(1, nr_cells1):  # Skip exterior nr_cells + 1
    cellPixels = filtered_props[ii].coords
    cell_ave_Chi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 0])
    cell_ave_Phi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 1])
    for row, col in cellPixels:
        Cell_Img1[row, col, 0] = cell_ave_Chi
        Cell_Img1[row, col, 1] = cell_ave_Phi
        Cell_Img1[row, col, 2] = 0  # Set blue channel to 0




# Overlay on skeletonized image
Cell_Img[skel_Img] = [0, 0, 0]
Cell_Img1[skel_Img] = [0, 0, 0]

# Plot the Cell_Img
plt.figure()
plt.imshow(Cell_Img, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.title('Cells and skeleton')
plt.draw()

# Plot the Cell_Img
plt.figure()
plt.imshow(Cell_Img1, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.title('Cells and skeleton')
plt.draw()

############################################################################################################
############################################################################################################
#######                                                                                           ##########
#######                                   LOOKING AT SELECTED ORIENTATIONS                        ##########
#######                                                                                           ##########
############################################################################################################
############################################################################################################

chi = np.copy(Chi_Img)
phi = np.copy(Phi_Img)

# Assuming Chi_Img[grain1] and Phi_Img[grain1] are your data arrays
orientations = np.stack((Chi_Img[grain1], Phi_Img[grain1]), axis=-1)
orientations_image = np.stack((Chi_Img, Phi_Img, np.ones_like(Chi_Img)), axis=-1)

# Creating combined mask
combined_mask = (Chi_Img > -0.15) & (Chi_Img < 0.15) & (Phi_Img> -0.15) & (Phi_Img < -0.12)

plt.figure()
plt.imshow(combined_mask, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])

combined_mask = binary_erosion(combined_mask, se)
combined_mask = binary_dilation(combined_mask, se)

plt.figure()
plt.imshow(combined_mask, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])

# making the image with the mask applied
orientations_image[~combined_mask] = [0, 0, 0]

# Find the indices where the mask is True
y_indices, x_indices = np.where(combined_mask)


# Compute the 2D histogram
Z, xedges, yedges = np.histogram2d(orientations[:, 1], orientations[:, 0], bins=[25, 22])

# Create meshgrid
X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')

# Plotting the original contour
plt.figure()
plt.contour(X, Y, Z, levels=25, linewidths=1.5, cmap='jet')
plt.xlabel('Chi')
plt.ylabel('Phi')
plt.title('Orientation distribution, frequency map')

mosa1 = colors.hsv_to_rgb(orientations_image)  # Convert HSV to RGB
# Overlay skeleton on Mosa overlay and plot
Mosa_overlay = np.copy(mosa1)  # Assuming mosa is in RGB format
Mosa_overlay[skel_Img] = [0, 0, 0]
plt.figure()
plt.imshow(Mosa_overlay, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.axis('equal')
plt.title('KAM skeleton overlay on Mosa map with highlighted skeleton for selected orientations')

orientation_props = []
overlap_threshold = 0.1  # 60% overlap

for region in props:
    region_coords = region.coords
    # Calculate the number of pixels in the region that are True in the combined_mask
    overlap_count = np.sum(combined_mask[region_coords[:, 0], region_coords[:, 1]])
    # Calculate the overlap percentage
    overlap_percentage = overlap_count / len(region_coords)
    # Check if the overlap percentage is above the threshold
    if overlap_percentage > overlap_threshold:
        orientation_props.append(region)


nr_cells_orientation = len(orientation_props)
print(f"Number of cells with given orientation: {nr_cells_orientation}")

orinetation_centroids = np.array([prop.centroid for prop in orientation_props])[0:]

orinetation_centroids = [[sublist[0]*pixel_y, sublist[1]*pixel_x] for sublist in orinetation_centroids]

orientation_x =[sublist[0] for sublist in orinetation_centroids]
orientation_y =[sublist[1] for sublist in orinetation_centroids]

distances =[]
for _ in range(len(orinetation_centroids)):
    a = orinetation_centroids[_]
    for i in range(len(orinetation_centroids)):
        b = orinetation_centroids[i]
        if i == _:
            continue
        else:
            distances.append(math.dist(a, b))

distance_vectors = []
print('length of orinetation_centroids', len(orinetation_centroids))    
for i in range(len(orinetation_centroids)):
    a = orinetation_centroids[i]
    for j in range(len(orinetation_centroids)):
        if i != j and i < j:
            b = orinetation_centroids[j]
            # Calculate the vector from a to b
            vector = (abs(b[0] - a[0]), abs(b[1] - a[1]))
            distance_vectors.append(vector)
    
print('Difference vector calculation done!')
plt.figure()
plt.hist(distances, bins=100)
plt.xlabel('Distance in micrometer')
plt.ylabel('Frequency')
plt.title('Distance between cells with given orientation')


# Convert the list of vectors to a NumPy array for easy plotting
distance_vectors = np.array(distance_vectors)

print(f"Number of distance vectors: {len(distance_vectors)}")

X = [sublist[0] for sublist in distance_vectors]
Y = [sublist[1] for sublist in distance_vectors]


plt.figure()
plt.scatter(X, Y, s=1)
plt.xlabel('Distance along X')
plt.ylabel('Distance along Y')
plt.title('Scatter Plot of Distance Vectors')

# Compute the 2D histogram
Z, xedges, yedges = np.histogram2d(X, Y, bins=[30, 50])

# Create meshgrid
X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')

# Plotting the contour
plt.figure()
plt.contour(X, Y, Z, levels=7, linewidths=1.5, cmap='jet')
plt.xlabel('Distance along X')
plt.ylabel('Distance along Y')
plt.title('Distance vector distribution, frequency map')


Cell_Img_orientations = np.ones_like(Mosa_Img)  # Start with a blank image
for ii in range(len(orientation_props)):  # Iterate through all cells with the desired orientation
    cellPixels = orientation_props[ii].coords
    cell_ave_Chi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 0])
    cell_ave_Phi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 1])
    for row, col in cellPixels:
        Cell_Img_orientations[row, col, 0] = cell_ave_Chi
        Cell_Img_orientations[row, col, 1] = cell_ave_Phi
        Cell_Img_orientations[row, col, 2] = 0  # Set blue channel to 0

# Overlay the skeleton on the Cell_Img_orientations
Cell_Img_orientations[skel_Img] = [0, 0, 0]

int_centroids = np.round(orinetation_centroids).astype(int)
# Initialize an array to hold the pixel values
centroid_pixel_values = np.zeros((len(int_centroids), orientations_image.shape[2]))

# Loop over the centroids and extract the pixel values
for i, (x, y) in enumerate(int_centroids):
    centroid_pixel_values[i] = orientations_image[x, y]


centroid_pixel_values = [list(sub_array[:2]) for sub_array in centroid_pixel_values]

# Assuming centroid_pixel_values is already defined
x_values, y_values = zip(*centroid_pixel_values)

# Convert to numpy arrays for easier handling of NaNs
x_values = np.array(x_values)
y_values = np.array(y_values)

# Check for NaNs and handle them
# For example, you could remove the pairs where either x or y is NaN
mask = ~np.isnan(x_values) & ~np.isnan(y_values)
x_values = x_values[mask]
y_values = y_values[mask]

# Now you can compute the 2D histogram
Z, xedges, yedges = np.histogram2d(x_values, y_values, bins=[25, 22])

# Create meshgrid
X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')

# Plotting the contour
plt.figure()
plt.contour(X, Y, Z, levels=25, linewidths=1.5, cmap='jet')
plt.xlabel('Chi')
plt.ylabel('Phi')
plt.title('Orientation distribution of cells, frequency map')
#plt.show()

orientation_centroids = np.array([prop.centroid for prop in orientation_props])[0:]
# Transform centroids to pixel values
transformed_centroids = np.copy(orientation_centroids)
transformed_centroids[:, 0] *= pixel_y  # Scale y-coordinates (rows)
transformed_centroids[:, 1] *= pixel_x  # Scale x-coordinates (columns)

# Create a mask and plot
mask = np.ones_like(Mosa_Img)
mask[skel_Img] = [0, 0, 0]
mask = np.flipud(mask)
plt.figure()
plt.imshow(mask, cmap='gray', origin='upper', extent=[0, pixel_x * row_size, 0, pixel_y * col_size])
plt.scatter(transformed_centroids[:, 1], transformed_centroids[:, 0], c='red', s=1)  # Scatter plot 
plt.xlabel('x in microns')
plt.ylabel('y in microns')
plt.title('Centroid on skeletonized image')
plt.gca().invert_yaxis()

# Flip the image data upside down
flipped_mask = np.flipud(mask)

# Define the thresholds for yellow
red_threshold = 0.84  # adjust as needed
green_threshold = 0.82  # adjust as needed
blue_threshold = 0.001  # adjust as needed

# Find yellow pixels
yellow_pixels = (Cell_Img_orientations[:, :, 0] > red_threshold) & \
                (Cell_Img_orientations[:, :, 1] > green_threshold) & \
                (Cell_Img_orientations[:, :, 2] < blue_threshold)

# Set yellow pixels to white
Cell_Img_orientations[yellow_pixels] = [1, 1, 1]


# Plot the Cell_Img
plt.figure()
plt.imshow(Cell_Img_orientations, extent=[0, pixel_x * row_size, 0, pixel_y * col_size])
plt.xlabel('x in micrometer')
plt.ylabel('y in micrometer')
plt.title('Cells and skeleton of selected orientations')
plt.draw()

plt.show()