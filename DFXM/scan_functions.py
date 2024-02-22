import fabio
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from matplotlib.widgets import Slider
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize, remove_small_objects, binary_opening, dilation, medial_axis
from scipy.ndimage import label
from skimage import feature, exposure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm, lognorm, rayleigh, chi, maxwell, kstest
from scipy.optimize import curve_fit
from rtree import index
import time 
import csv
import os
import imageio

def load_data(path, type):

    # Load data from the given path
    # type: 'com' or 'FWHM', the two types of data that can is extracted from the DARFIX fits.
    # returns two lists, one with the phi files and one with the chi files of the given type

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    phi = [file for file in files if 'phi' in file and type in file]
    chi = [file for file in files if 'chi' in file and type in file]
    return phi, chi

def extract_number(file):

    # Extract the number from the file name
    # this key can be used to sort the files in the correct order
    # type_phi = sorted(type_phi, key=extract_number) if type_phi is the list of phi files

    u_index = file.find('u')
    substring = file[:u_index] if u_index != -1 else file
    return int(''.join(filter(str.isdigit, substring)))

def process_data(path, file, method, grain_mask=None):
    # Process the data from the given file based on the method
    if method == 'COM':
        file = fabio.open(os.path.join(path, file))
        A = file.data
        B1 = A.T
        B = np.flipud(B1)
        TF = np.isnan(B)
        average = np.nanmean(A)
        Img = B - average
        maximum = np.nanmax(Img)
        minimum = np.nanmin(Img)
        row_size, col_size = A.shape
        header = file.header
        return Img, maximum, minimum, average, TF, row_size, col_size, header
    elif method == 'FWHM':
        if grain_mask is None:
            raise ValueError("grain_mask and fwhm_chi must be provided for FWHM processing.")
        file = fabio.open(os.path.join(path, file))
        header = file.header
        A = file.data
        B1 = A.T
        B = np.flipud(B1)
        row_size, col_size = A.shape
        TF = np.isnan(A)
        B[~grain_mask] = np.NaN  # Apply grain mask and set outside values to 2
        B = np.minimum(np.abs(B), 2) 
        Img = B
        average = np.nanmean(A)
        maximum = np.nanmax(Img)
        minimum = np.nanmin(Img)
        return Img, maximum, minimum, average, TF, row_size, col_size, header
    else:
        raise ValueError("Invalid method. Please choose 'COM' or 'FWHM'.")


    

def find_grain(TF):

    grain = ~TF
    se = disk(3, strict_radius= True)
    grain = binary_erosion(grain, se)
    grain = binary_erosion(grain, se)

    return grain

def values_histogram(Img, maximum, grain):
    
    # Calculate the histogram of the image values
    # calculate the standard deviation of the values inside the grain
    # return the values inside the grain and the standard deviation of those values
    grain_mask = binary_dilation(grain, disk(3, strict_radius= True))
    grain_mask = binary_dilation(grain_mask, disk(3, strict_radius= True))
    Img[~grain_mask] = maximum * 2
    masked_values= Img[grain_mask]
    sigma = np.std(masked_values)

    return masked_values, sigma, grain_mask

def filter_grain(grain_mask, img, max_img):
    
    # Filter the grain mask
    img[~grain_mask] = max_img * 1.8

    return img


def scale_image(Img):
    
    # Scale the image between 0 and 1
    # return the scaled image

    Min, Max = np.nanmin(Img), np.nanmax(Img)
    scaled_Img = (Img - Min) / (Max - Min)

    return scaled_Img

def scale_image_global(Img, global_min, global_max):
    # Scale the image between 0 and 1 using global min and max
    scaled_Img = (Img - global_min) / (global_max - global_min)
    return scaled_Img
    

def RGB_image(chi_scaled, phi_scaled):

    # Create a RGB mosaicity map from the scaled chi and phi images
    # return the RGB image

    mosa = np.stack((chi_scaled, phi_scaled, np.ones_like(chi_scaled)), axis=-1)
    RGB_scaled = mosa.copy()
    RGB_scaled[np.isnan(RGB_scaled)] = 0
    RGB_scaled[RGB_scaled > 1] = 1
    RGB_scaled[RGB_scaled < 0] = 0
    RGB_scaled = colors.hsv_to_rgb(RGB_scaled)

    RGB_scaled = RGB_scaled * 0.85 #Make the colours pop baby

    return mosa, RGB_scaled

def calculate_KAM(col_size, row_size, grain_mask, Chi_Img, Phi_Img, kernelSize):

    KAM = np.zeros((col_size, row_size))
    print(KAM.shape)

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
                kernel_diff = np.sqrt(np.abs(Chi_Img[iStart:iEnd+1, jStart:jEnd+1] - Chi_Img[ii, jj])**2 + \
                            np.abs(Phi_Img[iStart:iEnd+1, jStart:jEnd+1] - Phi_Img[ii, jj])**2)
                nr_pixels_ROI = (iEnd - iStart + 1) * (jEnd - jStart + 1)

                # Store the average misorientation angle in the KAM map
                KAM[ii, jj] = np.sum(kernel_diff) / nr_pixels_ROI

    return KAM


def KAM_refine(KAM, grain_mask):
    # Create KAM filter and calculate area ratio
    KAM_list = np.arange(0.005, 0.085, 0.001).tolist()
    KAM_threshold_updated = False

    for value in KAM_list:
        KAM_threshold = value
        KAM_filter = np.zeros_like(KAM, dtype=bool)

        # Apply the threshold to create the filter
        KAM_filter[grain_mask & (KAM > KAM_threshold)] = True

        # Calculate the area ratio
        area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
        print(f'KAM mask: percentage in walls {area_ratio * 100:.2f}% with KAM threshold: {KAM_threshold}')
        
        # Adjust KAM_threshold based on the area ratio
        if 0.69 < area_ratio < 0.71 or area_ratio < 0.65:
            if area_ratio < 0.65:
                # Update KAM_threshold to 0.015 if the area ratio is less than 0.65
                KAM_threshold = 0.01
                KAM_threshold_updated = True
            break

    # Recompute KAM_filter with the final KAM_threshold, if it was updated
    if KAM_threshold_updated:
        KAM_filter = np.zeros_like(KAM, dtype=bool)
        KAM_filter[grain_mask & (KAM > KAM_threshold)] = True
    
    # Apply morphological operations to refine the KAM mask
    se = disk(1)
    #KAM_mask = binary_dilation(KAM_filter, se)
    KAM_mask = binary_erosion(KAM_filter, se)
    KAM_mask = binary_dilation(KAM_mask, se)
    
    # Skeletonize the refined KAM mask
    skel_KAM = skeletonize(KAM_mask)

    return KAM_mask, skel_KAM

def calculate_FWHM(Img, grain_mask, row_size, col_size, threshold, kernelSize):

    # Calculate the FWHM of the image
    # return the FWHM map and the FWHM mask
    # the threshold is used to filter the image and is in degrees
    # the kernelSize is used to define the kernel boundaries

    ll = 0
    kk = 0
    FWHM_filter = np.zeros((col_size, row_size))
    # Generate FWHM filter
    for ii in range(col_size):
        for jj in range(row_size):
            if grain_mask[ii, jj]:
                kk += 1
                iStart = max(ii - kernelSize, 0)
                iEnd = min(ii + kernelSize, col_size)
                jStart = max(jj - kernelSize, 0)
                jEnd = min(jj + kernelSize, row_size)

                kernel_sum = np.sum(Img[iStart:iEnd, jStart:jEnd])
                nr_pixels_ROI = (iEnd - iStart) * (jEnd - jStart)
                kernel_ave = kernel_sum / nr_pixels_ROI

                if kernel_ave > threshold:
                    FWHM_filter[ii, jj] = 1
                    ll += 1
    
    area_ratio = ll / kk
    print(f'Area ratio of FWHM mask:{area_ratio*100:.2f}% with threshold {threshold:.2f}')

    return FWHM_filter

def FWHM_mask(FWHM_filter, grain_mask):

    # Create a mask of the FWHM map
    # return the FWHM mask and the skeleton of the FWHM mask
    # the threshold is used to filter the FWHM map and is in degrees

    Edge_Img = binary_dilation(feature.canny(grain_mask), disk(1))
    FWHM_filter[Edge_Img] = 1
    FWHM_filter = binary_erosion(FWHM_filter, disk(1, strict_radius= True))
    FWHM_mask = binary_dilation(FWHM_filter, disk(1, strict_radius= True))
    skel_FWHM = skeletonize(FWHM_mask)

    return FWHM_mask, skel_FWHM

def find_regions(skel):

    # Find the regions in the skeleton
    # return the regions and the number of regions

    #BW_img = ~binary_dilation(skel, disk(1))
    BW_img = ~skel
    labeled_array, num_features = label(BW_img)
    nr_cells = num_features
    print(f'Number of regions: {nr_cells}')
    regions = regionprops(labeled_array)

    return regions, labeled_array

def filter_regions(regions, mosa, min_cell_size = 10):

    # Filter the regions in the skeleton
    # return the filtered regions
    # the min_cell_size is the minimum size of a cell in pixels

    mask = np.all(mosa == [1, 1, 1], axis = -1)
    mask = binary_erosion(mask, disk(3))

    filtered_regions = []
    for region in regions:
        region_coords = region.coords
        overlap = np.any(mask[region_coords[:, 0], region_coords[:, 1]]) # change to erroded_mask if needed
        if not overlap and region.area >= min_cell_size:
            filtered_regions.append(region)
    
    print(f'Number of filtered regions: {len(filtered_regions)}')
    return filtered_regions

def compute_bbox(coords):
    
        # Compute the bounding box of the region
        # return the bounding box
    
        min_row, min_col = np.min(coords, axis=0)
        max_row, max_col = np.max(coords, axis=0)
        return (min_row, min_col, max_row, max_col)

def dilate_mask(mask, size=5):
    """
    Dilates a given mask using a disk of a specified size.
    """
    return binary_dilation(mask, disk(size))

def parallel_dilate_masks(masks, size=5):
    """
    Dilates multiple masks in parallel.
    """
    with ThreadPoolExecutor() as executor:
        # Wrap the call to dilate_mask with size argument using lambda or partial
        results = list(tqdm(executor.map(lambda mask: dilate_mask(mask, size), masks), total=len(masks), desc="Dilating masks"))
    return results

def find_neighbours(regions, labeled_array):

    neighbours_dict = {prop.label: set() for prop in regions}
    idx = index.Index()
    masks = []
    
    for i, prop in enumerate(tqdm(regions, desc="Creating masks")):
        mask = np.zeros(labeled_array.shape, dtype=bool)
        mask[prop.coords[:, 0], prop.coords[:, 1]] = True
        masks.append(mask)
        bbox = compute_bbox(prop.coords)
        idx.insert(i, bbox)
    
    dilated_masks = parallel_dilate_masks(masks)

    for i, (prop, dilated_mask) in enumerate(tqdm(zip(regions, dilated_masks), desc="Finding neighbours")):
        dilated_bbox = compute_bbox(np.argwhere(dilated_mask))
        expanded_bbox = (dilated_bbox[0]-5, dilated_bbox[1]-5, dilated_bbox[2]+5, dilated_bbox[3]+5)

        # Query for nearby cells
        nearby_cells = list(idx.intersection(expanded_bbox))

        for j in nearby_cells:
            if j == i:
                continue

            other_prop = regions[j]
            other_bbox = compute_bbox(other_prop.coords)

            # Check if bounding boxes overlap
            if not (other_bbox[2] < expanded_bbox[0] or other_bbox[0] > expanded_bbox[2] or 
                    other_bbox[3] < expanded_bbox[1] or other_bbox[1] > expanded_bbox[3]):
                if np.any(dilated_mask[other_prop.coords[:, 0], other_prop.coords[:, 1]]):
                    neighbours_dict[prop.label].add(other_prop.label)
                    neighbours_dict[other_prop.label].add(prop.label)    
                
    return neighbours_dict

def neighbour_rotations(regions, neighbours_dict, chi_img, phi_img):

    # Precompute the average Chi and Phi for each cell
    # Now you can calculate the averages using Chi_Img and Phi_Img that have not been scaled for RGB values, true data
    ave_Chi = {prop.label: np.mean(chi_img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in regions}
    ave_Phi = {prop.label: np.mean(phi_img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in regions}


    chi_differences = []
    phi_differences = []
    num_neighbors = 0

    # Loop through each cell and its neighbors
    for cell_props in regions:
        cell_id = cell_props.label
        cell_ave_Chi = ave_Chi[cell_id]
        cell_ave_Phi = ave_Phi[cell_id]

        # Only look at neighbors that are in the dictionary and have a greater label
        neighbor_ids = [n_id for n_id in neighbours_dict.get(cell_id, []) if n_id > cell_id] 

        for neighbor_id in neighbor_ids:
            num_neighbors += 1
            neighbor_ave_Chi = ave_Chi[neighbor_id]
            neighbor_ave_Phi = ave_Phi[neighbor_id]

            # Calculate the differences and scale them
            chi_diff = (cell_ave_Chi - neighbor_ave_Chi) 
            phi_diff = (cell_ave_Phi - neighbor_ave_Phi) 

            # Add the differences to the lists
            chi_differences.append(chi_diff)
            phi_differences.append(phi_diff)
    
    return chi_differences, phi_differences, num_neighbors

def angular_difference(angle1, angle2):

    # Calculate the difference between two angles and adjust for wrap-around at 360 degrees

    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def neighbour_misorientation(regions, neighbours_dict, ave_Chi, ave_Phi):

    misorientations = []

    # Loop through each cell and its neighbors
    for cell_props in regions:
        cell_id = cell_props.label
        cell_Chi = ave_Chi[cell_id]
        cell_Phi = ave_Phi[cell_id]

        # Only look at neighbors that are in the dictionary and have a greater label
        neighbor_ids = [n_id for n_id in neighbours_dict.get(cell_id, []) if n_id > cell_id]

        for neighbor_id in neighbor_ids:
            neighbor_Chi = ave_Chi[neighbor_id]
            neighbor_Phi = ave_Phi[neighbor_id]

            # Calculate the angular differences for Chi and Phi
            chi_diff = angular_difference(cell_Chi, neighbor_Chi)
            phi_diff = angular_difference(cell_Phi, neighbor_Phi)

            # Combine the differences to estimate misorientation (simplified)
            # Note: This is a simplification and may not accurately represent crystallographic misorientation
            misorientation = np.sqrt(chi_diff**2 + phi_diff**2)
            
            misorientations.append(misorientation)
    return misorientations


def neighbour_GND(misorientations, ave_cell, k=1.5):
    # Burgers vector in Al considered here as sqrt(2)/2 * a_Al
    b_Al = 0.286e-9  

    # avergae cell size in meters
    D = ave_cell * 1e-6

    GND_densities = []

    for i in misorientations:
        misorientation_rad = np.deg2rad(i)  # Convert misorientation angle from degrees to radians

        # Calculate GND density using the corrected formula
        rho_GND = (k * misorientation_rad) / (b_Al * D)
        GND_densities.append(rho_GND)

    return GND_densities

def area_sizes(regions, pixel_y, pixel_x):
    # Calculate the area and size of each cell in micrometers
    # return the areas and sizes

    areas = [prop.area * pixel_y * pixel_x for prop in regions]
    sizes = [np.sqrt(area) for area in areas]

    return areas, sizes	

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
        ax.annotate(f'Mu: {mu:.2f}, Sigma: {shape:.2f}', xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.annotate(f'Mu/Sigma Ratio: {ratio_mu_sigma:.2f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.set_xlabel('Cell Size', fontsize=20)
        ax.set_ylabel('PDF', fontsize=20)
        ax.set_xlim(0, 18)
        #ax.set_title(label, fontsize=20)
        print(f'Fitted lognormal for {label} epsilon with mu={mu:.2f}, sigma={shape:.2f}, D={D:.2e}, p-value={p_value:.2e}, ratio={ratio_mu_sigma:.2f}')
    except Exception as e:
        print(f"Error fitting {label}: {e}")
        ax.text(0.5, 0.5, 'Error in fitting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    


def fit_and_plot_rayleigh(data, ax, label):
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
        params = rayleigh.fit(data)
        mu, sigma = params
        x = np.linspace(min(data), max(data), 100)
        pdf = rayleigh.pdf(x, *params)
        ax.hist(data, bins=50, range=(0, 1), density=True, alpha=0.9)
        ax.plot(x, pdf, 'r-', lw = 4)
        mean_size = np.mean(data)
        ax.annotate(f'Mean: {mean_size:.2f} mu', xy=(0.5, 0.7), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.annotate(f'Sigma: {sigma:.2f}', xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center', fontsize=14)
        ax.set_xlabel('Misorientation', fontsize=20)
        ax.set_ylabel('PDF', fontsize=20)
        ax.set_xlim(0, 18)
        #ax.set_title(label, fontsize=20)
        print(f'Fitted rayleigh for {label} epsilon with mu={mu:.2f}, sigma={sigma:.2f}, D={D:.2e}')
    except Exception as e:
        print(f"Error fitting {label}: {e}")
        ax.text(0.5, 0.5, 'Error in fitting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    

def anisotropy(regions):
    
    # Calculate the anisotropy of the cells
    # return the anisotropy of the cells
    
    major_axes = []
    minor_axes = []
    for prop in regions:
        major_axes.append(prop.major_axis_length)
        minor_axes.append(prop.minor_axis_length)
    
    return major_axes, minor_axes

def volume_fraction(areas, grain_mask, skeleton, pixel_x, pixel_y):

    # Calculate the volume fraction of the cells
    # return the volume fraction of the cells

    total_area = np.sum(areas)
    #BW_img = ~binary_dilation(skeleton, disk(1))
    BW_img = ~skeleton
    WB_img = ~BW_img
    mask_pixels = np.sum(WB_img) * pixel_y * pixel_x 
    grain_area = np.sum(grain_mask) * pixel_y * pixel_x
    grain_area = grain_area - mask_pixels

    volume_fraction = total_area / grain_area
    print(f'Volume fraction: {volume_fraction * 100:.2f}%')
    return volume_fraction


def create_2d_colormap(x, y, x_min, x_max, y_min, y_max):
    # Normalize x and y to range [0, 1]
    x_normalized = (x - x_min) / (x_max - x_min)
    y_normalized = (y - y_min) / (y_max - y_min)
    
    # Create RGB values based on x and y positions
    # Here we create a simple example where color varies with x and y linearly
    r = x_normalized
    g = np.ones_like(x_normalized)  # fixed at 1 for all points as per your description
    b = y_normalized
    
    # Ensure RGB values are between 0 and 1
    r = np.clip(r, 0, 1)
    b = np.clip(b, 0, 1)
    
    # Stack to create RGB array
    rgb = np.dstack((r, g, b))
    return rgb


def get_boundary_length_in_microns(boundary_pixels, pixel_size_x, pixel_size_y):
    """
    Calculate the length of a boundary in microns, taking into account anisotropic pixel dimensions.
    """
    distances = [np.sqrt(((p1[0] - p2[0]) * pixel_size_x) ** 2 + ((p1[1] - p2[1]) * pixel_size_y) ** 2)
                 for p1, p2 in zip(boundary_pixels[:-1], boundary_pixels[1:])]
    return sum(distances)

def get_pixel_boundary(labeled_array, cell1_label, cell2_label, dilated_masks):
    """
    Find the boundary pixels between two cells.
    """
    mask1 = dilated_masks[cell1_label]
    mask2 = dilated_masks[cell2_label]
    boundary = mask1 & mask2
    boundary_coords = np.argwhere(boundary)
    return boundary_coords.tolist()

def process_all_cells_and_neighbors(regions, labeled_array, neighbours_dict, pixel_size_x, pixel_size_y):
    """
    Calculate boundary lengths between each cell and its neighbors, storing results in the same dictionary,
    utilizing parallel processing for mask dilation.
    """
    # Prepare all masks for dilation
    masks = [(labeled_array == prop.label).astype(np.bool_) for prop in regions]
    
    # Dilate all masks in parallel
    dilated_masks_list = parallel_dilate_masks(masks)
    
    # Convert list of dilated masks back into a dictionary with prop.label as keys
    dilated_masks = {prop.label: mask for prop, mask in zip(regions, dilated_masks_list)}
    
    updated_neighbours_dict = {}
    for cell_id, neighbours in neighbours_dict.items():
        neighbour_list = list(neighbours)
        distance_list = []
        for neighbour_id in neighbour_list:
            boundary_pixels = get_pixel_boundary(labeled_array, cell_id, neighbour_id, dilated_masks)
            boundary_length = get_boundary_length_in_microns(boundary_pixels, pixel_size_x, pixel_size_y)
            distance_list.append(boundary_length)
        updated_neighbours_dict[cell_id] = [neighbour_list, distance_list]
    
    return updated_neighbours_dict

"""
## ADD TO Script if the cells want to be visualised interactively ## 
def plot_cells_and_neighbors(cell_id, ax):
    ax.clear()  # Clear the current axes
    
    # Obtain properties of all regions
    regions = regionprops(labeled_array)
    
    # Create masks for the cell and its neighbors
    cell_mask = (labeled_array == cell_id)
    neighbors_mask = np.zeros_like(cell_mask, dtype=bool)
    
    neighbors = neighbours_dict.get(cell_id, [])
    for neighbor_id in neighbors:
        neighbors_mask |= (labeled_array == neighbor_id)
    
    # Find the centroid of the cell
    cell_props = [prop for prop in regions if prop.label == cell_id]
    if cell_props:
        center_y, center_x = cell_props[0].centroid
    else:
        # Default center if cell is not found
        center_y, center_x = cell_mask.shape[0] // 2, cell_mask.shape[1] // 2
    
    # Define the region to display
    display_region = (max(int(center_y) - 300, 0), min(int(center_y) + 300, cell_mask.shape[0]),
                    max(int(center_x) - 300, 0), min(int(center_x) + 300, cell_mask.shape[1]))
    
    # Adjust the display region to ensure it is within bounds
    dy, dx = display_region[0], display_region[2]
    
    # Create RGBA images for the masks
    cell_mask_rgba = np.zeros((*cell_mask.shape, 4))
    neighbors_mask_rgba = np.zeros((*neighbors_mask.shape, 4))
    
    # Set RGBA colors
    cell_mask_rgba[cell_mask, :3] = [1, 0, 0]  # Red
    cell_mask_rgba[cell_mask, 3] = 0.7  # Alpha
    neighbors_mask_rgba[neighbors_mask, :3] = [0, 0, 1]  # Blue
    neighbors_mask_rgba[neighbors_mask, 3] = 0.7  # Alpha
    
    # Plot the original image within the display region
    ax.imshow(Cell_Img1[display_region[0]:display_region[1], display_region[2]:display_region[3]], cmap='jet', alpha=0.7)
    # Overlay the cell and neighbors masks within the same region
    ax.imshow(cell_mask_rgba[display_region[0]:display_region[1], display_region[2]:display_region[3]])
    ax.imshow(neighbors_mask_rgba[display_region[0]:display_region[1], display_region[2]:display_region[3]])
    
    # Draw lines between the cell and its neighbors using centroids
    for neighbor_id in neighbors:
        neighbor_props = [prop for prop in regions if prop.label == neighbor_id]
        if neighbor_props:
            center_y_neighbor, center_x_neighbor = neighbor_props[0].centroid
            
            # Adjust centroid coordinates to the display region
            adjusted_center_x = center_x - dx
            adjusted_center_y = center_y - dy
            adjusted_center_x_neighbor = center_x_neighbor - dx
            adjusted_center_y_neighbor = center_y_neighbor - dy
            
            # Draw a line from the current cell to the neighbor
            ax.plot([adjusted_center_x, adjusted_center_x_neighbor], [adjusted_center_y, adjusted_center_y_neighbor], 'yellow')
    
    ax.axis('off')

# Set up the figure and slider as before
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = Slider(ax_slider, 'Cell ID', 1, max(neighbours_dict.keys()), valinit=1, valstep=1)

def update(val):
    plot_cells_and_neighbors(int(val), ax)
    fig.canvas.draw_idle()

slider.on_changed(update)

plot_cells_and_neighbors(1, ax)

plt.show()
"""
