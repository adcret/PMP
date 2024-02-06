import fabio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from matplotlib.widgets import Slider
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize, remove_small_objects, binary_opening, dilation
from scipy.ndimage import label
from skimage.feature import canny
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import norm, lognorm, rayleigh, chi, maxwell
from scipy.optimize import curve_fit
from rtree import index
import time 
import csv
import os

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

def process_data(path, file):

    # Process the data from the given file
    
    chi_file = fabio.open(os.path.join(path, file))
    A = chi_file.data
    row_size, col_size = A.shape
    B1 = A.T
    B = np.flipud(B1)
    TF = np.isnan(B)
    average = np.nanmean(A)
    Img = B - average
    maximum = np.nanmax(Img)

    return Img, maximum, average, TF, row_size, col_size

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

def RGB_image(chi_scaled, phi_scaled):

    # Create a RGB mosaicity map from the scaled chi and phi images
    # return the RGB image

    mosa = np.stack((chi_scaled, phi_scaled, np.ones_like(chi_scaled)), axis=-1)
    RGB_scaled = mosa.copy()
    RGB_scaled[np.isnan(RGB_scaled)] = 0
    RGB_scaled[RGB_scaled > 1] = 1
    RGB_scaled[RGB_scaled < 0] = 0
    RGB_scaled = colors.hsv_to_rgb(RGB_scaled)

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
                kernel_diff = np.abs(Chi_Img[iStart:iEnd+1, jStart:jEnd+1] - Chi_Img[ii, jj]) + \
                            np.abs(Phi_Img[iStart:iEnd+1, jStart:jEnd+1] - Phi_Img[ii, jj])
                nr_pixels_ROI = (iEnd - iStart + 1) * (jEnd - jStart + 1)

                # Store the average misorientation angle in the KAM map
                KAM[ii, jj] = np.sum(kernel_diff) / nr_pixels_ROI

    return KAM

def KAM_refine(KAM, grain_mask, threshold):

    # Create a mask of the KAM map
    # return the KAM mask and the skeleton of the KAM mask
    # the threshold is used to filter the KAM map and is in degrees

    KAM_filter = np.zeros_like(KAM, dtype=bool)
    KAM_filter[grain_mask & (KAM > threshold)] = True
    area_ratio = np.sum(KAM_filter) / np.sum(grain_mask)
    KAM = binary_erosion(KAM_filter, disk(1, strict_radius= True))
    KAM_mask = binary_dilation(KAM, disk(1, strict_radius= True))
    skel_KAM = skeletonize(KAM_mask)
    print(f'Area ratio of KAM mask:{area_ratio*100:.2f}% with threshold {threshold:.2f}')

    return KAM_mask, skel_KAM

def calculate_FWHM(Img, grain_mask, row_size, col_size, threshold, kernelSize):

    # Calculate the FWHM of the image
    # return the FWHM map and the FWHM mask
    # the threshold is used to filter the image and is in degrees
    # the kernelSize is used to define the kernel boundaries

    ll = 0
    kk = 0
    FWHM_filter = np.zeros((row_size, col_size))
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

    Edge_Img = binary_dilation(canny(grain_mask), disk(1))
    FWHM_filter[Edge_Img] = 1
    FWHM_filter = binary_erosion(FWHM_filter, disk(1, strict_radius= True))
    FWHM_mask = binary_dilation(FWHM_filter, disk(1, strict_radius= True))
    skel_FWHM = skeletonize(FWHM_mask)

    return FWHM_mask, skel_FWHM

def find_regions(skel):

    # Find the regions in the skeleton
    # return the regions and the number of regions

    BW_img = ~binary_dilation(skel, disk(1))
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
    erroded_mask = binary_erosion(mask, disk(3))
    dilated_mask = binary_dilation(erroded_mask, disk(3))
    dilated_mask = binary_dilation(dilated_mask, disk(3))
    dilated_mask = binary_dilation(dilated_mask, disk(20))

    filtered_regions = []
    for region in regions:
        region_coords = region.coords
        overlap = np.any(erroded_mask[region_coords[:, 0], region_coords[:, 1]])
        if not overlap and region.area > min_cell_size:
            filtered_regions.append(region)
    
    print(f'Number of filtered regions: {len(filtered_regions)}')
    return filtered_regions

def compute_bbox(coords):
    
        # Compute the bounding box of the region
        # return the bounding box
    
        min_row, min_col = np.min(coords, axis=0)
        max_row, max_col = np.max(coords, axis=0)
        return (min_row, min_col, max_row, max_col)

def dilate_mask(mask):
    return binary_dilation(mask, disk(5))

def parallel_dilate_masks(masks):
    with ProcessPoolExecutor() as executor:
        result = list(tqdm(executor.map(dilate_mask, masks), total=len(masks), desc="Dilating masks in parallel"))
    return result

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


def neighbour_GND(misorientations, k=1.5):
    # Constants for Aluminium
    b_Al = 0.286e-9  # Burgers vector in meters
    
    # Effective pixel sizes in micrometers (mu)
    pixel_y = 0.203
    pixel_x = 0.6575

    GND_densities = []

    for i in misorientations:
        misorientation_rad = np.deg2rad(i)

        # Calculate GND density
        rho_GND = (k * misorientation_rad) / (b_Al * 0.0000000003)
        GND_densities.append(rho_GND)

    return GND_densities