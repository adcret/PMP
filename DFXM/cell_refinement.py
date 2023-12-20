import fabio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import norm
from skimage.morphology import binary_erosion, binary_dilation, disk, skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

class ImageProcessor:
    """ Class for processing image data. """

    def __init__(self, file_path, pixel_x, pixel_y):
        self.file_path = file_path
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y


    def load_and_process_image(self, filename):
        file = fabio.open(self.file_path + filename)
        A = file.data
        A = np.flipud(A.T)
        A -= np.nanmean(A)  # Normalize
        return A, np.nanmax(A)

    def find_grain(self, image, A, B, max_chi, max_phi, outside=1.8, radius=3):
        TF = np.isnan(image)
        grain = ~TF
        se = disk(radius, strict_radius=True)
        grain = binary_erosion(grain, se)
        grain = binary_erosion(grain, se)
        grain = binary_dilation(grain, se)
        grain = binary_dilation(grain, se)
        A[~grain] = max_chi * outside
        B[~grain] = max_phi * outside
        return grain, A, B


    def RGB_image(self, A, B):
        min_valA, max_valA = np.nanmin(A), np.nanmax(A)
        A_Scale =  (A - min_valA) / (max_valA - min_valA)
        min_valB, max_valB = np.nanmin(B), np.nanmax(B)
        B_Scale =  (B - min_valB) / (max_valB - min_valB)
        Mosa_Img = np.stack((A_Scale, B_Scale, np.ones_like(A_Scale)), axis=-1)
        Mosa_Img = np.where(np.isnan(Mosa_Img), 0, Mosa_Img)
        Mosa_Img = np.clip(Mosa_Img, 0, 1)
        return Mosa_Img


    def calculate_kam(self, chi_img, phi_img, mask, kernel_size = 2):
        kam = np.zeros_like(chi_img)
        for ii in range(chi_img.shape[0]):
            for jj in range(chi_img.shape[1]):
                if mask[ii, jj]:
                    i_start, i_end = max(ii - kernel_size, 0), min(ii + kernel_size, chi_img.shape[0] - 1)
                    j_start, j_end = max(jj - kernel_size, 0), min(jj + kernel_size, chi_img.shape[1] - 1)
                    kernel_diff = np.abs(chi_img[i_start:i_end+1, j_start:j_end+1] - chi_img[ii, jj]) + \
                                  np.abs(phi_img[i_start:i_end+1, j_start:j_end+1] - phi_img[ii, jj])
                    kam[ii, jj] = np.sum(kernel_diff) / ((i_end - i_start + 1) * (j_end - j_start + 1))
        return kam
    

    def create_kam_filter(self, kam, threshold, grain_mask):
        kam_filter = np.zeros_like(kam, dtype=bool)
        kam_filter[grain_mask & (kam > threshold)] = True
        return kam_filter

    def calculate_area_ratio(self, filter_mask, grain_mask):
        return np.sum(filter_mask) / np.sum(grain_mask)

    def create_skeleton(self, filter_mask, radius=1):
        se = disk(radius)
        filter_mask = binary_erosion(filter_mask, se)
        filter_mask = binary_dilation(filter_mask, se)
        skel_Img = skeletonize(filter_mask)
        return skel_Img

    def overlay_skeleton(images, skeleton, method='standard'):
        overlaid_images = []
        for img in images:
            if img.ndim == 3:  # For RGB images
                if method == 'method1':
                    overlay = np.where(skeleton[..., np.newaxis], [0, 0, 0], img)
                else:  # method2
                    overlay = np.copy(img)
                    overlay[skeleton] = [2.5, 2.5, 2.5]
            else:  # For grayscale images
                if method == 'method1':
                    overlay = np.where(skeleton, 2.5, img)
                else:  # method2
                    overlay = np.where(skeleton, 0.99, img)
            overlaid_images.append(overlay)

        return overlaid_images
    
    def calculate_fwhm(self, A, B, grain):
        A[~grain] = 2
        B[~grain] = 2
        A = np.minimum(np.abs(A), 2)
        B = np.minimum(np.abs(B), 2)
        FWHM_Img = A + B
        FWHM_Img[FWHM_Img > 4] = 4
        return FWHM_Img


    def create_fwhm_filter(self, fwhm_img, grain_mask, threshold, kernel_size=2):
        fwhm_filter = np.zeros_like(fwhm_img, dtype=bool)
        count_filtered = 0
        total_count = 0

        for ii in range(fwhm_img.shape[0]):
            for jj in range(fwhm_img.shape[1]):
                if grain_mask[ii, jj]:
                    total_count += 1
                    i_start = max(ii - kernel_size, 0)
                    i_end = min(ii + kernel_size, fwhm_img.shape[0])
                    j_start = max(jj - kernel_size, 0)
                    j_end = min(jj + kernel_size, fwhm_img.shape[1])

                    kernel_sum = np.sum(fwhm_img[i_start:i_end, j_start:j_end])
                    nr_pixels_roi = (i_end - i_start) * (j_end - j_start)
                    kernel_ave = kernel_sum / nr_pixels_roi

                    if kernel_ave > threshold:
                        fwhm_filter[ii, jj] = True
                        count_filtered += 1

        area_ratio = count_filtered / total_count
        return fwhm_filter, area_ratio

    def calculate_cell_properties(self, labeled_array, mask, min_cell_size = 2):
        props = regionprops(labeled_array)
        filtered_props = [prop for prop in props if not np.any(mask[prop.coords[:, 0], prop.coords[:, 1]]) and prop.area >= min_cell_size]
        return filtered_props

    def calculate_cell_sizes(self, properties, pixel_x, pixel_y):
        areas = [prop.area * pixel_x * pixel_y for prop in properties]
        return np.sqrt(areas)

    def plot_centroids(self, properties, base_image, color, size):
        centroids = np.array([prop.centroid for prop in properties])
        plt.figure()
        plt.imshow(base_image, cmap='gray')
        plt.scatter(centroids[:, 1], centroids[:, 0], c=color, s=size)
        plt.xlabel('x in pixel')
        plt.ylabel('y in pixel')
        plt.title('Centroid on skeletonized image')

    def create_cell_img(self, properties, base_img, nr_cells):
        cell_img = np.ones_like(base_img)
        for ii in range(1, nr_cells):
            cell_pixels = properties[ii].coords
            cell_ave_chi = np.mean(base_img[cell_pixels[:, 0], cell_pixels[:, 1], 0])
            cell_ave_phi = np.mean(base_img[cell_pixels[:, 0], cell_pixels[:, 1], 1])
            for row, col in cell_pixels:
                cell_img[row, col, 0] = cell_ave_chi
                cell_img[row, col, 1] = cell_ave_phi
                cell_img[row, col, 2] = 0  # Set blue channel to 0
        return cell_img

    def calculate_cell_properties(self, labeled_array, grain_mask, pixel_x, pixel_y):
        """ Calculate properties of cells. """
        properties = regionprops(labeled_array)
        areas = []
        centroids = []

        for prop in properties:
            if grain_mask[prop.coords[:, 0], prop.coords[:, 1]].any():
                areas.append(prop.area * pixel_x * pixel_y)
                centroids.append(prop.centroid)

        return np.sqrt(areas), centroids

    def plot_cell_properties(self, skel_img, centroids, base_image, title):
        """ Plot centroids on a base image. """
        plt.figure()
        plt.imshow(skel_img, cmap='gray')
        plt.imshow(base_image)
        plt.scatter([c[1] for c in centroids], [c[0] for c in centroids], c='black', s=1)  # Scatter plot
        plt.xlabel('x in pixel')
        plt.ylabel('y in pixel')
        plt.title(title)
        plt.draw()

    def create_adjacency_list(self, props):
        neighbors_dict = {prop.label: set() for prop in props}
        dist_transform = distance_transform_edt(~skel_Img)

        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            bbox_dilated = dilation(labeled_array[minr:maxr, minc:maxc], disk(1))
            unique_labels = np.unique(bbox_dilated)
            neighbors_dict[prop.label].update(label for label in unique_labels if label != prop.label and label > 0)
        
        return neighbors_dict

    def calculate_orientation_differences(self, props, neighbors_dict, Mosa_Img, MaxChi, MinChi, MaxPhi, MinPhi):
        chi_differences = []
        phi_differences = []

        for cell_props in props:
            cell_id = cell_props.label
            cell_ave_Chi, cell_ave_Phi = self._calculate_averages(Mosa_Img, cell_props.coords)
            for neighbor_id in neighbors_dict.get(cell_id, []):
                neighbor_props = next((prop for prop in props if prop.label == neighbor_id), None)
                if neighbor_props:
                    neighbor_ave_Chi, neighbor_ave_Phi = self._calculate_averages(Mosa_Img, neighbor_props.coords)
                    chi_diff, phi_diff = self._calculate_diffs(cell_ave_Chi, cell_ave_Phi, neighbor_ave_Chi, neighbor_ave_Phi, MaxChi, MinChi, MaxPhi, MinPhi)
                    chi_differences.append(chi_diff)
                    phi_differences.append(phi_diff)

        return chi_differences, phi_differences

    def _calculate_averages(self, Mosa_Img, coords):
        ave_Chi = np.mean(Mosa_Img[coords[:, 0], coords[:, 1], 0])
        ave_Phi = np.mean(Mosa_Img[coords[:, 0], coords[:, 1], 1])
        return ave_Chi, ave_Phi

    def _calculate_diffs(self, cell_ave_Chi, cell_ave_Phi, neighbor_ave_Chi, neighbor_ave_Phi, MaxChi, MinChi, MaxPhi, MinPhi):
        chi_diff = (cell_ave_Chi - neighbor_ave_Chi) * (MaxChi - MinChi) + MinChi
        phi_diff = (cell_ave_Phi - neighbor_ave_Phi) * (MaxPhi - MinPhi) + MinPhi
        return chi_diff, phi_diff

    def plot_orientation_differences(self, chi_differences, phi_differences, title):
        plt.figure()
        plt.scatter(chi_differences, phi_differences, alpha=0.5, s=10)
        plt.xlabel('Difference in Chi Orientation')
        plt.ylabel('Difference in Phi Orientation')
        plt.title(title)
        plt.draw()

    # Function to reverse scaling
    def reverse_scaling(scaled_img, min_val, max_val):
        return (scaled_img * (max_val - min_val)) + min_val

    # Function to calculate average values
    def calculate_averages(img, props):
        return {prop.label: np.mean(img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in props}

    # Function to calculate differences
    def calculate_differences(props, ave_vals, neighbors_dict):
        differences = []
        for prop in props:
            cell_id = prop.label
            cell_val = ave_vals[cell_id]
            neighbor_ids = [n_id for n_id in neighbors_dict.get(cell_id, []) if n_id > cell_id]
            for neighbor_id in neighbor_ids:
                neighbor_val = ave_vals[neighbor_id]
                differences.append(cell_val - neighbor_val)
        return differences

    # Function to create a contour plot
    def create_contour_plot(data, title, xlabel, ylabel, cmap='viridis'):
        plt.figure()
        x_edges = np.linspace(min(data[0]), max(data[0]), 50)
        y_edges = np.linspace(min(data[1]), max(data[1]), 50)
        H, x_edges, y_edges = np.histogram2d(data[0], data[1], bins=[x_edges, y_edges])
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        plt.contourf(X, Y, H.T, levels=100, cmap=cmap)
        plt.colorbar(label='Count')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        return X, Y, H.T

    # Main processing
    def process_image(Mosa_Img, filtered_props, neighbors_dict, MinChi, MaxChi, MinPhi, MaxPhi):
        # Extract and reverse scale
        Chi_reversed = reverse_scaling(Mosa_Img[:, :, 0], MinChi, MaxChi)
        Phi_reversed = reverse_scaling(Mosa_Img[:, :, 1], MinPhi, MaxPhi)

        # Calculate averages
        ave_Chi = calculate_averages(Chi_reversed, filtered_props)
        ave_Phi = calculate_averages(Phi_reversed, filtered_props)

        # Calculate differences
        chi_differences = calculate_differences(filtered_props, ave_Chi, neighbors_dict)
        phi_differences = calculate_differences(filtered_props, ave_Phi, neighbors_dict)

        print(f'Number of neighbors: {len(chi_differences)}')

        # Plot scatter plot
        plt.scatter(chi_differences, phi_differences, alpha=0.5, s=10)
        plt.xlabel('Difference in Chi Orientation')
        plt.ylabel('Difference in Phi Orientation')
        plt.title('Orientation Differences Between Each Cell and Its Neighbors')
        plt.show()

        # Contour Plot
        X, Y, Z = create_contour_plot([chi_differences, phi_differences],
                                    'Contour Plot with Orientation Differences',
                                    'Difference in Chi Orientation',
                                    'Difference in Phi Orientation')

        # Fit Gaussian and plot
        initial_guess = [35, 0, 0, 1, 1, 5, 0]  # Initial guess for Gaussian parameters
        popt, _ = curve_fit(gaussian, [X.ravel(), Y.ravel()], Z.ravel(), p0=initial_guess, bounds=(0, np.inf))
        plt.contour(X, Y, gaussian([X, Y], *popt).reshape(X.shape), colors='w')
        plt.show()

        return popt
