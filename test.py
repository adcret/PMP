from DFXM.scan_functions import *

def main():
    path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
    pixel_y = 0.203; pixel_x = 0.6575; # effective pixel sizes in mu

    com_phi, com_chi = load_data(path, 'com')
    fwhm_phi, fwhm_chi = load_data(path, 'fwhm')

    print('com_phi', com_phi)

    com_phi = sorted(com_phi, key=extract_number)
    com_chi = sorted(com_chi, key=extract_number)
    fwhm_phi = sorted(fwhm_phi, key=extract_number)
    fwhm_chi = sorted(fwhm_chi, key=extract_number)

    print('com_phi', com_phi)

    Img_chi, maximum_chi, average_chi, TF_chi, row_size_chi, col_size_chi = process_data(path, com_chi[6])
    Img_phi, maximum_phi, average_phi, TF_phi, row_size_phi, col_size_phi = process_data(path, com_phi[6])


    print(maximum_chi, average_chi, row_size_chi, col_size_chi)

    grain = find_grain(TF_chi)
    masked_chi_values, sigma_chi, grain_mask = values_histogram(Img_chi, maximum_chi, grain)
    masked_phi_values, sigma_phi, _ = values_histogram(Img_phi, maximum_phi, grain)

    Img_chi = filter_grain(grain_mask, Img_chi, maximum_chi)
    Img_phi = filter_grain(grain_mask, Img_phi, maximum_phi)

    scaled_Img_chi = scale_image(Img_chi)
    scaled_Img_phi = scale_image(Img_phi)

    mosa, Mosa_Img = RGB_image(scaled_Img_chi, scaled_Img_phi)


    KAM = calculate_KAM(col_size_chi, row_size_chi, grain_mask, Img_chi, Img_phi, 2)


    KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask, 0.041)

    regions, labeled_array = find_regions(skel_KAM)

    filtered_regions = filter_regions(regions, mosa)

    neighbours_dict = find_neighbours(filtered_regions, labeled_array)
    
    Cell_Img1 = np.ones_like(mosa)
    for ii in range(1, len(filtered_regions)):  # Skip exterior nr_cells + 1
        cellPixels = filtered_regions[ii].coords
        cell_ave_Chi = np.mean(mosa[cellPixels[:, 0], cellPixels[:, 1], 0])
        cell_ave_Phi = np.mean(mosa[cellPixels[:, 0], cellPixels[:, 1], 1])
        for row, col in cellPixels:
            Cell_Img1[row, col, 0] = cell_ave_Chi
            Cell_Img1[row, col, 1] = cell_ave_Phi
            Cell_Img1[row, col, 2] = 0  # Set blue channel to 0

    Cell_Img1[skel_KAM] = [0, 0, 0]

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
        display_region = (max(int(center_y) - 100, 0), min(int(center_y) + 100, cell_mask.shape[0]),
                        max(int(center_x) - 100, 0), min(int(center_x) + 100, cell_mask.shape[1]))
        
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

    chi_differences, phi_differences, num_neighbours = neighbour_rotations(filtered_regions, neighbours_dict, Img_chi, Img_phi)
    misorientations = neighbour_misorientation(filtered_regions, neighbours_dict, chi_differences, phi_differences)

    GND_densities = neighbour_GND(misorientations)

    plt.figure()
    plt.hist(misorientations, bins=20)
    plt.show()

    plt.figure()
    plt.hist(GND_densities, bins=20)
    plt.show()

                                  
if __name__ == '__main__':
    main()