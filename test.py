from DFXM.scan_functions import *
from DFXM.image_processor import inv_polefigure_colors 


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

    strain = [0, 0.005, 0.008, 0.013, 0.024, 0.035, 0.046, 0.046, 0.046, 0.046]

        # Use Sina stuff to make the colour map
    
    Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, header_chi = process_data(path, com_chi[6])
    Img_phi, maximum_phi, minimum_phi, average_phi, TF_phi, row_size_phi, col_size_phi, header_phi = process_data(path, com_phi[6])
    

    global_max_chi = np.max(scale_image(Img_chi))
    global_max_phi = np.max(scale_image(Img_phi))
    global_min_chi = np.min(scale_image(Img_chi))
    global_min_phi = np.min(scale_image(Img_phi))




    v_fraction, mean_grain, av_neighbours, av_misorientaion, GNDs, ave_GNDs = [], [], [], [], [], []
    i = 0
    for phi_file, chi_file in zip(com_phi, com_chi):
        i += 1
        Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, header_chi = process_data(path, chi_file)
        Img_phi, maximum_phi, minimum_phi, average_phi, TF_phi, row_size_phi, col_size_phi, header_phi = process_data(path, phi_file)

        o_grid = np.array((np.linspace(minimum_chi, maximum_chi, 25), np.linspace(minimum_phi, maximum_phi, 22)), dtype=object)
        test_grid = np.array((np.linspace(minimum_chi, maximum_chi, 2000), np.linspace(minimum_phi, maximum_phi, 2000)), dtype=object)
        colours, colour_data = inv_polefigure_colors(o_grid, test_grid)


        print(maximum_chi, average_chi, row_size_chi, col_size_chi)

        grain = find_grain(TF_chi)
        masked_chi_values, sigma_chi, grain_mask = values_histogram(Img_chi, maximum_chi, grain)
        masked_phi_values, sigma_phi, _ = values_histogram(Img_phi, maximum_phi, grain)

        Img_chi = filter_grain(grain_mask, Img_chi, maximum_chi)
        Img_phi = filter_grain(grain_mask, Img_phi, maximum_phi)

        #scaled_Img_chi = scale_image_global(Img_chi, global_max_chi, global_min_chi)
        #scaled_Img_phi = scale_image_global(Img_phi, global_max_phi, global_min_phi)

        scaled_Img_chi = scale_image(Img_chi)
        scaled_Img_phi = scale_image(Img_phi)

        mosa, Mosa_Img = RGB_image(scaled_Img_chi, scaled_Img_phi)


        KAM = calculate_KAM(col_size_chi, row_size_chi, grain_mask, Img_chi, Img_phi, 2)


        KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask)

        regions, labeled_array = find_regions(skel_KAM)

        filtered_regions = filter_regions(regions, mosa)
        filtered_regions = [prop for prop in filtered_regions if prop.area * pixel_x * pixel_y <= 400]

        neighbours_dict = find_neighbours(filtered_regions, labeled_array)

        #updated_neighbours_dict = process_all_cells_and_neighbors(filtered_regions, labeled_array, neighbours_dict, pixel_x, pixel_y)

        #all_distances = []
        #for neighbours, distances in updated_neighbours_dict.values():
        #    all_distances.extend(distances)

        #plt.figure(figsize=(8, 6))
        #plt.hist(all_distances, bins=30, color='skyblue', edgecolor='black')
        #plt.title('Histogram of Boundary Lengths between Cells and Their Neighbors')
        #plt.xlabel('Boundary Length in Microns')
        #plt.ylabel('Frequency')
        #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        #plt.tight_layout()
        
        
        chi_differences, phi_differences, num_neighbours = neighbour_rotations(filtered_regions, neighbours_dict, Img_chi, Img_phi)

        if num_neighbours > 1000:
            misorientations = neighbour_misorientation(filtered_regions, neighbours_dict, chi_differences, phi_differences)
            fig, ax1 = plt.subplots(figsize=(8, 6))
            fit_and_plot_rayleigh(misorientations, ax1, strain[i-1])

            #plt.hist(misorientations, bins=30, density=True)
            #plt.title('Misorientations')
            
            av_mis = np.mean(misorientations)
            av_misorientaion.append(av_mis)
            #GND_densities = neighbour_GND(misorientations)
        else:
            av_misorientaion.append(0)

        areas, sizes = area_sizes(filtered_regions, pixel_x, pixel_y)

        fig, ax = plt.subplots(figsize=(8, 6)) 

        #fit_and_plot_lognorm(sizes, ax, '0.046')
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])



        vf = volume_fraction(areas, grain_mask, skel_KAM, pixel_x, pixel_y)
        v_fraction.append(vf)

        mean_grain_size = np.mean(sizes)

        GND_density = []
        if num_neighbours > 1000:
            GND_density = neighbour_GND(misorientations, mean_grain_size)
            GNDs.append(GND_density)

            av_GND = np.mean(GND_density)
            ave_GNDs.append(av_GND)
        else:
            ave_GNDs.append(0)

        mean_grain.append(mean_grain_size)

        av_neigh = num_neighbours / len(filtered_regions)
        av_neighbours.append(av_neigh)


        Cell_Img1 = np.ones_like(Mosa_Img)
        for ii in range(1, len(filtered_regions)):  # Skip exterior nr_cells + 1
            cellPixels = filtered_regions[ii].coords
            cell_ave_Chi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 0])
            cell_ave_Phi = np.mean(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 1])
            for row, col in cellPixels:
                Cell_Img1[row, col, 0] = cell_ave_Chi
                Cell_Img1[row, col, 1] = cell_ave_Phi
                Cell_Img1[row, col, 2] = 1  # Set blue channel to 0

        Cell_Img1[skel_KAM] = [0, 0, 0]

        Mosa_Img[skel_KAM] = [0, 0, 0]

        Mosa_Img[~grain_mask] = [1, 1, 1]

        plt.imshow(Mosa_Img)

        # New extent limits for x-axis
        x_min_new = 160
        x_max_new = 780

        # Total extent along the x-axis
        total_extent_x = pixel_x * row_size_chi

        # Image dimensions
        _, image_width, _ = Mosa_Img.shape

        # Calculate new pixel indices for cropping on x-axis
        left = int((x_min_new / total_extent_x) * image_width)
        right = int((x_max_new / total_extent_x) * image_width)

        # Since y-axis remains unchanged, use full height
        top = 0
        bottom = Mosa_Img.shape[0]

        # Crop the image accordingly
        Cell_Img1 = Cell_Img1[top:bottom, left:right]
        Mosa_Img = Mosa_Img[top:bottom, left:right]

        image_dir = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\Documents\\DTU_Project\\data\\Figures\\First_sample_of_PhD\\videos\\cells'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Calculate the center of the image
        center_y, center_x = Mosa_Img.shape[0] // 2, Mosa_Img.shape[1] // 2

        # Define the region of interest (ROI) dimensions
        roi_size_x = 200  # 200x200 pixels
        roi_size_y = 400
        half_roi_size_x = roi_size_x // 2
        half_roi_size_y = roi_size_y // 2

        scale_bar_physical_length = 10  # e.g., 50 μm

        # Calculate the scale bar length in pixels for the main image
        scale_bar_pixel_length_main = scale_bar_physical_length / pixel_x

        # Calculate ROI coordinates
        roi_start_x = max(center_x - half_roi_size_x + 100, 0)
        roi_end_x = min(center_x + half_roi_size_x + 100, Mosa_Img.shape[1])
        roi_start_y = max(center_y - half_roi_size_y, 0)
        roi_end_y = min(center_y + half_roi_size_y, Mosa_Img.shape[0])

        # Calculate the scale bar length in pixels for the zoomed image
        # First, find out how many pixels per micrometer in the zoomed region
        zoom_factor_x = roi_size_x / (roi_end_x - roi_start_x)
        zoom_factor_y = roi_size_y / (roi_end_y - roi_start_y)
        pixels_per_micrometer_zoomed_x = zoom_factor_x / pixel_x
        pixels_per_micrometer_zoomed_y = zoom_factor_y / pixel_y
        scale_bar_pixel_length_zoomed = scale_bar_physical_length * pixels_per_micrometer_zoomed_x

        # Scaling factor for the x-axis
        scale_x = (x_max_new - x_min_new) / Mosa_Img.shape[1]
        # Scaling factor for the y-axis
        scale_y = (pixel_y * col_size_chi) / Mosa_Img.shape[0]

        # Apply scaling to ROI coordinates
        roi_start_x_scaled = roi_start_x * scale_x + x_min_new
        roi_start_y_scaled = roi_start_y * scale_y

        # Adjust ROI size if necessary based on the extent
        roi_size_scaled_x = roi_size_x * scale_x
        roi_size_scaled_y = roi_size_y * scale_y
        

        # Extract the region of interest for the zoomed figure
        zoomed_region = Cell_Img1[roi_start_y:roi_end_y, roi_start_x:roi_end_x]



        # Create a figure with subplots arranged horizontally
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))  # Adjust the figure size as per your needs

        # Display the original image on the first subplot
        ax[0].imshow(Cell_Img1, extent=[x_min_new, x_max_new, 0, pixel_y * col_size_chi])
        # Highlight the ROI with a red rectangle
        rect = patches.Rectangle((roi_start_x_scaled, roi_start_y_scaled), roi_size_scaled_x, roi_size_scaled_y, linewidth=3, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        # Add scale bar
        scale_bar_length = 50
        scale_bar_thickness = 10
        x_position = x_max_new * 0.8
        y_position = pixel_y * col_size_chi * 0.2
        # Draw a larger, lighter rectangle behind the original rectangle to create a shadow/bright outline effect
        rect_shadow = patches.Rectangle((x_position - 2, y_position + 2), scale_bar_length + 4, -(scale_bar_thickness + 4), linewidth=1, edgecolor='white', facecolor='white')
        ax[0].add_patch(rect_shadow)

        # Draw the original rectangle
        rect_scale = patches.Rectangle((x_position, y_position), scale_bar_length, -scale_bar_thickness, linewidth=1, edgecolor='black', facecolor='black')
        ax[0].add_patch(rect_scale)

        # Text shadow or outline effect by drawing the text in a lighter color with slight offsets
        for dx, dy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:  # Offset positions for the shadow
            ax[0].text(x_position + scale_bar_length / 2 + dx, y_position - scale_bar_thickness * 2 + dy, '50 μm', color='white', ha='center', va='top', fontsize=25, fontweight='bold')

        # Draw the original text
        ax[0].text(x_position + scale_bar_length / 2, y_position - scale_bar_thickness * 2, '50 μm', color='black', ha='center', va='top', fontsize=25, fontweight='bold')
        ax[0].axis('off')  # Turn off axis for a cleaner look


        # Display the zoomed-in region on the second subplot
        aspect_ratio = pixel_y / pixel_x
        ax[1].imshow(zoomed_region, aspect=aspect_ratio)  # Set the custom aspect ratio
        # Add a red border around the zoomed-in image by creating a red rectangle that frames the subplot
        frame = patches.Rectangle((-0.01, -0.01), 1.02, 1.02, transform=ax[1].transAxes, linewidth=6, edgecolor='r', facecolor='none', clip_on=False)
        scale_bar_position_zoomed = (zoomed_region.shape[1] - 25, zoomed_region.shape[0] - 50)
        rect_scale_zoomed = patches.Rectangle(scale_bar_position_zoomed, scale_bar_pixel_length_zoomed, scale_bar_thickness,  
                                      linewidth=1, edgecolor='black', facecolor='black')
        ax[1].add_patch(rect_scale_zoomed)

        # Optionally, add label for the scale bar on the zoomed image
        ax[1].text(scale_bar_position_zoomed[0] + scale_bar_pixel_length_zoomed / 2, scale_bar_position_zoomed[1] + scale_bar_thickness * 2, 
            f'{scale_bar_physical_length} μm', color='black', ha='center', va='top', fontsize=25, fontweight='bold')
        ax[1].add_patch(frame)
        ax[1].axis('off')

        # Adjust spacing between the subplots to ensure they are visually appealing
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.suptitle('$\epsilon$ = ' + str(strain[i-1]), fontsize=30, fontweight='bold', y=0.85)
        # Save the figure with both the original image with ROI highlighted and the zoomed-in view, both framed in red
        plt.savefig(os.path.join(image_dir, f'Cell_Image_{i}.png'), dpi=300) 
        plt.close()




        ## SECOND IMAGE ##




        image_dir = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\Documents\\DTU_Project\\data\\Figures\\First_sample_of_PhD\\videos\\mosa'

        zoomed_region = Mosa_Img[roi_start_y:roi_end_y, roi_start_x:roi_end_x]


                # Create a figure with subplots arranged horizontally
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))  # Adjust the figure size as per your needs

        # Display the original image on the first subplot
        ax[0].imshow(Mosa_Img, extent=[x_min_new, x_max_new, 0, pixel_y * col_size_chi])
        # Highlight the ROI with a red rectangle
        rect = patches.Rectangle((roi_start_x_scaled, roi_start_y_scaled), roi_size_scaled_x, roi_size_scaled_y, linewidth=3, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        # Add scale bar
        scale_bar_length = 50
        scale_bar_thickness = 10
        x_position = x_max_new * 0.8
        y_position = pixel_y * col_size_chi * 0.2
        # Draw a larger, lighter rectangle behind the original rectangle to create a shadow/bright outline effect
        rect_shadow = patches.Rectangle((x_position - 2, y_position + 2), scale_bar_length + 4, -(scale_bar_thickness + 4), linewidth=1, edgecolor='white', facecolor='white')
        ax[0].add_patch(rect_shadow)

        # Draw the original rectangle
        rect_scale = patches.Rectangle((x_position, y_position), scale_bar_length, -scale_bar_thickness, linewidth=1, edgecolor='black', facecolor='black')
        ax[0].add_patch(rect_scale)

        # Text shadow or outline effect by drawing the text in a lighter color with slight offsets
        for dx, dy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:  # Offset positions for the shadow
            ax[0].text(x_position + scale_bar_length / 2 + dx, y_position - scale_bar_thickness * 2 + dy, '50 μm', color='white', ha='center', va='top', fontsize=25, fontweight='bold')

        # Draw the original text
        ax[0].text(x_position + scale_bar_length / 2, y_position - scale_bar_thickness * 2, '50 μm', color='black', ha='center', va='top', fontsize=25, fontweight='bold')
        ax[0].axis('off')  # Turn off axis for a cleaner look

        # Create an inset axis in the bottom left corner for the scatter plot
        inset_ax = inset_axes(ax[0], width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax[0].transAxes)

        # Plot the scatter plot in the inset axis
        inset_ax.scatter(colour_data.T[0], colour_data.T[1], c=colours, s=65, marker=',')
        inset_ax.set_xlabel('$\chi$ (°)', fontsize=16)
        inset_ax.set_ylabel('$\phi$ (°)', fontsize=16)
    
        # Optionally, adjust the appearance of the inset_ax as needed, e.g., remove axis labels or set limits
        inset_ax.axis('on')  # for a cleaner look without axis labels

        # Display the zoomed-in region on the second subplot
        aspect_ratio = pixel_y / pixel_x
        ax[1].imshow(zoomed_region, aspect=aspect_ratio)  # Set the custom aspect ratio
        # Add a red border around the zoomed-in image by creating a red rectangle that frames the subplot
        frame = patches.Rectangle((-0.01, -0.01), 1.02, 1.02, transform=ax[1].transAxes, linewidth=6, edgecolor='r', facecolor='none', clip_on=False)
        scale_bar_position_zoomed = (zoomed_region.shape[1] - 25, zoomed_region.shape[0] - 50)
        rect_shadow_zoomed = patches.Rectangle((scale_bar_position_zoomed[0] - (2*aspect_ratio), scale_bar_position_zoomed[1] + 10.5), scale_bar_pixel_length_zoomed + 4*aspect_ratio, -(scale_bar_thickness + 4*aspect_ratio), linewidth=1, edgecolor='white', facecolor='white')
        rect_scale_zoomed = patches.Rectangle(scale_bar_position_zoomed, scale_bar_pixel_length_zoomed, scale_bar_thickness,  
                                      linewidth=1, edgecolor='black', facecolor='black')
        ax[1].add_patch(rect_shadow_zoomed)
        ax[1].add_patch(rect_scale_zoomed)

        for dx, dy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:  # Offset positions for the shadow
            ax[1].text(scale_bar_position_zoomed[0] + scale_bar_pixel_length_zoomed / 2 + (dx*aspect_ratio), scale_bar_position_zoomed[1] + scale_bar_thickness * 2 + dy*aspect_ratio, 
                f'{scale_bar_physical_length} μm', color='white', ha='center', va='top', fontsize=25, fontweight='bold')
            
        # Optionally, add label for the scale bar on the zoomed image
        ax[1].text(scale_bar_position_zoomed[0] + scale_bar_pixel_length_zoomed / 2, scale_bar_position_zoomed[1] + scale_bar_thickness * 2, 
            f'{scale_bar_physical_length} μm', color='black', ha='center', va='top', fontsize=25, fontweight='bold')
        ax[1].add_patch(frame)
        ax[1].axis('off')
        # Adjust spacing between the subplots to ensure they are visually appealing
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.suptitle('$\epsilon$ = ' + str(strain[i-1]), fontsize=30, fontweight='bold', y=0.85)
        # Save the figure with both the original image with ROI highlighted and the zoomed-in view, both framed in red
        plt.savefig(os.path.join(image_dir, f'Mosa_Image_{i}.png'), dpi=300) 
        plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(strain, v_fraction, 'o-', color='black')
    ax1.set_xlabel('Strain')
    ax1.set_ylabel('Volume Fraction')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(strain, mean_grain, 'o-', color='red')
    ax2.set_ylabel('Mean Grain Size')

    ax1.legend(['Volume Fraction'], loc='upper left', labelcolor='black')
    ax2.legend(['Mean Grain Size'], loc='upper right', labelcolor='red')

    fig.tight_layout()

    ax2.tick_params(axis='y', labelcolor='black')

    ax1.set_label('Volume Fraction')
    ax2.set_label('Mean Grain Size')

    fig.tight_layout()
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.plot(strain, av_neighbours, 'o-')
    plt.xlabel('Strain')
    plt.ylabel('Average Number of Neighbours')
    plt.title('Average Number of Neighbours vs Strain')

    plt.figure(figsize=(8, 6))
    plt.plot(strain, av_misorientaion, 'o-')
    plt.xlabel('Strain')
    plt.ylabel('Average Misorientation')
    plt.title('Average Misorientation vs Strain')

    plt.figure(figsize=(8, 6))
    plt.plot(strain, ave_GNDs, 'o-')
    plt.xlabel('Strain')
    plt.ylabel('Average GND Density')
    plt.title('Average GND Density vs Strain')
    plt.show()
                                  
if __name__ == '__main__':
    main()