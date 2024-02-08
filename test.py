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
    i = 0
    for phi_file, chi_file in zip(com_phi, com_chi):
        i += 1
        Img_chi, maximum_chi, average_chi, TF_chi, row_size_chi, col_size_chi = process_data(path, chi_file)
        Img_phi, maximum_phi, average_phi, TF_phi, row_size_phi, col_size_phi = process_data(path, phi_file)


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


        KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask)

        regions, labeled_array = find_regions(skel_KAM)

        filtered_regions = filter_regions(regions, mosa)
        filtered_regions = [prop for prop in filtered_regions if prop.area * pixel_x * pixel_y <= 400]

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

        # New extent limits for x-axis
        x_min_new = 200
        x_max_new = 780

        # Total extent along the x-axis
        total_extent_x = pixel_x * row_size_chi

        # Image dimensions
        _, image_width, _ = Cell_Img1.shape

        # Calculate new pixel indices for cropping on x-axis
        left = int((x_min_new / total_extent_x) * image_width)
        right = int((x_max_new / total_extent_x) * image_width)

        # Since y-axis remains unchanged, use full height
        top = 0
        bottom = Cell_Img1.shape[0]

        # Crop the image accordingly
        Cell_Img1 = Cell_Img1[top:bottom, left:right]

        image_dir = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\Documents\\DTU_Project\\data\\Figures\\First_sample_of_PhD\\videos\\cells'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Inside your loop, save each figure with a unique name
        plt.figure(figsize=(19.2, 10.8))  # This size is in inches. DPI will affect the final pixel size.
        plt.imshow(Cell_Img1, extent=[x_min_new, x_max_new, 0, pixel_y * col_size_chi])
        plt.axis('off')
        scale_bar_length = 50  
        scale_bar_thickness = 10  
        x_position = x_max_new * 0.95  
        y_position = pixel_y * col_size_chi * 0.05  

        rect = patches.Rectangle((x_position, y_position), scale_bar_length, -scale_bar_thickness, linewidth=1, edgecolor='black', facecolor='black')
        plt.gca().add_patch(rect)

        # Add scale bar label
        plt.text(x_position + scale_bar_length / 2, y_position - scale_bar_thickness * 2, '50 μm', color='black', ha='center', va='top', fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(image_dir, f'Cell_Averaged_Image_{i}.png'), dpi=100)  # Adjust DPI to get close to 1080p
        plt.close()

        chi_differences, phi_differences, num_neighbours = neighbour_rotations(filtered_regions, neighbours_dict, Img_chi, Img_phi)

        if num_neighbours > 1000:
            misorientations = neighbour_misorientation(filtered_regions, neighbours_dict, chi_differences, phi_differences)

            plt.hist(misorientations, bins=30, density=True)
            plt.title('Misorientations')
            

        #GND_densities = neighbour_GND(misorientations)

        areas, sizes = area_sizes(filtered_regions, pixel_x, pixel_y)

        #fig, ax = plt.subplots(figsize=(8, 6)) 

        #fit_and_plot_lognorm(sizes, ax, '0.046')
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        vf = volume_fraction(areas, grain_mask, skel_KAM, pixel_x, pixel_y)

    image_files = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) if img.endswith('.png')]

    # Output video file path
    output_video_path = os.path.join(image_dir, 'output_video.mp4')

    # Create a video from the images
    writer = imageio.get_writer(output_video_path, fps=1)  # Adjust fps as needed

    for img_path in image_files:
        image = imageio.imread(img_path)
        writer.append_data(image)

    writer.close()
                                  
if __name__ == '__main__':
    main()