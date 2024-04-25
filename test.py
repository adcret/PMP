from DFXM.scan_functions import *
from DFXM.image_processor import inv_polefigure_colors 


def main():
    path = r'C:\Users\adacre\OneDrive - Danmarks Tekniske Universitet\Documents\MEGA\Rocking_curves_AprilBeamtime'
    pixel_y = 0.069; pixel_x = 0.157; # effective pixel sizes in mu

    com_phi, com_chi = load_data(path, 'com')
    fwhm_phi, fwhm_chi = load_data(path, 'fwhm')

    print('com_phi', com_phi)

    com_phi = sorted(com_phi, key=extract_number)
    com_chi = sorted(com_chi, key=extract_number)
    fwhm_phi = sorted(fwhm_phi, key=extract_number)
    fwhm_chi = sorted(fwhm_chi, key=extract_number)

    print('com_phi', com_phi)

    #strain = [0, 0.005, 0.008, 0.013, 0.024, 0.035, 0.046, 0.046, 0.046, 0.046]
    strain = [4.6]

        # Use Sina stuff to make the colour map
    
    Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, header_chi = process_data(path, com_chi[0], method='COM')
    Img_phi, maximum_phi, minimum_phi, average_phi, TF_phi, row_size_phi, col_size_phi, header_phi = process_data(path, com_phi[0], method='COM')
    

    global_max_chi = np.max(scale_image(Img_chi))
    global_max_phi = np.max(scale_image(Img_phi))
    global_min_chi = np.min(scale_image(Img_chi))
    global_min_phi = np.min(scale_image(Img_phi))

    v_fraction, mean_grain, av_neighbours, av_misorientaion, GNDs, ave_GNDs = [], [], [], [], [], []
    i = 0
    for i in range(len(com_chi)):
        Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, header_chi = process_data(path, com_phi[i], method='COM')
        Img_phi, maximum_phi, minimum_phi, average_phi, TF_phi, row_size_phi, col_size_phi, header_phi = process_data(path, com_chi[i], method='COM')

        o_grid = np.array((np.linspace(minimum_chi, maximum_chi, 41), np.linspace(minimum_phi, maximum_phi, 14)), dtype=object)
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


        KAM = calculate_KAM(col_size_chi, row_size_chi, grain_mask, Img_chi, Img_phi, 10)

        KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask)

        regions, labeled_array = find_regions(skel_KAM)

        filtered_regions = filter_regions(regions, mosa)
        filtered_regions = [prop for prop in filtered_regions if prop.area * pixel_x * pixel_y <= 10000]

        areas, sizes = area_sizes(filtered_regions, pixel_x, pixel_y)

        Cell_Img1 = np.ones_like(Mosa_Img)
        for ii in range(1, len(filtered_regions)):  # Skip exterior nr_cells + 1
            cellPixels = filtered_regions[ii].coords
            cell_ave_Chi = np.median(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 0])
            cell_ave_Phi = np.median(Mosa_Img[cellPixels[:, 0], cellPixels[:, 1], 1])
            for row, col in cellPixels:
                Cell_Img1[row, col, 0] = cell_ave_Chi
                Cell_Img1[row, col, 1] = cell_ave_Phi
                Cell_Img1[row, col, 2] = .75  # Set blue channel to 0.75 (as defined in the colour scale)

        Cell_Img1[skel_KAM] = [0, 0, 0]

        Mosa_Img[skel_KAM] = [0, 0, 0]

        Mosa_Img[~grain_mask] = [1, 1, 1]

        plt.figure()
        plt.imshow(Mosa_Img, extent=[0, col_size_chi * pixel_x, 0, row_size_chi * pixel_y])

        plt.figure()
        plt.imshow(KAM_mask, extent=[0, col_size_chi * pixel_x, 0, row_size_chi * pixel_y])

        FWHM_chi, maximum_chi, minimum_chi, _, _, _, _, _ = process_data(path, fwhm_chi[i], method='FWHM', grain_mask=grain_mask)
        FWHM_phi, maximum_phi, minimum_phi, _, _, _, _, _ = process_data(path, fwhm_phi[i], method='FWHM', grain_mask=grain_mask)

        # Combine the Phi and Chi contributions to the FWHM
        FWHM_img = np.sqrt(FWHM_chi**2 + FWHM_phi**2) 
        FWHM_img = FWHM_chi + FWHM_phi
        FWHM_img[FWHM_img > 4] = 4

        #FWHM_img = np.where(skel_KAM, 2.5, FWHM_img)
        plt.figure()
        plt.imshow(FWHM_img, cmap='jet_r',extent=[0, col_size_chi * pixel_x, 0, row_size_chi * pixel_y], vmin=0, vmax=2)
        plt.colorbar(label='FWHM (°)')

        plt.figure()
        plt.imshow(Cell_Img1, extent=[0, col_size_chi * pixel_x, 0, row_size_chi * pixel_y])
        plt.show()
                                  
if __name__ == '__main__':
    main()