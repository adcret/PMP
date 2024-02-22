from DFXM.scan_functions import *
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage import exposure

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

    v_fraction, mean_grain, av_neighbours, av_misorientaion = [], [], [], []
    i = 0

    grain = find_grain(TF_chi)
    _, _, grain_mask = values_histogram(Img_chi, maximum_chi, grain)

    scaled_img_chi = scale_image(Img_chi)
    scaled_img_phi = scale_image(Img_phi)

    _, Mosa_Img = RGB_image(scaled_img_chi, scaled_img_phi)

    slic =  segmentation.slic(Mosa_Img, n_segments=3500, mask = grain_mask, compactness=15, sigma=0)

    KAM = calculate_KAM(col_size_chi, row_size_chi, grain_mask, Img_chi, Img_phi, 2)


    KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask)

    regions, labeled_array = find_regions(skel_KAM)

    Mosa_2 = Mosa_Img.copy()
    Mosa_2[skel_KAM] = [0, 0, 0]
    Mosa_2[~grain_mask] = [1, 1, 1]
    Mosa_Img[~grain_mask] = [1, 1, 1]

    # Read in FWHM data for Ch
    chi_fwhm_file = fabio.open(path + fwhm_phi[6])
    A = chi_fwhm_file.data
    B1 = A.T
    B = np.flipud(B1)
    B[~grain_mask] = 2  # Apply grain mask and set outside values to 2
    FWHM_Chi = np.minimum(np.abs(B), 2)

    # Read in FWHM data for Phi
    phi_fwhm_file = fabio.open(path + fwhm_chi[6])
    A = phi_fwhm_file.data
    B1 = A.T
    B = np.flipud(B1)
    B[~grain_mask] = 2  # Apply grain mask and set outside values to 2
    FWHM_Phi = np.minimum(np.abs(B), 2)

    FWHM_filter_chi = calculate_FWHM(FWHM_Chi, grain_mask, row_size_chi, col_size_chi, 0.11, 2)
    FWHM_filter_phi = calculate_FWHM(FWHM_Phi, grain_mask, row_size_chi, col_size_chi, 0.11, 2)

    FWHM_masks, skel_FWHM = FWHM_mask(FWHM_filter_chi, grain_mask)
    FWHM_masks_phi, skel_FWHM_phi = FWHM_mask(FWHM_filter_phi, grain_mask)

    edges = feature.canny(color.rgb2gray(Mosa_Img), sigma=1.0, low_threshold=0.4, high_threshold=0.9)

    hist = exposure.histogram(FWHM_Chi, nbins=256)
    hist2 = exposure.histogram(FWHM_Phi, nbins=256)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)  
    ax4, ax5, ax6 = ax.ravel()
    #ax1.imshow(color.rgb2gray(Mosa_Img), extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi], cmap='gray')
    #ax2.imshow(edges, extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi])
    #ax3.imshow(Mosa_Img, extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi])
    ax4.set_title('Cells on mosaicity map')
    ax4.imshow(Mosa_2, extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi])
    ax5.set_title('FWHM Chi')
    ax5.imshow(FWHM_Chi, extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi])
    ax6.set_title('FWHM Phi')
    ax6.imshow(FWHM_Phi, extent=[0, pixel_x*row_size_chi, 0, pixel_y * col_size_chi])

    

    for ax in ax.ravel():
        ax.set_axis_off()

    plt.tight_layout()

    plt.figure()
    plt.plot(hist[1], hist[0], label='Chi')
    plt.plot(hist2[1], hist2[0], label='Phi')
    plt.xlabel('FWHM pixel value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()