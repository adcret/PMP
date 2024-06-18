from DFXM.scan_functions import *
from DFXM.image_processor import inv_polefigure_colors 
import matplotlib.patheffects as path_effects


path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\APRIL2024\\NoFitting\\'
#path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\fit15\\'
#pixel_y = 0.203; pixel_x = 0.6575; # effective pixel sizes in mu
pixel_y = 0.066; pixel_x = 0.15


com_phi, com_chi = load_data(path, 'com')
fwhm_phi, fwhm_chi = load_data(path, 'fwhm')

strain = [0, 0.005, 0.008, 0.013, 0.024, 0.035, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046]

com_phi = sorted(com_phi, key=extract_number)
com_chi = sorted(com_chi, key=extract_number)
fwhm_phi = sorted(fwhm_phi, key=extract_number)
fwhm_chi = sorted(fwhm_chi, key=extract_number)
print(com_phi)


j = 4 # index of the file

i = 1 # index of the strain
 
# load the COM images
Img_chi, maximum_chi, minimum_chi, average_chi, TF_chi, row_size_chi, col_size_chi, _ = process_data(path, com_chi[j], method='COM')
Img_phi, maximum_phi, minimum_phi, _, _, _, _, _ = process_data(path, com_phi[j], method='COM')

# Define the colour grid for the mosaicity map
test_grid = np.array((np.linspace(minimum_chi, maximum_chi, 25), np.linspace(minimum_phi, maximum_phi, 22)), dtype=object)
o_grid = np.array((np.linspace(minimum_chi, maximum_chi, 500), np.linspace(minimum_phi, maximum_phi, 500)), dtype=object)
colours, colour_data = inv_polefigure_colors(o_grid, test_grid)

# Create the mosaicity map
grain = find_grain(TF_chi)
_, _, grain_mask = values_histogram(Img_chi, maximum_chi, grain)

Img_chi = filter_grain(grain_mask, Img_chi, maximum_chi)
Img_phi = filter_grain(grain_mask, Img_phi, maximum_phi)

scaled_Img_chi = scale_image(Img_chi)
scaled_Img_phi = scale_image(Img_phi)

mosa, Mosa_Img = RGB_image(scaled_Img_chi, scaled_Img_phi)



#KAM = calculate_KAM(col_size_chi, row_size_chi, grain_mask, Img_chi, Img_phi, 5)

#KAM_mask, skel_KAM = KAM_refine(KAM, grain_mask)

#skel_KAM_dilated = binary_dilation(skel_KAM, disk(1))

scale_bar_length = 20
scale_bar_thickness = 5
x_position = 280
y_position = pixel_y * col_size_chi * 0.2
aspect_ratio = pixel_y / pixel_x


Mosa_Img[~grain_mask] = [1, 1, 1]

plt.figure()
plt.imshow(Mosa_Img, extent=[0, pixel_x * row_size_chi, 0, pixel_y * col_size_chi])
#plt.show()

mosa = np.copy(Mosa_Img)

Mosa = np.copy(Mosa_Img)

#Load the FWHM data

FWHM_chi, maximum_chi, minimum_chi, _, _, _, _, _ = process_data(path, fwhm_chi[j], method='FWHM', grain_mask=grain_mask)
FWHM_phi, maximum_phi, minimum_phi, _, _, _, _, _ = process_data(path, fwhm_phi[j], method='FWHM', grain_mask=grain_mask)

# Combine the Phi and Chi contributions to the FWHM
FWHM_img = np.sqrt(FWHM_chi**2 + FWHM_phi**2) 
#FWHM_img[FWHM_img > 6] = 4

FWHM_img2 = FWHM_img


#FWHM_filter = calculate_FWHM(FWHM_img, grain_mask, row_size_chi, col_size_chi, 0.195, 2)

#mask_FWHM, skel_FHWM = FWHM_mask(FWHM_filter, grain_mask)


image_dir = r'C:\Users\adacre\OneDrive - Danmarks Tekniske Universitet\Documents\DTU_Project\data\Figures\First_sample_of_PhD\Fig_paper'
#plt.savefig(os.path.join(image_dir, f'PB_{j}.png'), dpi=600)

#FWHM_img[skel_KAM] = 10

plt.figure()
plt.imshow(FWHM_img, cmap='jet_r', extent=[0, pixel_x * row_size_chi, 0, pixel_y * col_size_chi], vmin=0, vmax =2.2)

Mosa[:,0:230] = [1,1,1]

# Initialize figure and GridSpec
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(1, 1)
# Axes for the original image
ax0 = fig.add_subplot(gs[:, 0]) 
ax0.imshow(Mosa, extent=[0, pixel_x * row_size_chi, 0, pixel_y * col_size_chi])


# Scale bar and its shadow
rect_shadow = patches.Rectangle((x_position - 2, y_position + 2), scale_bar_length + 4, -(scale_bar_thickness + 4), linewidth=1, edgecolor='white', facecolor='white')
ax0.add_patch(rect_shadow)
rect_scale = patches.Rectangle((x_position, y_position), scale_bar_length, -scale_bar_thickness, linewidth=1, edgecolor='black', facecolor='black')
ax0.add_patch(rect_scale)

# Add text shadow or outline effect for scale bar label
text = ax0.text(x_position + scale_bar_length / 2, y_position - scale_bar_thickness * 2, '20 μm', color='black', ha='center', va='top', fontsize=25, fontweight='bold')
text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
ax0.axis('off')

# Inset axis for the scatter plot
inset_ax = inset_axes(ax0, width="15%", height="20%", loc='lower left', bbox_to_anchor=(0.11, -0.01, 1, 1), bbox_transform=ax0.transAxes)
inset_ax.scatter(colour_data.T[0], colour_data.T[1], c=colours, s=65, marker=',')
inset_ax.set_xlabel('$\chi$ (°)', fontsize=24)
#inset_ax.set_xticks([-0.5, 0, 0.5])
inset_ax.set_ylabel('$\phi$ (°)', fontsize=24)
inset_ax.axis('on')
inset_ax.tick_params(labelsize=14)
plt.suptitle('$\epsilon$ = 5.30%', fontsize=36, fontweight='bold', y=0.87)
plt.tight_layout()
# Save the figure
image_dir = r'C:\Users\adacre\OneDrive - Danmarks Tekniske Universitet\Documents\DTU_Project\data\Figures\First_sample_of_PhD\Fig_paper'
plt.savefig(os.path.join(image_dir, f'GNBs.png'), dpi=600)


plt.show()


