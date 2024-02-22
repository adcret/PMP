# Make image with KAM overlay 

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

Mosa_Img[skel_KAM] = [0, 0, 0]

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
#Cell_Img1 = Cell_Img1[top:bottom, left:right]
Mosa_Img = Mosa_Img[top:bottom, left:right]

image_dir = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\Documents\\DTU_Project\\data\\Figures\\First_sample_of_PhD\\videos\\cells_mosa_newKAM'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Inside your loop, save each figure with a unique name
plt.figure(figsize=(19.2, 10.8))  # This size is in inches. DPI will affect the final pixel size.
plt.imshow(Mosa_Img, extent=[x_min_new, x_max_new, 0, pixel_y * col_size_chi])
plt.axis('off')
scale_bar_length = 50  
scale_bar_thickness = 10  
x_position = x_max_new * 0.95  
y_position = pixel_y * col_size_chi * 0.05  

rect = patches.Rectangle((x_position, y_position), scale_bar_length, -scale_bar_thickness, linewidth=1, edgecolor='black', facecolor='black')
plt.gca().add_patch(rect)

# Add scale bar label
plt.text(x_position + scale_bar_length / 2, y_position - scale_bar_thickness * 2, '50 μm', color='black', ha='center', va='top', fontsize=14, fontweight='bold')

plt.savefig(os.path.join(image_dir, f'Mosa_Image_{i}.png'), dpi=100)  # Adjust DPI to get close to 1080p
plt.close()