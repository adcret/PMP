import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, util
from scipy import ndimage as ndi
import numpy as np
import os
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_dog, blob_log, blob_doh
from tqdm import tqdm

# Load the images in grayscale
path = 'C:\\Users\\adacre\\OneDrive - Danmarks Tekniske Universitet\\Documents\\DTU_Project\\data\\Raw_images_fit15'
images = [os.path.join(path, img) for img in os.listdir(path) if img.endswith('.png')]

mean_sizes = []
median_sizes = []
max_sizes = []
min_sizes = []

mean_sizes_log = []
mean_sizes_dog = []
mean_sizes_doh = []

for i in range(len(images)):
    image = io.imread(images[i])

    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        image = rgb2gray(image)

    # Noise reduction
    image_smoothed = filters.gaussian(image, sigma=1)

    # Thresholding
    thresh = filters.threshold_otsu(image_smoothed)
    binary = image_smoothed > thresh

    # Morphological operations to close gaps and open touching objects
    binary_closed = morphology.closing(binary, morphology.square(3))
    binary_opened = morphology.opening(binary_closed, morphology.square(1))

    label_objects, nb_labels = ndi.label(~binary_opened)
    sizes = np.bincount(label_objects.ravel())
    # Filter out size 0 which is the background
    sizes = sizes[1:]

    print("Number of distinct regions in binary_opened:", nb_labels)
    print("Sizes of the detected regions:", sizes)


    image_binary = ~binary

    # label the regions in image_binary
    labels, _ = ndi.label(image_binary)

    inv_image_smoothed = util.invert(image_smoothed)

    blob_Log = blob_log(inv_image_smoothed, max_sigma=50, min_sigma=3, threshold=0.08)
    blob_Dog = blob_dog(inv_image_smoothed, max_sigma=50, min_sigma=3, threshold=0.08)
    blob_Doh = blob_doh(inv_image_smoothed, max_sigma=50, min_sigma=2, threshold=0.008)

    mean_size_log = np.mean(blob_Log[:, 2])*np.sqrt(2)
    mean_size_dog = np.mean(blob_Dog[:, 2])*np.sqrt(2)
    mean_size_doh = np.mean(blob_Doh[:, 2])*np.sqrt(2)

    mean_sizes_log.append(mean_size_log)
    mean_sizes_dog.append(mean_size_dog)
    mean_sizes_doh.append(mean_size_doh)

    # Plotting size distribution for each method
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].hist(blob_Log[:, 2]*np.sqrt(2), bins=20, color='blue', alpha=0.7)
    ax[0].set_title('Size Distribution for LOG')
    ax[0].set_xlabel('Size')
    ax[0].set_ylabel('Frequency')

    ax[1].hist(blob_Dog[:, 2]*np.sqrt(2), bins=20, color='green', alpha=0.7)
    ax[1].set_title('Size Distribution for DOG')
    ax[1].set_xlabel('Size')
    ax[1].set_ylabel('Frequency')



    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image_smoothed, cmap=plt.cm.gray)
    ax[0].set_title('laplacian of gaussian')
    for blob in blob_Log:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax[0].add_patch(c)
    
    ax[1].imshow(image_smoothed, cmap=plt.cm.gray)
    ax[1].set_title('difference of gaussian')
    for blob in blob_Dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax[1].add_patch(c)
    
    for a in ax:
        a.set_axis_off()

    plt.tight_layout()


    distance = ndi.distance_transform_edt(image_binary)
    coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=labels, num_peaks_per_label=4)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    watershed_img = watershed(-distance, markers, mask=image_binary, watershed_line=True)
    watershed_labels = measure.label(watershed_img)
    binary_regions = np.where(watershed_labels > 0, 0, 1)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image_smoothed, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(binary_regions, cmap=plt.cm.gray, interpolation='nearest')
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    

    # After watershed segmentation
    watershed_labels = measure.label(watershed_img)

    region_props = measure.regionprops(watershed_labels)

    # Extract the area (size) of each region
    sizes = [prop.area for prop in region_props]

    mean_size = np.mean(sizes) 
    median_size = np.median(sizes)
    max_size = np.max(sizes)
    min_size = np.min(sizes)

    if mean_size < 1500:
        mean_sizes.append(mean_size)
    median_sizes.append(median_size)
    max_sizes.append(max_size)
    min_sizes.append(min_size)

    print(f"Statistics for Image {i+1}:")
    print(f"Mean size of regions: {mean_size}")
    print(f"Median size of regions: {median_size}")
    print(f"Maximum size of regions: {max_size}")
    print(f"Minimum size of regions: {min_size}")

    plt.figure()
    plt.hist(sizes, bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of Feature Sizes for Image {i+1}')
    plt.xlabel('Size of Feature')
    plt.ylabel('Count')
    plt.grid(True)


plt.figure()
plt.plot(median_sizes, label='Mean Size')
plt.xlabel('Image Number')
plt.ylabel('Median Size in Pixels')

plt.figure()
plt.plot(mean_sizes_dog, label='Mean Size DoG')
plt.plot(mean_sizes_log, label='Mean Size LoG')
plt.plot(mean_sizes_doh, label='Mean Size DoH')
plt.xlabel('Image Number')
plt.ylabel('Mean Size in Pixels')
plt.legend()
plt.show()
