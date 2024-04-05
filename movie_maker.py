import imageio.v2 as imageio
import numpy as np
import os

def create_transition(start_img, end_img, steps=2):
    """
    Creates a smooth transition between two images over a given number of steps.
    """
    transition_images = []
    for step in range(1, steps + 1):
        alpha = step / (steps + 1)
        blended_img = (1 - alpha) * start_img + alpha * end_img
        transition_images.append(blended_img.astype(np.uint8))
    return transition_images

image_dir = r'C:\Users\adacre\OneDrive - Danmarks Tekniske Universitet\Documents\DTU_Project\data\Figures\First_sample_of_PhD\videos\cells'
image_files = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) if img.endswith('.png')]
output_video_path = os.path.join(image_dir, 'output_video_with_transitions.mp4')

fps = 30  # Frames per second
image_frames = 60  # Number of frames to display each image
transition_frames = 15  # Number of frames for the transition

writer = imageio.get_writer(output_video_path, fps=fps, format='FFMPEG')

for i in range(len(image_files) - 1):
    start_img = imageio.imread(image_files[i])
    end_img = imageio.imread(image_files[i + 1])
    
    # Append the start image for its duration
    for _ in range(image_frames):
        writer.append_data(start_img)
    
    # Generate and append transition frames
    for transition_img in create_transition(start_img, end_img, transition_frames):
        writer.append_data(transition_img)

# For the last image, display it for the full duration without a transition
last_img = imageio.imread(image_files[-1])
for _ in range(image_frames):
    writer.append_data(last_img)

writer.close()
