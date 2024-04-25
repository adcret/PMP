import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\adacre\OneDrive - Danmarks Tekniske Universitet\Documents\MEGA\Rocking_curves_AprilBeamtime\100_refinement_study_7-15pct_13_layers_6-rockingCurves.h5"
with h5py.File(file_path, 'r') as file:
    entry_group = file['entry']
    
    # Assuming each group has a dataset with the same name as the group
    amplitude_data = entry_group['Amplitude']['Amplitude'][:]
    background_data = entry_group['Background']['Background'][:]
    correlation_data = entry_group['Correlation']['Correlation'][:]
    fwhm_first_motor_data = entry_group['FWHM first motor']['FWHM first motor'][:]
    fwhm_second_motor_data = entry_group['FWHM second motor']['FWHM second motor'][:]
    peak_pos_first_motor_data = entry_group['Peak position first motor']['Peak position first motor'][:]
    peak_pos_second_motor_data = entry_group['Peak position second motor']['Peak position second motor'][:]
    residuals_data = entry_group['Residuals']['Residuals'][:]


plt.figure()
plt.imshow(fwhm_first_motor_data, cmap='plasma', aspect='auto', vmax=0.2, vmin=0) 
plt.colorbar()
plt.show()
