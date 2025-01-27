"""This code was developed as part of the LFEUI project. It aims to modernize part of the
existing routines in matlab to python. The code reads the EEG data from a BrainVision file,
applies band-pass and notch filters, computes the Power Spectral Density (PSD) using the Welch
method, and plots the PSD. It also creates topographic maps for specific frequencies and computes
the relative alpha power for each channel."""

import numpy as np
import matplotlib.pyplot as plt
import mne

# Path to the data
path = "/home/duarte/Desktop/lfeui/Data"

# Read the BrainVision file (.vhdr)
raw = mne.io.read_raw_brainvision(path + "/VP02.vhdr", preload=True)
participant = 2

# Apply band-pass filter (1–40 Hz) to remove low and high-frequency noise
raw.filter(l_freq=1.0, h_freq=100.0)

# Apply notch filter to remove power line interference (50 Hz)
raw.notch_filter(freqs=50, picks='eeg')

# Select only EEG channels
raw.pick('eeg')

# Compute the Power Spectral Density (PSD) using the Welch method
psds = raw.compute_psd(method='welch', fmin=1, fmax=70, n_fft=4096)
psd_data = psds.get_data()
freqs = psds.freqs

# Convert PSD to microvolts squared per hertz (µV²/Hz)
psd_data *= 1e12

# Convert PSD to logarithmic scale (10 * log10)
psd_data_log = 10 * np.log10(psd_data)

# Plot the power spectrum
plt.figure()
plt.plot(freqs, psd_data_log.T)
plt.xlabel('Frequency (Hz)')

# y units = 10 ^{µV²/Hz}
plt.ylabel('Power Spectral Density 10^(µV²/Hz)')

plt.title('Power Spectral Density (PSD)')

# Folder for saving is lfeui/im
save_name = f"/home/duarte/Desktop/lfeui/im/PSD_Participant_{participant}.png"
plt.savefig(save_name)
print(f"PSD plot saved")
plt.show()

# Specific frequencies for topographic analysis
specific_freqs = [6, 10, 12, 22]

# Create topographic maps for each specific frequency
for freq in specific_freqs:
    # Find the index of the closest frequency
    freq_index = np.argmin(np.abs(freqs - freq))
    
    # Extract PSD values at the specific frequency
    psd_at_freq = psd_data[:, freq_index]
    
    # Create the topographic map
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(psd_at_freq, raw.info, axes=ax, show=False, cmap='coolwarm', res=4096)
    
    # Add title and color bar
    ax.set_title(f'Topographic Map of PSD at {freq} Hz')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Power Spectral Density (µV²/Hz)')
    save_name = f"/home/duarte/Desktop/lfeui/im/Topographic_Map_{freq}Hz_Participant_{participant}.png"
    plt.savefig(save_name)
    print(f"Topographic map at {freq} Hz saved")
    plt.show()

# Define the alpha band
alpha_band = (8, 12)

# Extract the alpha band from the PSD
alpha_psd = psd_data[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])]

# Compute the mean alpha power for each channel
alpha_power = alpha_psd.mean(axis=1)

# Compute the total power for normalization
total_power = psd_data.sum(axis=1)

# Compute the relative alpha power
relative_alpha_power = alpha_power / total_power

# Plot the relative alpha power for each channel
plt.figure()
plt.bar(np.arange(len(relative_alpha_power)), relative_alpha_power)
plt.xlabel('Channels')
plt.ylabel('Relative Alpha Power (Alpha Power / Total Power)')
plt.title('Relative Alpha Power (8-12 Hz) for Each Channel')
save_name = f"/home/duarte/Desktop/lfeui/im/Relative_Alpha_Power_Participant_{participant}.png"
plt.savefig(save_name)
print(f"Relative alpha power plot saved")
plt.show()

# Display the relative alpha power for each channel
# print("Relative alpha power for each channel:")
# print(relative_alpha_power)

# Compute the total relative alpha power
total_relative_alpha_power = relative_alpha_power.sum()
print(f"Total relative alpha power: {total_relative_alpha_power}")
