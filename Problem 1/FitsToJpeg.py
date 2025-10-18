from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def fits_to_jpeg(fits_file, output_jpeg):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
    if data is not None:
        data = np.nan_to_num(data)
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='nipy_spectral', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity')
        plt.title(f'FITS: {fits_file}')
        plt.xlabel('X pixels')
        plt.ylabel('Y pixels')
        plt.savefig(output_jpeg, dpi=1500, bbox_inches='tight')
        plt.close()
        
        print(f"Converted {fits_file} to {output_jpeg}")
    else:
        print(f"No data found in {fits_file}")
for i in range(1,11):
    fits_to_jpeg(f"q{i}.fits", f"q{i}.jpeg")
