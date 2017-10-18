#
#	Very basic end-to-end test. Should generate three fits image files. These take about 1 minute to generate without GPU.
#	Tim Molteno tim@elec.ac.nz 2017.
#
#	This generates a test_output_restored.png 
#
test:
	runsane -ep dirty_2017-10-15_20:33:35_UTC.fits beam_2017-10-15_20:33:35_UTC.fits test_output

# Scale the data and write it to a PNG file so that it can be viewed.
png:
	vips scale test_output_restored.fits test_output_restored.png

# Generate some new test data from the live TART telescope. The nfft parameter of 8 creates a nice small FITS file
new_data:
	tart_image --nfft 8 --fits --dirty --beam
