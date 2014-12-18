# PyMORESANE

This is a Python and pyCUDA-accelerated implementation of the MORESANE deconvolution algorithm (see [Dabbech et al. 2014](http://arxiv.org/abs/1412.5387), and 
see [Dabbech et al. 2012](http://www.academia.edu/1942933/Astronomical_image_deconvolution_using_sparse_priors_An_analysis-by-synthesis_approach)).

### BASIC INSTALLATION

Install [numpy and scipy](http://www.scipy.org/install.html). 
Install [pyfits](http://www.stsci.edu/institute/software_hardware/pyfits/Download).

The following assumes you have numpy, scipy and pyfits installed.

Simply download all the files/zip from GitHub.

Place all files in a single directory, e.g. ~/PyMORESANE

Add the following to your .bashrc or .bash_aliases:  
alias runsane='python ~/PyMORESANE/pymoresane.py'

If you have put the PyMORESANE folder elsewhere, simply amend the alias to reflect its location.

Running pymoresane.py is as simple as typing runsane followed by the inputs you desire. runsane --help should provide some assistance, otherwise the docstrings/comments in the code are very detailed.

Feel free to contact me at jonosken@gmail.com with comments, questions, criticism and bug-reports. 




