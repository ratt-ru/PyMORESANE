# PyMORESANE

This is a Python and pyCUDA-accelerated implementation of the MORESANE
deconvolution algorithm (see [Dabbech et al. 2014](http://arxiv.org/abs/1412.5387)
for a full description of the algorithm, and
[Dabbech et al. 2012](http://www.academia.edu/1942933/Astronomical_image_deconvolution_using_sparse_priors_An_analysis-by-synthesis_approach)
for earlier work).

## BASIC INSTALLATION

Requirements:
  * [numpy](http://www.scipy.org/install.html)
  * [scipy](http://www.scipy.org/install.html)
  * [pyfits](http://www.stsci.edu/institute/software_hardware/pyfits/Download)
  * [CUDA](https://developer.nvidia.com/cuda-downloads)
  * [pycuda](http://mathema.tician.de/software/pycuda/)
  * [scikits.cuda](http://scikit-cuda.readthedocs.org/)

You can use pip to install these:

```
$ pip install -r requirements.txt
```

Now you can use the setup script to install pymoresane:

```
$ python setup.py install
```



## USAGE

Running pymoresane.py is as simple as typing pymoresane.py followed by the
inputs you desire. runsane --help should provide some assistance, otherwise the
docstrings/comments in the code are very detailed.



## AUTHOR


Feel free to contact me at jonosken@gmail.com with comments, questions,
criticism and bug-reports.




