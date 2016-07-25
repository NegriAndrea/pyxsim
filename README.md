What is pyXSIM?
---------------

pyXSIM is a Python package for simulating X-ray observations from astrophysical sources.

X-rays probe the high-energy universe, from hot galaxy clusters to compact objects such as
neutron stars and black holes and many interesting sources in between. pyXSIM makes it
possible to generate synthetic X-ray observations of these sources from a wide variety of 
models, whether from grid-based simulation codes such as FLASH, Enzo, and Athena, to
particle-based codes such as Gadget and AREPO, and even from datasets that have been created
"by hand", such as from NumPy arrays. pyXSIM also provides facilities for manipulating the 
synthetic observations it produces in various ways, as well as ways to export the simulated
X-ray events to other software packages to simulate the end products of specific X-ray
observatories. 

The Heritage of pyXSIM
----------------------

pyXSIM is an implementation of the [PHOX](http://www.mpa-garching.mpg.de/~kdolag/Phox/)
algorithm, developed for constructing mock X-ray observations from SPH datasets by
Veronica Biffi and Klaus Dolag. There are two relevant papers:

[Biffi, V., Dolag, K., Bohringer, H., & Lemson, G. 2012, MNRAS, 420, 3545](http://adsabs.harvard.edu/abs/2012MNRAS.420.3545B)

[Biffi, V., Dolag, K., Bohringer, H. 2013, MNRAS, 428, 1395](http://adsabs.harvard.edu/abs/2013MNRAS.428.1395B)

pyXSIM had a previous life as the `photon_simulator` analysis module as a part of the
[yt Project](http://yt-project.org). pyXSIM still depends critically on yt to provide the
link between the simulation data and the algorithm for generating the X-ray photons. For
detailed information about the design of the algorithm in yt, check out
[the SciPy 2014 Proceedings](http://conference.scipy.org/proceedings/scipy2014/zuhone.html).