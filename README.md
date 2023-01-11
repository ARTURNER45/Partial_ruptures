# Partial ruptures
Example code to find repeating earthquakes and partial ruptures from the NCEC earthquake catalog.

A simple tutorial to search for repeating earthquakes and partial ruptures of the NCEC earthquake catalog (Waldhauser and Schaff, 2008; Schaff and Waldhauser, 2005; Waldhauser, 2013; https://www.ldeo.columbia.edu/~felixw/NCAeqDD/) using a simple locaiton based criteria. In this tutorial you will locate repeating earthquakes and partial ruptures and then plot the group of co-located events. 

# Definitions

**Repeating earthquake** an earthquake which occurs within 1 rupture radius of another event and also has a magnitude 
within 0.3 magnitude units.

**Partial rupture** an earthquake that occurs within 1 rupture radius of a previously identified repeating earthquake 
but has a magnitude that is smaller than the co-located repeating earthquake. 

# Summary of methodology 

we use a simple approach to identify pairs of co-located earthquakes without waveform correlation. We take advantage of the high-quality earthquake locations already obtained in this area (Waldhauser and  Schaff, 2008; Schaff and Waldhauser, 2005; Waldhauser, 2013) and identify co-located earthquakes as pairs of earthquakes located within one rupture radius of each other. We then search for earthquakes whose catalogue locations are within one radius horizontally as well as vertically. A repeating earthquake is initially identified as any earthquake with another event within one radius and a magnitude within 0.3 magnitude units.

We then search for partial ruptures of each of these earthquakes. We search the entire catalogue for events within one radius but with a magnitude at least 0.3 Mw units smaller.

For a full description of the methodology please see Turner et al. (2023). 

# Getting set up 

The dependencies to run this code are provided in the yml. To create this conda enviroment run: 

```
conda env create --name partial_ruptures --file=partial_ruptures.yml
```
At the top of the script eqcat_pandas_GITHUB.py you will also have to set the path to the DATA 

# References 

F. Waldhauser. Real-time double-difference earthquake locations for northern california,447
2013.448

F. Waldhauser and D. Schaff. Large-scale cross correlation based relocation of two decades449
of northern california seismicity. J. geophys. Res, 113:B08311, 2008.450

F. Waldhauser and D. P. Schaff. A comprehensive search for repeating earthquakes in451
northern california: Implications for fault creep, slip rates, slip partitioning, and transient452
stress. Journal of Geophysical Research: Solid Earth, 126(11):e2021JB022495, 2021

Double difference catalog: https://www.ldeo.columbia.edu/~felixw/NCAeqDD/
