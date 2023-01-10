# Partial_ruptures
Example code to find repeating earthquakes and partial ruptures from the NCEC earthquake catalog.

A simple tutorial to search for repeating earthquakes and partial ruptures of the NCEC earthquake catalog
using a simple locaiton based criteria. In this tutorial you will locate repeating earthquakes and partial ruptures and then plot the group of co-located events. 

# Definitions

**Repeating earthquake** an earthquake which occurs within 1 rupture radius of another event and also has a magnitude 
within 0.3 magnitude units 

**Partial rupture** an earthquake that occurs within 1 rupture radius of a previously identified repeating earthquake 
but has a magnitude that is smaller than the co-located repeating earthquake. 

# Summary of methodology 

we use a simple approach to identify pairs of co-located earthquakes without waveform correlation. We take advantage of the high-quality earthquake locations already obtained in this area (waldhauser 2008) and identify co-located earthquakes as pairs of earthquakes located within one rupture radius of each other. We then search for earthquakes whose catalogue locations are within one radius horizontally as well as vertically. A repeating earthquake is initially identified as any earthquake with another event within one radius and a magnitude within $\pm$ 0.3 magnitude units.

We then search for partial ruptures of each of these earthquakes. We search the entire catalogue for events within one radius but with a magnitude at least 0.3 Mw units smaller.

For a full description of the methodology please see Turner et al. (2023). 

# Getting set up 

The dependencies to run this code are provided in the yml. To create this conda enviroment run: 

```
conda install --file partial_ruptures.yml 
```

