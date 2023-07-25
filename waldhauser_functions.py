

def waldhauser_index():
    import pandas as pd
    from eqcat_pandas import Eqv
    from eqcat_pandas import readpoly
    import numpy as np 
    waldhauser_repeaters2 = load_waldhauser_events(filter_to_parkfield=True)
    self = Eqv(Update=True)
    self.remove_parkfield()
    self.filtertopoly(polyname= 'NearParkfield')
    self.projtofault(sdr = [135,0])
    self.moments(scl='Parkfield',sdrop = 10e6)
    
    #finding the index of waldhauser repeaters in the main catalouge
    a = (np.in1d(self.ids,waldhauser_repeaters2.evId))
    
    ixreffull=np.arange(0,self.neq())

    index  = ixreffull[a]
    
    
    return(index) 
   
        

def load_waldhauser_events(filter_to_parkfield=True):
    import pandas as pd
    from eqcat_pandas import Eqv 
    from eqcat_pandas import readpoly
    import numpy as np 
    """
    loading all repeating events from the waldhauser cat
    """
    
    waldhauser_repeaters = pd.read_csv('/Users/TheStuffofAlice/Dropbox/partial_ruptures/NCA_REPQcat_20210919_eventsonly2.csv',names=["YR","MO","DY","HR","MN","SC","DAYS","LAT","LON","DEP","EX","EY","EZ","MAG","DMAG","DMAGE","CCm","evId"])
 
    if filter_to_parkfield==True:
        xy=readpoly(polyname= 'NearParkfield')
        minlatitude=np.min(xy[:,1])
        maxlatitude=np.max(xy[:,1])
        minlongitude=np.min(xy[:,0])
        maxlongitude=np.max(xy[:,0])

        waldhauser_repeaters = waldhauser_repeaters[(waldhauser_repeaters['LAT'].between(minlatitude, maxlatitude)) &(waldhauser_repeaters['LON'].between(minlongitude, maxlongitude)) ]
       
    return waldhauser_repeaters
   

def load_waldhauser_sequences(filter_to_parkfield=True):
    import pandas as pd
    import pandas as pd
    from eqcat_pandas import Eqv 
    from eqcat_pandas import readpoly
    import numpy as np 
    """
    loads the repeating sequences from the waldhauser sequences
    """

    waldhauser_repeaters = pd.read_csv('/Users/TheStuffofAlice/Dropbox/partial_ruptures/CA_REPQcat_20210919_sequences2.csv',names=["NEV",   "LATm",     "LONm",     "DEPm",   "DMAGm",  "DMAGs",    "RCm",    "RCs",  "RCcv",   "RCm1",   "RCs1",  "RCcv1",  "CCm",  "seqID"])
    
    if filter_to_parkfield == True:
        xy=readpoly(polyname= 'NearParkfield')
        minlatitude=np.min(xy[:,1])
        maxlatitude=np.max(xy[:,1])
        minlongitude=np.min(xy[:,0])
        maxlongitude=np.max(xy[:,0])

        waldhauser_repeaters2 = waldhauser_repeaters[(waldhauser_repeaters['LATm'].between(minlatitude, maxlatitude)) &(waldhauser_repeaters['LONm'].between(minlongitude, maxlongitude)) ]

        waldhauser_repeaters = waldhauser_repeaters2[(waldhauser_repeaters2.NEV > 2)]
    
    return waldhauser_repeaters 
