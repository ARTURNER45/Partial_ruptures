import socket
import math
import general
import os
import glob
import obspy
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import inf
import random 
from math import cos, atan
from scipy.optimize import minimize
import scipy.stats as stats

###################
##SET PATH 
#set full path to the directory in which the code and data are.
EQPATH= '/Users/TheStuffofAlice/Dropbox/partial_ruptures/Partial_ruptures/'


####### General functions ############


def readpoly(polyname):
    """
    A function that defines a polygon of interest
    :param    polyname: name of the polygon of interest
    :return         xy: the lon, lat coordinates defining the polygon
    """

    # read the data
    fdir=os.path.join(os.environ['DATA'],'EQCATALOG',
                      'POLYGONS')
    fname=os.path.join(fdir,polyname)
    xy=np.loadtxt(fname,dtype=float)
    ii=xy[:,0]>180.
    xy[ii,0]=xy[ii,0]-360

    return xy

def init_NCDC(Update=True):
    """
    A function that initiates the NCDC earthquake catalog
    :param    Update: If True includes the real time NCeqDD catalog 
    :return   final_NCeqDD: The catalog as a pandas dataframe  
    """
        
    NCeqDD_old = pd.read_csv(EQPATH + '/DATA/EQCATALOG/DDNC_Waldhauser/NCAeqDD.v201112.1AT.csv', sep=",", header=None,names=["Year", "Month", "Day", "Hour", "Minute", "Second","Lat","Lon","Dep","EX","EY","Az","EZ","Mag","ID"])
    NCeqDD_old = NCeqDD_old.drop(['Az'], axis=1)
    
    if Update==True:
        NCeqDD_update = pd.read_csv(EQPATH + '/DATA/EQCATALOG/DDNC_Waldhauser/NCAeqDDRT.v201201.csv', sep=",", header=None,names=["Year", "Month", "Day", "Hour", "Minute", "Second","Lat","Lon","Dep","EX","EY","EZ","Mag","ID","Ver","Base","Meth"])


        NCeqDD_update = NCeqDD_update.drop(['Ver',"Base","Meth"], axis=1)

        frames = [NCeqDD_old, NCeqDD_update]


        NCeqDD = pd.concat(frames,ignore_index=True)


        final_NCeqDD = NCeqDD.sort_values(by=['Lon'], ascending=True)
    else:
        
        final_NCeqDD = NCeqDD_old

    return final_NCeqDD


def reccurance_seconds_moment(my_dates_list, input_date, my_moments_list, input_moment):
    """
    A function that finds the co-located earthquake closest in time to a target event and calcualtes the reccurance 
    time between the pair. 
    :param    my_dates_list: dates of all identified co-located repeating earthquakes 
    :param    input_date: date of the target earthquake
    :param    my_moments_list: moments of identified co-located repeating earthquakes 
    :param    input_moment: moment of the target event
    :return   mean_moment: The mean moment of the pair of events (N-m)
    :return   reccurance: reccurance time (s)
    """
    results = [d for d in sorted(my_dates_list) if d > input_date]
    output = results[0] if results else None
    idx = my_dates_list.index(output) if output else None
    reccurance = (np.datetime64(output) - np.datetime64(input_date)).item().total_seconds() if output else None
    mean_moment = (input_moment + my_moments_list[idx])/2 if output else None
    if np.datetime64(input_date) < np.datetime64('2004-09-27') and np.datetime64(output) > np.datetime64('2014-09-29'):
        #print('Pair of events span time removed for parkfield earthquake')
        reccurance = None 
        mean_moment = None
    if reccurance:
        if reccurance * 3.17098e-8 < 0.136986:
            reccurance = None 
            mean_moment = None 
    
    return mean_moment,reccurance

def deg2km():
    """
    returns kilometers per degree at the equator
    """
    return 111.32

def mag2prop(mag,scl='Parkfield',sdrop=3.e6,shmod=3.e10):
    """
    A function that calculates a range of earthquake properties 
    :param      mag:   magnitudes
    :param      scl:   scaling---either a dictionary or string
    :param    sdrop:   assumed stress drop in Pa
    :param    shmod:   shear modulus (default: 3e10)
    :return     mom:   moment in N-m
    :return      rd:   radius in meters
    :return   cfreq:   corner frequency in Hz
    :return    slip:   slip in meters
    """
    
    if isinstance(scl,str):
        # built-in scalings
        scls = {}
        scls['Mw']={'beta':1.5,'ofst':16.1}
        scls['Wyss']={'beta':1.6,'ofst':15.8}
        scls['SJB']={'beta':1.1,'intercept':3.5}
        scls['Parkfield']={'beta':1.2,'intercept':3.5}
        scl=scls.get(scl)
        
    # coefficients
    beta=scl.get('beta',None)
    ofst=scl.get('ofst',None)
    intercept=scl.get('intercept',None)
    
    if ofst is None:
        # need to compute an offset from the Mw intercept
        ofst = 16.1+(1.5-beta)*intercept

    # to numpy array if necessary
    if isinstance(mag,list):
        mag=np.array(mag)

    # log10 of moments in N-m
    mom = beta*mag+ofst-7

    # moment in N-m
    mom = np.power(10,mom)
    
    # radius cubed, for a circular crack
    rd = 7./16.*mom/sdrop
    rd = np.power(rd,1./3.)

    # corner frequency
    # see shearer, 2006 and madariaga, 1976
    cfreq = np.power(mom/sdrop,-1./3.)
    cfreq = 0.42*3.5e3*cfreq

    # slip

    slip = 1.134*sdrop/shmod*rd
    
    return mom,rd,cfreq,slip


def new_ratio(Molim,A,B):
    """
     A function that takes the ratio between the area beneath a theoretical G-R distribution and the observed G-R distribution. Used as a correction for missed moment. 
    :param      Molim:  The upper moment limit of the intergration  
    :param      A:   observed a value of G-R distribution 
    :param      B:   observed b value of G-R distribution 
    :return   ratio:   The ratio of the area beneath observed and theoretical G-R distributions 
    """
        
    A = np.array(list(A))
    B = np.array(list(B))
    A2= A[np.where(A>=1)]
    B2 = B[np.where(A>=1)]

    A3= A[np.where(A<=1)]
    B3 = B[np.where(A<=1)]

    mag3 = mag2prop(A3,scl='Mw')[0] #below the magnitude of completeness 
    mag2 = mag2prop(A2,scl='Mw')[0] #above the magnitude of completeness 
    a,b = np.polyfit(np.log10(mag3),np.log10(B3),1) #fit of below the magntiude of completeness 
    c,d = np.polyfit(np.log10(mag2),np.log10(B2),1)# fir above the magnitude of completeness 

    magr = np.arange(9715193052.376093,Molim - 32359365692.962944,28183829312.644722) #subtract 0.3
    magr = np.arange(1258925411.7941713,Molim - 3548133892.3357606,2985382618.9179692)
    logmagr = np.log10(magr)
        
        

    store = [] #below the magnitude of completeness (observed)
    store2 = [] #above the magnitude of completeness (theoretical)
    for log in logmagr:
        store2.append((((c*log)+d)))
    for log in logmagr:
        if log > 10.75:
            store.append((((c*log)+d)))
        elif log <= 10.75:
            store.append((((a*log)+b)))

    TRY = np.trapz(10**np.array(store),10**np.array(logmagr))
    TR2 = np.trapz(10**np.array(store2),10**np.array(logmagr))
    ratio = TRY/TR2
    return(ratio) 
        

#### creating a class to hold the earhtquakes #########

class Eqv:
    """
    an earthquake catalog class
    keeps track of all the values 
    """

    def __init__(self,Update):
        # just grab the most important things
        #if isinstance(eqci,eqc):
        self.df = init_NCDC(Update = Update)
        #self.df['Mag'] = self.df['Mag'].fillna(0)
        

    def neq(self):
        """
        :return  neq: number of events
        """

        neq = self.lon.size

        return neq
    
    def calc_b_value(self,magnitudes, completeness, max_mag=None, plotvar=True, plotdist = True):
        from collections import Counter
        """
        Calculate the b-value for a range of completeness magnitudes.

        Calculates a power-law fit to given magnitudes for each completeness
        magnitude.  Plots the b-values and residuals for the fitted catalogue
        against the completeness values. Computes fits using numpy.polyfit,
        which uses a least-squares technique.

        :type magnitudes: list
        :param magnitudes: Magnitudes to compute the b-value for.
        :type completeness: list
        :param completeness: list of completeness values to compute b-values for.
        :type max_mag: float
        :param max_mag: Maximum magnitude to attempt to fit in magnitudes.
        :type plotvar: bool
        :param plotvar: Turn plotting on or off.
        """
        b_values = []
        # Calculate the cdf for all magnitudes
        counts = Counter(magnitudes)
        cdf = np.zeros(len(counts))
        mag_steps = np.zeros(len(counts))
        for i, magnitude in enumerate(sorted(counts.keys(), reverse=True)):
            mag_steps[i] = magnitude
            if i > 0:
                cdf[i] = cdf[i - 1] + counts[magnitude]
            else:
                cdf[i] = counts[magnitude]

        if not max_mag:
            max_mag = max(magnitudes)
        for m_c in completeness:
            if m_c >= max_mag or m_c >= max(magnitudes):
                Logger.warning('Not computing completeness at %s, above max_mag' %
                               str(m_c))
                break
            complete_mags = []
            complete_freq = []
            complete_counts = [] 
            for i, mag in enumerate(mag_steps):
                if mag >= m_c <= max_mag:
                    complete_mags.append(mag)
                    complete_freq.append(np.log10(cdf[i]))
                    complete_counts.append(np.log10(counts[mag]))
            if len(complete_mags) < 4:
                Logger.warning('Not computing completeness above ' + str(m_c) +
                               ', fewer than 4 events')
                break
            fit = np.polyfit(complete_mags, complete_freq, 1, full=True)
            # Calculate the residuals according to the Wiemer & Wys 2000 definition
            predicted_freqs = [fit[0][1] - abs(fit[0][0] * M)
                               for M in complete_mags]
            r = 100 - ((np.sum([abs(complete_freq[i] - predicted_freqs[i])
                               for i in range(len(complete_freq))]) * 100) /
                       np.sum(complete_freq))
            b_values.append((m_c, abs(fit[0][0]), r, len(complete_mags)))
        if plotvar:
            fig, ax1 = plt.subplots()
            b_vals = ax1.scatter(list(zip(*b_values))[0], list(zip(*b_values))[1],
                                 c='k')
            resid = ax1.scatter(list(zip(*b_values))[0],
                                [100 - b for b in list(zip(*b_values))[2]], c='r')
            ax1.set_ylabel('b-value and residual')
            plt.xlabel('Completeness magnitude')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of events used in fit')
            n_ev = ax2.scatter(list(zip(*b_values))[0], list(zip(*b_values))[3],
                               c='g')
            fig.legend((b_vals, resid, n_ev),
                       ('b-values', 'residuals', 'number of events'),
                       'lower right')
            ax1.set_title('Possible completeness values')
            plt.show()
        if plotdist:

            fig, ax1 = plt.subplots()
            #plt.scatter(complete_mags, complete_freq)

            plt.plot(mag_steps,np.log10(cdf),'o')
            plt.plot(complete_mags,predicted_freqs,'grey')
            fit2 = np.polyfit(complete_mags, complete_counts, 1, full=True)
            predicted_freqs = [fit2[0][1] - abs(fit[0][0] * M)
                               for M in complete_mags]
            plt.xlabel('Magnitude')
            plt.ylabel('Log 10 cumulative frequency')
            plt.savefig(EQPATH + '/FIGS/GMT_plot/GR.png',dpi=550)
            fig, ax1 = plt.subplots()
            Mags = np.arange(-0.5,6,0.1)
            theoretical_freqs = [fit2[0][1] - abs(fit[0][0] * M)
                               for M in Mags]
            plt.plot(Mags,abs(np.array(theoretical_freqs)),'grey','--')

            plt.show()
            
            self.b_values = b_values
            self.bvalue_grad = fit2[0][0]
            self.bvalue_intercept = fit2[0][1]
            
        return counts.keys(),list(counts.values())


    
    def remove_parkfield(self):
        """
        Reomves 10 years after the Mw 6 2004 parkfield earthquake from catalog
        """
        self.df["Date"] = np.nan
        for index, row in self.df.iterrows():
                if (row["Month"]<10) & (row["Day"]<10):
                        self.df.at[index, 'Date'] = np.datetime64(str(int(row["Year"]))+'-0'+str(int(row["Month"]))+'-0'+str(int(row["Day"])))
                elif (row["Month"]>10) & (row["Day"]<10):
                        self.df.at[index, 'Date'] =  np.datetime64(str(int(row["Year"]))+'-'+str(int(row["Month"]))+'-0'+str(int(row["Day"])))
                elif (row["Month"]<10) & (row["Day"]>10):
                        self.df.at[index, 'Date'] = np.datetime64(str(int(row["Year"]))+'-0'+str(int(row["Month"]))+'-'+str(int(row["Day"])))
                elif(row["Month"]>10) & (row["Day"]>10):
                        self.df.at[index, 'Date'] = np.datetime64(str(int(row["Year"]))+'-'+str(int(row["Month"]))+'-'+str(int(row["Day"])))
                parkfield_start_time = np.datetime64("2004-09-27")
                parkfield_end_time = np.datetime64("2014-09-29")

        print('filtering')

        df1 = self.df[(self.df['Date'] < parkfield_start_time)]


        df3 =  self.df[(self.df['Date'] > parkfield_end_time)]
        self.df  = pd.concat([df1,df3])
        
        
        return
        
    def nofilter(self):
        """
        Function that converts to class attributes without any filtering
        """
        print('converting to class attributes without filtering')
        df2 = self.df 
        self.lon = np.array(df2.Lon.astype(float))
        print('lon done')
        self.lat = np.array(df2.Lat.astype(float))
        print('lat done')
        self.depth = np.array((df2.Dep.astype(float)))
        print('dep done')
        self.mpref = np.array(df2.Mag.astype(float))
        self.time = [] 
        year = np.array([yr for yr in df2.Year])
        month = np.array([mnth for mnth in df2.Month])
        day = np.array([day for day in df2.Day])
        hour = np.array([hr for hr in df2.Hour])
        minute = np.array([mine for mine in df2.Minute])
        second = np.array([sc for sc in df2.Second])
        for i in range(0,len(year)):
            if second[i] == 60.0:
                second[i] = 0 
                minute[i] = minute[i] + 1
            self.time.append(obspy.UTCDateTime(year[i],month[i],day[i],hour[i],minute[i],second[i]))
        self.ids = np.array(df2.ID.astype(int))
        self.EX = np.array(df2.EX.astype(float))
        self.EZ =np.array( df2.EZ.astype(float))
            
        for mif in  self.mpref:
            if np.isnan(mif):
                print('Replacing nan with 0.0')  
                self.mpref[i] = np.float(0) 

    
 
                   
                    
    def filtertopoly(self,polyname=None,depth=None,mag=None,Loc=None):
        """
        Filters the catalog to a chosen polygon 
        :param   df: pandas dataframe containing the event infomation 
        :param   polyname: the name of a polygon of interest
        :return        
        """


        if polyname:
            # read the polygon
            xy=readpoly(polyname)

            # set minimimum and maximum ranges
            minlatitude=np.min(xy[:,1])
            maxlatitude=np.max(xy[:,1])
            minlongitude=np.min(xy[:,0])
            maxlongitude=np.max(xy[:,0])
            print(minlatitude,maxlatitude,minlongitude,maxlongitude)
            df2 = self.df[(self.df['Lat'].between(minlatitude, maxlatitude)) & (self.df['Lon'].between(minlongitude, maxlongitude)) ]
            
            if depth is not None: 
                df2 = df2[df2['Dep'].between(depth[0],depth[1])]
            if mag is not None:
                if len(mag) == 1:
                    df2 = df2[df2['Mag'] >= mag[0]]
                if len(mag) == 2:
                    df2 = df2[df2['Mag'].between(mag[0],mag[1])]
            if Loc is not None: 
                df2 = df2[df2['Lat'].between(Loc[0]-0.01,Loc[0]+0.01)]
                df2 = df2[df2['Lon'].between(Loc[1]-0.01,Loc[1]+0.01)]
        
            self.lon = np.array(df2.Lon.astype(float))
            self.lat = np.array(df2.Lat.astype(float))
            self.depth = np.array((df2.Dep.astype(float)))
            self.mpref = np.array(df2.Mag.astype(float))
            self.time = np.array(df2.Date.astype(str))
            self.ids = np.array(df2.ID.astype(int))
            self.EX = np.array(df2.EX.astype(float))
            self.EZ =np.array( df2.EZ.astype(float))
            
            for mif in  self.mpref:
                if np.isnan(mif):
                    print('Replacing nan with 0.0')  
                    self.mpref[i] = np.float(0) 
        
    def setdef(self,vname,vl):
        """
        :param   vname: the name of a value to set if not already there
        :param      vl: the value to give it
        """

        try:
            if getattribute(self,vname) is None:
                setatt(self,vname,vl)
        except:
            setattr(self,vname,vl)
            
            
    def projtofault(self,rloc=None,sdr=None):
        """
        Project earthquake locations onto the fault 
        :param    rloc: reference location 
        :param     sdr: strike,dip,rake
        """

        # set reference locations
        if rloc is not None:
            self.rlon=rloc[0]
            self.rlat=rloc[1]
            if len(rloc)>=3:
                self.rdepth=rloc[2]
            else:
                self.setdef('rdepth',0.)
        else:
            self.setdef('rlon',np.median(self.lon))
            self.setdef('rlat',np.median(self.lat))
            self.setdef('rdepth',0.)

        # set orientation
        if sdr is not None:
            self.strike=sdr[0]
            self.dip=sdr[1]
            if len(sdr)>=3:
                self.rake=sdr[2]
        else:
            self.setdef('strike',0.)
            self.setdef('dip',90.)

        # just assume a grid
        lonrat = np.cos(math.pi/180.*self.rlat)
        xyz=np.vstack([(self.lon-self.rlon)*(deg2km()*lonrat),
                       (self.lat-self.rlat)*deg2km(),
                       self.depth-self.rdepth])


        # project to preferred directions
        # along strike
        sdir=np.array([np.sin(self.strike*np.pi/180),
                       np.cos(self.strike*np.pi/180),0])
        # perpendicular to strike
        pdir=np.array([np.sin((self.strike+90.)*np.pi/180),
                       np.cos((self.strike+90.)*np.pi/180),0])
        # along dip
        ddir=np.array([pdir[0]*np.cos(self.dip*np.pi/180),
                       pdir[1]*np.cos(self.dip*np.pi/180),
                       np.sin(self.dip*np.pi/180)])
        # third direction
        tdir = np.cross(sdir,-ddir)
        

        
        self.x = np.dot(sdir,xyz)
        self.y = np.dot(ddir,xyz)
        self.z = np.dot(tdir,xyz)
        self.xp = np.dot(pdir,xyz)    
        
    def moments(self,scl='Parkfield',sdrop=3.e6,shmod=3.e10):
            """
            compute moments from the earthquake magnitudes
            :param      scl:   scaling---either a dictionary or string
            :param    sdrop:   assumed stress drop in Pa
            :param    shmod:   shear modulus (default: 3e10)
            """

                    
            self.mom,self.rad,self.cfreq,self.slip=mag2prop(self.mpref,scl=scl,sdrop=sdrop,shmod=shmod)
            
                
    
    def findnearbyevents(self,ixref=None,ixtar=None,dsts=0.2,herr=0.02,verr=0.05):
        """
        Find co-located events 
        :param    ixref: indices of reference events to look near
        :param    ixtar: indices of target events to be close
        :param     dsts: maximum distance for one or all events 
        :param     herr: horizontal uncertainty, in km
        :param     verr: vertical uncertainty, in km
        """
        print('staring finding nearby events')
        # default is to use all the events
        if ixref is None:
            ixref=np.arange(0,self.neq())
        if ixtar is None:
            ixtar=np.arange(0,self.neq())
        
        # duplicate distances if necessary
        dsts=np.atleast_1d(dsts)

        if dsts.size==1:
            dsts=np.repeat(dsts,len(self.x[ixref]))
  
        if ixref.dtype=='bool':
            ixref=np.where(ixref)[0]
        if ixtar.dtype=='bool':
            ixtar=np.where(ixtar)[0]
            
        # initialize output
        iref=np.array([],dtype=int)
        itar=np.array([],dtype=int)
            
        if ixref.size:
            
            # bin into boxes
            spc=np.max(dsts)*2.05 #why is this multiplied by 2.05. I have changed this so its not giving nan
            if herr is not None:
                spc=spc+np.maximum(herr,verr)*2.05
            self.biningrid(spc=spc)
            
            # identify the boxes
            rbxs=self.ibox[ixref]
            igdx,igdxp,igdd=self.igdx[ixref],self.igdxp[ixref],self.igdd[ixref]
            tbxs=self.ibox[ixtar]

            # sort the target events by box number
            ix=np.argsort(tbxs)
            tbxs,ixtar=tbxs[ix],ixtar[ix]
        
            # identify the locations
            xr,yr,zr,momr=self.x[ixref],self.y[ixref],self.z[ixref],self.mom[ixref]
            xt,yt,zt,momt=self.x[ixtar],self.y[ixtar],self.z[ixtar],self.mom[ixtar]
    

            # how much to shift boxes
            shf=np.arange(-1,1.01).astype(int)
            xshf,yshf,zshf=np.meshgrid(shf,shf,shf)
            xshf,yshf,zshf=xshf.flatten(),yshf.flatten(),zshf.flatten()
            
            # the plausible target boxes
            print('number of grids'+str(len(xshf)))
            for k in range(0,len(xshf)):
                print('working on grid'+str(k))
                #print('Box {:d} of {:d}'.format(k+1,len(xshf)))

                # the plausible target boxes
                igdxs,igdxps,igdds=xshf[k]+igdx,yshf[k]+igdxp,zshf[k]+igdd
                tbx=self.xpd2box(igdxs,igdxps,igdds)
            
                # identify the range of reference earthquakes that have box 
                # numbers that could be matched by the target earthquakes 
                print('identify events in box')
                i1=np.searchsorted(tbxs,tbx,'left')
                i2=np.searchsorted(tbxs,tbx,'right')

                # indices of target earthquakes
                ivl=np.arange(0,i1.size,dtype=int)

                # keep track of the possible pairs
                id1,id2=np.array([],dtype=int),np.array([],dtype=int)
                    
                # look if there is anything to look for
                ilk,=np.where(i2!=i1)
                while ilk.size:
                    # identify relevant pairs
                    i1,i2,ivl=i1[ilk],i2[ilk],ivl[ilk]

                    # add these pairs to the set
                    id1=np.append(id1,i1)
                    id2=np.append(id2,ivl)
                    
                    # and move on to the next
                    i1=i1+1
                    ilk,=np.where(i2>i1)

                # note all the pairs
                iok=id1!=id2
                id1,id2=id1[iok],id2[iok]
                iref=np.append(iref,id2)
                itar=np.append(itar,id1)
            
            # and check acceptable distances
            print('checking distances')
            dst2=np.power(xr[iref]-xt[itar],2)\
                +np.power(yr[iref]-yt[itar],2)
            vdst2=np.power(zr[iref]-zt[itar],2)


            if herr is not None:
                iok=np.logical_and(dst2<np.power(dsts[iref]+herr,2),
                                   vdst2<np.power(dsts[iref]+verr,2))

            else:
                dst2=dst2+vdst2
                iok<=np.power(dsts[iref],2)
            iref,itar=iref[iok],itar[iok]
        
            # back to original indices
            iref,itar=ixref[iref],ixtar[itar]
	

        return iref,itar

    def find_repeaters(self,irefo,itaro):
        """
        Find co-located repeating events (within 1 radius and 0.3 magintude units)
        :param    irefo: reference events
        :param    itaro: colocated events
        :return   irefo_repeat: repeating earthquake reference events
        :return   itaro_repeat: repeating earthquake co-located events
        """
        irefo_resamp,itaro_resamp = irefo,itaro
        irefo_repeat = [irefo_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ]
        itaro_repeat = [itaro_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ]
        return irefo_repeat,itaro_repeat

    def find_partials(self,irefo,itaro):
        """
        Find co-located repeating events (within 1 radius less than 0.3 magintude units smaller )
        :param    irefo: reference events
        :param    itaro: colocated events
        :return   irefo_repeat: repeating earthquake reference events
        :return   itaro_repeat: partial rupture co-located events
        """
            
        irefo_partial = np.array([irefo[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])

        itaro_partial = np.array([itaro[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])

        return irefo_partial,itaro_partial

    
    def ratio_plot(self,irefo,itaro):      
        """
        Calcualte ratio of total moment in groups of co-located events to the moment only in repeating earthquakes 
        :param    irefo: reference events
        :param    itaro: colocated events
        :return   repeaters_moment: moment in repeaters 
        return:   partial_moment: moment in partial ruptures
        return:   repeaters_moment_med: median moment in repeaters 
        return:   partial_moment_norm: partial moment normalized by number of repeaters 
        """

        irefo_repeat = np.array([irefo[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] + 0.3   >= self.mpref[itaro[x]]  >= self.mpref[irefo[x]] - 0.3 ])

        itaro_repeat = np.array([itaro[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] + 0.3   >= self.mpref[itaro[x]]  >= self.mpref[irefo[x]] - 0.3 ])


        irefo_partial = np.array([irefo[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])

        itaro_partial = np.array([itaro[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])
        print(irefo_repeat)
        
        print('minimum moment in repeat list'+str(np.format_float_scientific(np.min(self.mom[irefo_repeat]))))

        #Getting rid of the ones that are just finding themselves 
        irefo_repeat_unique = irefo_repeat[itaro_repeat - irefo_repeat != 0 ]
        itaro_repeat_unique = itaro_repeat[itaro_repeat - irefo_repeat != 0 ]
        
        print('minimum moment in unnique list'+str(np.format_float_scientific(np.min(self.mom[irefo_repeat_unique]))))

        repeaters_moment = np.zeros(len(irefo_repeat_unique))
        repeaters_moment_med = np.zeros(len(irefo_repeat_unique))
        partial_moment =  np.zeros(len(irefo_repeat_unique))
        partial_moment_norm = np.zeros(len(irefo_repeat_unique))
        partial_momentnocor = np.zeros(len(irefo_repeat_unique))
        sumpr = np.zeros(len(irefo_repeat_unique))
        time = np.zeros(len(irefo_repeat_unique))

        j = 0 
        for item in irefo_repeat_unique:
            if self.mom[item] <= 1e11:
                print('moment of event being considered: ' + str(np.format_float_scientific(self.mom[item])))
            tar = np.array(np.where(irefo_repeat_unique == item)[0])
            par = np.array(np.where(irefo_partial== item)[0])
            if len(tar) != 0 :
                repeaters_moment[j] = np.sum(self.mom[itaro_repeat_unique][tar]) + self.mom[item]
                repeaters_moment_med[j] = np.median(np.append(np.array(self.mom[itaro_repeat_unique][tar]), self.mom[item]))
                par = par.astype(int)
                partial_moment[j] = np.sum(self.mom[itaro_partial][par])
                partial_moment_norm[j] = np.sum(self.mom[itaro_partial][par])/np.array(len(tar))
            j = j+1


        return repeaters_moment,partial_moment,repeaters_moment_med,partial_moment_norm

    
 
    def jackknife_slope(self,irefo,itaro,num_iter):
        """
        Plotting the moment-reccurance of identified repeating earthquakes and jacknifing the slope 
        :param    irefo: reference events
        :param    itaro: colocated events
        :param    num_iter: number of iterations for jacknifing 
        """
        bins = np.log10(mag2prop(np.arange(0.99,5,0.43),'Parkfield',sdrop = 10e10))[0]  
        slope_store = np.zeros(shape=(num_iter))
        irefo_resamp,itaro_resamp = irefo,itaro
        irefo_repeat2 = np.array([irefo_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ])

        itaro_repeat2 = np.array([itaro_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ])
        
        irefo_repeat = np.array([irefo_repeat2[x] for x in range(0,len(irefo_repeat2)) if self.mpref[irefo_repeat2[x]] >=1.3 or self.mpref[itaro_repeat2[x]] >=1.3])
     
        itaro_repeat = np.array([itaro_repeat2[x] for x in range(0,len(itaro_repeat2)) if self.mpref[irefo_repeat2[x]] >=1.3 or self.mpref[itaro_repeat2[x]] >=1.3])
        
        irefo_repeat_unique = irefo_repeat[itaro_repeat - irefo_repeat != 0 ]
        itaro_repeat_unique = itaro_repeat[itaro_repeat - irefo_repeat != 0 ]
        
        
        
        Tr_store = np.zeros(shape = len(irefo_repeat_unique))
        Mr_store = np.zeros(shape = len(irefo_repeat_unique))
        j = 0 
        for item in irefo_repeat_unique:
            tar = np.array(np.where(irefo_repeat_unique == item)[0])
            a,b = reccurance_seconds_momentNEW([self.time[itaro_repeat_unique][t] for t in tar], self.time[item], [self.mom[itaro_repeat_unique][t] for t in tar], self.mom[item])
            Tr_store[j] = b 
            Mr_store[j] = a 
            j = j+1

        mean_moments_cut_short = Mr_store
        Tr_seconds_cut_short = Tr_store
        
    
        Mr_store = Mr_store[~np.isnan(Mr_store)]
        Tr_store = Tr_store[~np.isnan(Tr_store)]

        X = np.array(np.log10(Mr_store)).reshape(-1,1)
        y = np.array(np.log10(Tr_store)).reshape(1,-1)[0]
        bins = np.log10(mag2prop(np.arange(0.99,5,0.43),'Parkfield',sdrop = 10e10)[0])
        
        Median_bin_store = np.zeros(shape=(len(bins)-1,num_iter) )

        plt.scatter(mean_moments_cut_short,Tr_seconds_cut_short)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        f = np.stack((mean_moments_cut_short,Tr_seconds_cut_short),axis =0)
        f = f.T
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/TMP/alice_seq_points.txt", f, delimiter=" ")
        
        hist, bins = np.histogram(np.log10(mean_moments_cut_short), bins = bins)
        
        HIST1 = hist
        
        print('these are the hist')
        print(HIST1)


        print('calculating weighted slope ')
        for Loopi in range(0,num_iter):
            print('Iteration:'+str(Loopi+1))
            idx = np.random.choice(np.arange(len(Tr_seconds_cut_short)), int(len(Tr_seconds_cut_short)))
            
            Tr_seconds_cut_short_resamp= np.array(Tr_seconds_cut_short)[idx]

            mean_moments_cut_short_resamp = np.array(mean_moments_cut_short)[idx]
            print(np.log10(mean_moments_cut_short_resamp),np.log10(abs(np.array(Tr_seconds_cut_short_resamp))))
            bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(mean_moments_cut_short_resamp),np.log10(abs(np.array(Tr_seconds_cut_short_resamp))), statistic=np.nanmedian,bins = bins)

      
            hist, bins = np.histogram(np.log10(mean_moments_cut_short_resamp), bins = bins)

    # Threshold frequency
            freq = 250

    # Zero out low values
            bin_means[np.where(hist <= freq)] = 'nan'
            x = (bin_edges[:-1] + bin_edges[1:])/2
            y = bin_means
            
            print('!!!')
            print(bin_edges)
            print(x,y)
            
            Median_bin_store[:,Loopi] = y
            
            x = (bin_edges[:-1] + bin_edges[1:])/2

            x = x[~np.isnan(y)]
            hist = hist[~np.isnan(y)]
            
            y = y[~np.isnan(y)]


            x = x[~np.isnan(y)]
            histo = hist[~np.isnan(y)]
            y = y[~np.isnan(y)]

            X, Y = x.reshape(-1,1), y.reshape(-1,1)
            REG = linear_model.LinearRegression().fit(X, Y)

    
    
        from scipy import stats
        NUMEST = [np.count_nonzero(~np.isnan(Median_bin_store[i,:])) for i in range(0,len(bins)-1)]
        idx = np.where(np.array(NUMEST) > num_iter -25)
        percentile95 = np.nanpercentile(Median_bin_store[idx,:][0], 95, axis = 1)
        percentile5 = np.nanpercentile(Median_bin_store[idx,:][0], 5, axis = 1)
        median_median = np.nanmedian(Median_bin_store[idx,:][0],axis =1) 
        std = np.nanstd(Median_bin_store[idx,:][0],axis =1) 
        sample_weight = 1/std**2
        print(len(sample_weight))
        print(sample_weight)
        print('----')
        print(len(x),len(y),len(HIST1))
        print(idx)
        x = (bin_edges[1:])[idx]
        x = x
        print(len(x),len(y),len(HIST1))
        plt.figure()

        plt.plot(np.log10(mean_moments_cut_short),np.log10(abs(np.array(Tr_seconds_cut_short))),'y.',zorder = 0)
        plt.scatter(x,median_median,zorder = 2)
        plt.vlines(x,percentile5,percentile95,zorder = 1)
        
        x = x = (bin_edges[1:])[idx]
        HIST2=sample_weight
        y = median_median
        
        ('something is going wrong with exporting to GMT') 
        print(x)
        print(median_median)

        f = np.stack((x,y),axis =0)
        f = f.T
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/TMP/alice_binmed.txt", f, delimiter=" ")
        f = np.stack((x,percentile5,percentile95),axis = 0)
        f = f.T
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/TMP/alice_binerr.txt", f, delimiter=" ")

        
        print(x,y)

        X, Y = x.reshape(-1,1), y.reshape(-1,1)
        REG = linear_model.LinearRegression().fit(X, Y,HIST2)
        
        print('weighted SLOPE:'+str(REG.coef_))
        
        FINALslope = REG.coef_

        plt.plot(X, REG.predict(X), color='blue', linewidth=3, label='Weighted model')
        plt.show()
        y = REG.predict(X)
        x1=np.arange(9,19,1).reshape(-1,1)
        f = np.stack((x1,REG.predict(x1)),axis = 0)
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/TMP/alice_bfline.txt", f[:,:,0].T, delimiter=" ")
        idx2 = idx
        print('bootstrapping slope')
        for Loopi in range(0,num_iter):
            print('Iteration:'+str(Loopi+1))
            idx = np.random.choice(np.arange(len(Tr_seconds_cut_short)), int(len(Tr_seconds_cut_short)))
            Tr_seconds_cut_short_resamp= np.array(Tr_seconds_cut_short)[idx]

            mean_moments_cut_short_resamp = np.array(mean_moments_cut_short)[idx]
            bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(mean_moments_cut_short_resamp),np.log10(abs(np.array(Tr_seconds_cut_short_resamp))), statistic=np.nanmedian, bins = bins)

    # Get histogram to see what points do not have enough points in each bin 
            hist, bins = np.histogram(np.log10(mean_moments_cut_short_resamp), bins = bins)

    # Threshold frequency
            freq = 50

    # Zero out low values
            bin_means[np.where(hist <= freq)] = 'nan'
            x = (bin_edges[:-1] + bin_edges[1:])/2
            y = bin_means
            
            Median_bin_store[:,Loopi] = y

    
            plt.scatter(x,y)
            x = x[idx2]
            y = y[idx2]
     

            X, Y = x.reshape(-1,1), y.reshape(-1,1)
            REG = linear_model.LinearRegression().fit(X, Y,HIST2)

    
            slope_store[Loopi] = REG.coef_
        
        percentile95_slope = np.nanpercentile(slope_store, 95, axis = 0)
        percentile5_slope = np.nanpercentile(slope_store, 5, axis = 0)

        print("weighted slope:"+ str(FINALslope))
        print("95th percentile slope:"+ str(percentile95_slope))
        print("5th percentile slope:" + str(percentile5_slope))
        print("BIN MEANS:" + str(bin_means))
        
        return 
    
    def jackknife_slope_corrected(self,irefo,itaro,num_iter,bin_size,ratio):
        """
        Plotting the moment-reccurance of identified repeating earthquakes and jacknifing the slope               corrected by the ratio of small magnitude missing events 
        :param    irefo: reference events
        :param    itaro: colocated events
        :param    ratio: ratio of observed moment to theoretical moment in the G-R distribution 
        :param    num_iter: number of iterations for jacknifing 
        """     
        print('THIS IS THE INPUT RATIO')
        print(ratio)
        num_iter = num_iter
        bins = np.log10(mag2prop(np.arange(0.99,5,0.43),'Parkfield',sdrop = 10e10)[0])
        Median_bin_store = np.zeros(shape=(len(bins)-1,num_iter) )
        Corrected_Median_bin_store = np.zeros(shape=(len(bins)-1,num_iter) )

        slope_store = np.zeros(shape=(num_iter))
        Corrected_slope_store = np.zeros(shape=(num_iter))

        slope_store = np.zeros(shape=(num_iter))
        irefo_resamp,itaro_resamp = irefo,itaro
        irefo_repeat2 = np.array([irefo_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ])

        itaro_repeat2 = np.array([itaro_resamp[x] for x in range(0,len(itaro_resamp)) if self.mpref[irefo_resamp[x]] + 0.3   >= self.mpref[itaro_resamp[x]]  >= self.mpref[irefo_resamp[x]] - 0.3 ])
        
        irefo_repeat = np.array([irefo_repeat2[x] for x in range(0,len(irefo_repeat2)) if self.mpref[irefo_repeat2[x]] >=1.3 or self.mpref[itaro_repeat2[x]] >=1.3])
     
        itaro_repeat = np.array([itaro_repeat2[x] for x in range(0,len(itaro_repeat2)) if self.mpref[irefo_repeat2[x]] >=1.3 or self.mpref[itaro_repeat2[x]] >=1.3])
        
        irefo_repeat_unique = irefo_repeat[itaro_repeat - irefo_repeat != 0 ]
        itaro_repeat_unique = itaro_repeat[itaro_repeat - irefo_repeat != 0 ]
        
        
        
        Tr_store = np.zeros(shape = len(irefo_repeat_unique))
        Mr_store = np.zeros(shape = len(irefo_repeat_unique))
        j = 0 
        for item in irefo_repeat_unique:
            tar = np.array(np.where(irefo_repeat_unique == item)[0])
            a,b = reccurance_seconds_momentNEW([self.time[itaro_repeat_unique][t] for t in tar], self.time[item], [self.mom[itaro_repeat_unique][t] for t in tar], self.mom[item])
            Tr_store[j] = b 
            Mr_store[j] = a 
            j = j+1

        Tr_seconds_cut_short= Tr_store
        mean_moments_cut_short = Mr_store

        hist, bins = np.histogram(np.log10(mean_moments_cut_short), bins = bins)
        
        HIST1 = hist
        
        print('these are the hist')
        print(HIST1)

#         plt.scatter(mean_moments_cut_short,Tr_seconds_cut_short)
#         plt.yscale('log')
#         plt.xscale('log')
        f = np.stack((mean_moments_cut_short,Tr_seconds_cut_short),axis =0)
        f = f.T
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/TMP/alice_seq_points.txt", f, delimiter=" ")

        
        print('calculating weighted slope ')
        for Loopi in range(0,num_iter):
            print('Iteration:'+str(Loopi+1))
            #idx = np.random.choice(np.arange(len(Tr_seconds_cut_short)), int(0.6*len(Tr_seconds_cut_short)))
            #idx = (random.sample(list(np.arange(len(Tr_seconds_cut_short))),k=int(0.8*len(Tr_seconds_cut_short))))
            idx = np.random.choice(np.arange(len(Tr_seconds_cut_short)), int(len(Tr_seconds_cut_short)))
            
            Tr_seconds_cut_short_resamp= np.array(Tr_seconds_cut_short)[idx]

            mean_moments_cut_short_resamp = np.array(mean_moments_cut_short)[idx]
 
            bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(mean_moments_cut_short_resamp),np.log10(abs(np.array(Tr_seconds_cut_short_resamp))), statistic='median',bins = bins)

    # Get histogram to see what points do not have enough points in each bin 
            hist, bins = np.histogram(np.log10(mean_moments_cut_short_resamp), bins = bins)

    # Threshold frequency
            freq = 250

    # Zero out low values
    # Zero out low values
            bin_means[np.where(hist <= freq)] = 'nan'

            x = (bin_edges[:-1] + bin_edges[1:])/2
            y = bin_means
            print('111')
            print(len(y),len(ratio))
            z = np.log10((10**y)*((ratio)))
            
            Median_bin_store[:,Loopi] = y
            Corrected_Median_bin_store[:,Loopi] = z

            x = x[~np.isnan(y)]
            hist = hist[~np.isnan(y)]
            z = z[~np.isnan(y)]
            y = y[~np.isnan(y)]
            print(len(y),len(ratio))
     

            X, Y, Z = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)
        
            print('----')
            REG = linear_model.LinearRegression().fit(X, Y)
            CORRECTED_REG = linear_model.LinearRegression().fit(X, Z)
            



    
        from scipy import stats
        NUMEST = [np.count_nonzero(~np.isnan(Median_bin_store[i,:])) for i in range(0,len(bins)-1)]
        #exclude items that don't have very many bootstraps 
        idx = np.where(np.array(NUMEST) > num_iter-25)
        percentile95 = np.nanpercentile(Median_bin_store[idx,:][0], 95, axis = 1)
        percentile5 = np.nanpercentile(Median_bin_store[idx,:][0], 5, axis = 1)
        Corrected_percentile95 = np.nanpercentile(Corrected_Median_bin_store[idx,:][0], 95, axis = 1)
        Corrected_percentile5 = np.nanpercentile(Corrected_Median_bin_store[idx,:][0], 5, axis = 1)
        median_median = np.nanmedian(Median_bin_store[idx,:][0],axis =1) 
        Corrected_median_median = np.nanmedian(Corrected_Median_bin_store[idx,:][0],axis =1) 
        std = np.nanstd(Median_bin_store[idx,:][0],axis =1)
        sample_weight = 1/ std**2
        x = (bin_edges[1:])[idx]
        x = x
        x = x = (bin_edges[1:])[idx]
        HIST2=sample_weight
        y = median_median
        z= Corrected_median_median
       
        X, Y, Z = x.reshape(-1,1), y.reshape(-1,1),z.reshape(-1,1)
        REG = linear_model.LinearRegression().fit(X, Y,HIST2)
        plt.plot(np.log10(mean_moments_cut_short),np.log10(abs(np.array(Tr_seconds_cut_short))),'y.',zorder = 0)
        plt.scatter(x,median_median,zorder = 2)
        plt.vlines(x,percentile5,percentile95,zorder = 1)
        plt.plot(X, REG.predict(X), color='blue', linewidth=3, label='Weighted model')
        plt.show()


        
        percentile95 = np.log10(10**percentile95*7.29325e-10)
        percentile5 = np.log10(10**percentile5*7.29325e-10)
        Corrected_percentile95 = np.log10(10** Corrected_percentile95*7.29325e-10)
        Corrected_percentile5 = np.log10(10**Corrected_percentile5*7.29325e-10)
        median_median = np.log10(10**median_median*7.29325e-10)
        Corrected_median_median =  np.log10(10**Corrected_median_median*7.29325e-10)
        std = np.nanstd(Median_bin_store[idx,:][0],axis =1)
        std_corrected = np.nanstd(Corrected_Median_bin_store[idx,:][0],axis =1)
        sample_weight = 1/ std**2
        corrected_sample_weight = 1/ std_corrected**2
        


        x = x = (bin_edges[1:])[idx]
        HIST2=sample_weight
        y = median_median
        z= Corrected_median_median

        f = np.stack((x,y),axis =0)
        f = f.T
        #print('----')
        #print(f.shape)
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/alice_binmed.txt", f, delimiter=" ")
        
        f = np.stack((x,z),axis =0)
        f = f.T
        #print('----')
        #print(f.shape)
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/corrected_alice_binmed.txt", f, delimiter=" ")
        
        
        
        f = np.stack((x,percentile5,percentile95),axis = 0)
        f = f.T
        #print('----')
        #print(f.shape)
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/alice_binerr.txt", f, delimiter=" ")
        
        f = np.stack((x,Corrected_percentile5,Corrected_percentile95),axis = 0)
        f = f.T
        #print('----')
        #print(f.shape)
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/Corrected_alice_binerr.txt", f, delimiter=" ")

        

#         sample_weight = sample_weight[~np.isnan(y)]
#         Corrected_sample_weight = sample_weight[~np.isnan(y)]

        X, Y, Z = x.reshape(-1,1), y.reshape(-1,1),z.reshape(-1,1)
        REG = linear_model.LinearRegression().fit(X, Y,HIST2)
        plt.plot(X, REG.predict(X), color='blue', linewidth=3, label='Weighted model')
        plt.show()
        FINAL = REG.coef_
        CORRECTED_REG = linear_model.LinearRegression().fit(X, Z, HIST2)
        CORFINAL = CORRECTED_REG.coef_
        y = REG.predict(X)
        z = CORRECTED_REG.predict(X)
        x1=np.arange(9,19,1).reshape(-1,1)
        f = np.stack((x1,REG.predict(x1)),axis = 0)
        #print('----')
        #print(f[:,:,0].shape)
        #print(f[:,:,0])
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/alice_bfline.txt", f[:,:,0].T, delimiter=" ")
        f = np.stack((x1,CORRECTED_REG.predict(x1)),axis = 0)
        #print('----')
        #print(f[:,:,0].shape)
        #print(f[:,:,0])
        #print('----')
        np.savetxt("/EQPATH + '/FIGS/GMT_plot/DISCUSSIONFIG/Corrected_alice_bfline.txt", f[:,:,0].T, delimiter=" ")
        
        print('weighted SLOPE:'+str(FINAL))
        

        
        
        idx2= idx
        print('bootstrapping slope')
        for Loopi in range(0,num_iter):
            print('Iteration:'+str(Loopi+1))
            idx = np.random.choice(np.arange(len(Tr_seconds_cut_short)), int(len(Tr_seconds_cut_short)))


            Tr_seconds_cut_short_resamp= np.array(Tr_seconds_cut_short)[idx]

            mean_moments_cut_short_resamp = np.array(mean_moments_cut_short)[idx]

   
       
            bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(mean_moments_cut_short_resamp),np.log10(abs(np.array(Tr_seconds_cut_short_resamp))), statistic='median',bins = bins)

    # Get histogram to see what points do not have enough points in each bin 
            hist, bins = np.histogram(np.log10(mean_moments_cut_short_resamp), bins = bins)

    # Threshold frequency
            freq = 250

    # Zero out low values
            bin_means[np.where(hist <= freq)] = 'nan'

            x = (bin_edges[:-1] + bin_edges[1:])/2
            y = bin_means
            z = np.log10((10**y)*((ratio)))
            
            Median_bin_store[:,Loopi] = y
            Corrected_Median_bin_store[:,Loopi] = z

    
            #plt.scatter(x,y)

            x = x[idx2]
            HIST2 = sample_weight
            y = y[idx2]
            z = z[idx2]

     

            X, Y, Z = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)
        
            print('----')
            REG = linear_model.LinearRegression().fit(X, Y,HIST2)
            CORRECTED_REG = linear_model.LinearRegression().fit(X, Z,HIST2)
            
            slope_store[Loopi] = REG.coef_
            Corrected_slope_store[Loopi] = CORRECTED_REG.coef_
        
        
        percentile95_slope = np.nanpercentile(slope_store, 95, axis = 0)
        percentile5_slope = np.nanpercentile(slope_store, 5, axis = 0)
        
        Corrected_percentile95_slope = np.nanpercentile(Corrected_slope_store, 95, axis = 0)
        Corrected_percentile5_slope = np.nanpercentile(Corrected_slope_store, 5, axis = 0)

        print("weighted slope:"+ str(FINAL))
        print("95th percentile slope:"+ str(percentile95_slope))
        print("5th percentile slope:" + str(percentile5_slope))
        
        print("Corrected weighted slope:"+ str(CORFINAL))
        print("Corrected 95th percentile slope:"+ str(Corrected_percentile95_slope))
        print("Corrected 5th percentile slope:" + str(Corrected_percentile5_slope))
        
        return 
    
    
    def biningrid(self,spc=0.2):
        """
        bin all the earthquakes into a grid 
        :param     spc: the grid spacing in km
             sets  gdx: the edges of the bins along strike
                  igdx: the assigned along-strike bins
                  gdxp: the edges of the bins perpendicular to strike
                 igdxp: the assigned strike-perpendicular bins
                   gdy: the edges of the bins along dip
                  igdy: the assigned along-dip bins
                   gdz: the edges of the bins perpendicular to the fault
                  igdz: the assigned fault-perpendicular bins
                   gdd: the edges of the bins along depth
                  igdd: the assigned along-depth bins
        """

        # for each, find limits, divide, and sort the earthquakes into them
	
        lm=general.minmax(self.x)
        self.gdx=np.unique(np.append(np.arange(lm[0],lm[1],spc),lm[1]))
        self.igdx=np.searchsorted(self.gdx,self.x,'left')
        
        lm=general.minmax(self.xp)
        self.gdxp=np.unique(np.append(np.arange(lm[0],lm[1],spc),lm[1]))
        self.igdxp=np.searchsorted(self.gdxp,self.xp,'left')

        lm=general.minmax(self.y)
        self.gdy=np.unique(np.append(np.arange(lm[0],lm[1],spc),lm[1]))
        self.igdy=np.searchsorted(self.gdy,self.y,'left')

        lm=general.minmax(self.z)
        self.gdz=np.unique(np.append(np.arange(lm[0],lm[1],spc),lm[1]))
        self.igdz=np.searchsorted(self.gdz,self.z,'left')

        lm=general.minmax(self.depth)
        self.gdd=np.unique(np.append(np.arange(lm[0],lm[1],spc),lm[1]))
        self.igdd=np.searchsorted(self.gdd,self.depth,'left')

        # note the relevant boxes
        self.ibox=self.xpd2box()

        
    def box2xpd(self,ibox=None):
        """
        compute the bin numbers of each earthquake
        :param        ibox: the box for this earthquake
        :return       igdx: along-strike bin (default: self.igdx)
        :return      igdxp: fault-perpendicular bin (default: self.igdx)
        :return       igdd: depth bin (default: self.igdx)
        """

        # number of boxes
        Nx=self.gdx.size-1
        Np=self.gdxp.size-1
        Nd=self.gdd.size-1

        if ibox is None:
            ibox=self.ibox

        # depth bin
        igdd=ibox % Nd
        ibox=(ibox-igdd)/Nd

        # fault-perpendicular bin
        igdxp=ibox % Np
        igdx=(ibox-igdxp)/Np

        # make sure they're all integers
        igdd=np.round(igdd).astype(int)
        igdxp=np.round(igdxp).astype(int)
        igdx=np.round(igdx).astype(int)

        return igdx,igdxp,igdd
        
    def xpd2box(self,igdx=None,igdxp=None,igdd=None):
        """
        compute the box numbers of each earthquake
        :param       igdx: along-strike bin (default: self.igdx)
        :param      igdxp: fault-perpendicular bin (default: self.igdx)
        :param       igdd: depth bin (default: self.igdx)
        :return      ibox: the box for this earthquake
        """

        if igdx is None:
            igdx=self.igdx
        if igdxp is None:
            igdxp=self.igdxp
        if igdd is None:
            igdd=self.igdd

        # number of boxes
        Nx=self.gdx.size-1
        Np=self.gdxp.size-1
        Nd=self.gdd.size-1

        # identify the box numbers of all the reference earthquakes
        ibox=(Np*Nd)*igdx + \
            (Nd)*igdxp  + igdd

        return ibox
    
    
    def Figure_2(self,irefo,itaro,index=10):
        """
        Recrate figure 2 which plots all events in a group of co-located events
        :param      ifero: reference repeating earthquake
        :param      itaro: identified co-located event
        :param      index: index of induvidual repeating event to plot
        """
        plt.rcParams.update({'font.size': 60})
        irefo_repeat = np.array([irefo[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] + 0.3   >= self.mpref[itaro[x]]  >= self.mpref[irefo[x]] - 0.3 ])
        itaro_repeat = np.array([itaro[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] + 0.3   >= self.mpref[itaro[x]]  >= self.mpref[irefo[x]] - 0.3 ])
        irefo_partial = np.array([irefo[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])
        itaro_partial = np.array([itaro[x] for x in range(0,len(itaro)) if self.mpref[irefo[x]] - 0.3   > self.mpref[itaro[x]] ])

        #Getting rid of the ones that are just finding themselves 
        irefo_repeat_unique = irefo_repeat[itaro_repeat - irefo_repeat != 0 ]
        itaro_repeat_unique = itaro_repeat[itaro_repeat - irefo_repeat != 0 ]

        item = irefo_repeat_unique[index]
        tar = np.array(np.where(irefo_repeat_unique == item)[0])
        par = np.array(np.where(irefo_partial== item)[0])

        repeater_x = np.append(self.y[itaro_repeat_unique][tar], self.y[item], axis=None)
        repeater_time = np.append(self.time[itaro_repeat_unique][tar], self.time[item], axis=None)
        repeater_s = np.append(self.rad[itaro_repeat_unique][tar]/self.mpref[item], self.rad[item]/self.mpref[item], axis=None) 

        partial_x = self.y[itaro_partial][par]
        partial_time = self.time[itaro_partial][par]
        partial_s = self.rad[itaro_partial][par]/self.mpref[item]


        fig, ax = plt.subplots(figsize=(100, 6), dpi=72)
        s  = ((2 * repeater_s/15  * 72)**2 )


        time = []

        for t in repeater_time:

            time.append(np.datetime64(t))



        im = plt.scatter(time,repeater_x,s = s,zorder=10,facecolors='lightblue', edgecolors='lightblue')

        s  = ((2 * partial_s /15 * 72)**2 )
        time = []
        for t in partial_time:

            time.append(np.datetime64(t))
  

        im = plt.scatter(time,partial_x,s = s,zorder=10,facecolors='orange', edgecolors='orange')   

        plt.ylim(-1.6,-1)
        plt.yticks([])
        plt.ylabel('Distance along strike (m)')
        plt.show()
