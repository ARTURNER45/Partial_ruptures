import numpy as np
from math import log10,floor

def roundsigfigs(x,n):
    if isinstance(x,int) or isinstance(x,float):
        if x!=0:
            nr = int(floor(log10(abs(x))))
            x = round(x,n-nr-1)
    else:
        x = np.array([roundsigfigs(xi,n) for xi in x])

    return x

def masknans(x):
    """
    :param      x:  array or masked array
    :return     x:  masked array, with nans masked
    """

    if isinstance(x,np.ma.masked_array):
        x.mask = np.logical_or(x.mask,np.isnan(x))
    else:
        x = np.ma.masked_array(x,mask=np.isnan(x))

    return x

def excludesegs(tdes,texc,bfr=0.):
    """
    :parm      tdes: desired time intervals
    :param     texc: time intervals to exclude
    """

    tdes=np.atleast_2d(tdes)
    texc=np.atleast_2d(texc)

    # start by simplifying
    tdes = sortintosegs(tdes,bfr=bfr)
    tdes = noshortsegs(tdes,bfr=bfr)

    if tdes.shape[0]>0:
        # min and max
        texc[:,0] = np.maximum(texc[:,0],tdes[0,0])
        texc[:,1] = np.minimum(texc[:,1],tdes[-1,1])

        # start by simplifying
        texc = sortintosegs(texc,bfr=0.)
        texc = noshortsegs(texc,bfr=0.)

        # where the excluded intervals go
        ix = np.searchsorted(tdes[:,1],texc[:,0])
        t2 = np.insert(tdes[:,1],ix,texc[:,0])
        t1 = np.insert(tdes[:,0],np.minimum(ix+1,tdes.shape[0]),
                       texc[:,1])

        # the whole interval
        tdes = np.vstack([t1,t2]).transpose()

        # simplify if necessary
        tdes = sortintosegs(tdes,bfr=bfr)
        tdes = noshortsegs(tdes,bfr=bfr)
        
    return tdes


def noshortsegs(vls,bfr=0.):
    """
    :param    vls:   ? x 2 set of values
           to sort and delete overlaps
    :param    bfr:   how  much time to allow between segments
    :return   vls:   vls, but sorted and with overlaps deleted
    """

    # sort
    ixi = np.argsort(vls[:,0])
    vls = vls[ixi,:]
    vls = np.atleast_2d(vls)

    # identify intervals with no time in them
    ii, = np.where(vls[:,1]<=vls[:,0]+bfr)
    while len(ii):
        vls = np.delete(vls,ii[0],axis=0)
        ii, = np.where(vls[:,1]<=vls[:,0]+bfr)

    return vls


def sortintosegs(vls,bfr=0.):
    """
    :param    vls:   ? x 2 set of values
           to sort and delete overlaps
    :param    bfr:   how  much time to allow between segments
    :return   vls:   vls, but sorted and with overlaps deleted
    """

    # sort
    ixi = np.argsort(vls[:,0])
    vls = vls[ixi,:]
    vls = np.atleast_2d(vls)

    # identify overlaps
    ii, = np.where(vls[1:,0]<=vls[:-1,1]+bfr)
    while len(ii):
        # just one at a time
        ii = ii[0]
        vls[ii,1] = vls[ii+1,1]
        vls = np.delete(vls,ii+1,axis=0)
        
        # identify overlaps again
        ii, = np.where(vls[1:,0]<=vls[:-1,1]+bfr)

    return vls


def closest(xvals,x):
    """
    :param      xvals:    set of sorted values
    :param          x:    values of interest
    :return        ix:    index of closest value
    """

    xvals = np.atleast_1d(xvals)
    x = np.atleast_1d(x)

    # index before and after
    ix = np.searchsorted(xvals,x,'left')

    # in range
    ix = np.maximum(ix,1)
    ix = np.minimum(ix,len(xvals)-1)
    
    # before or after?
    dx1 = np.abs(xvals[ix-1]-x)
    dx2 = np.abs(xvals[ix]-x)
    dx1 = dx1<dx2

    ix[dx1] = ix[dx1]-1
    ix = np.maximum(ix,0)

    return ix

def minmax(x,bfr=1.):
    """
    :param      x:   set of values
    :param    bfr:   how much to multiply the limits by (default: 1.)
    :return   lms:   limits
    """

    # minmax
    lms = np.array([np.min(x),np.max(x)])

    if bfr!=1.:
        lms = np.mean(lms)+np.diff(lms)[0]*bfr*np.array([-.5,.5])

    return lms


def logsmooth(freq,fvl,fstd=1.3,logy=True,subsamp=False):
    """
    :param  freq: a set of frequencies or other x-values
    :param   fvl: a set of amplitude or other y-values
    :param  fstd: a factor variation to use as width
                    in log Gaussian
    :param  logy: average the log y-values (default: True)
    :param subsamp: subsample the output
    :return  fsm: smoothed values
    :return freqsm: frequencies for smoothed output
    """

    Ndim = fvl.ndim
    if fvl.ndim==1:
        fvl = fvl.reshape([fvl.size,1,1])
    elif fvl.ndim==2:
        fvl = fvl.reshape([fvl.shape[0],fvl.shape[1],1])

    # copy initial for output
    fsm = fvl.copy()
    
    # but otherwise avoid zero frequency
    iok = freq>0.
    freqo=freq.copy()
    freq,fvl=freq[iok],fvl[iok,:,:]
   
    # because we want to smooth the log values
    if logy:
        fvl = np.log(fvl)
    freq = np.log(freq)
    fstd = np.log(fstd)
    fvl = fvl.transpose([1,0,2])
    fsmh = fvl.copy()

    if subsamp:
        # desired frequencies
        fdes = np.ceil((freq[-1]-freq[0])/fstd*4)
        fdes = np.linspace(freq[0],freq[-1],int(fdes)+1)
        ixd = closest(freq,fdes)
        ixd = np.unique(ixd)
    else:
        # all frequencies
        ixd = np.arange(0,len(freq))

    for k in ixd:
        # weighting for this frequency
        fct = np.power((freq-freq[k])/fstd,2)
        fct = np.exp(-fct)
        fct = fct/np.sum(fct)

        # but only use some fraction
        ilm=np.interp([0.001,0.999],np.cumsum(fct),
                      np.arange(0,fct.size))
        i1=int(np.floor(ilm[0]))-4
        i2=int(np.ceil(ilm[1]))+4
        i1=np.maximum(i1,0)
        i2=np.minimum(i2,freq.size)

        # change weighting to be linear in log space
        wgt=np.divide(1.,np.diff(freq[i1:i2]))
        wgt=(np.append(wgt,wgt[-1])+np.append(wgt[0:1],wgt))/2.
        wgt=np.multiply(wgt,fct[i1:i2])
        wgt=wgt/np.sum(wgt)

        
        fsmh[:,k,:]=np.matmul(wgt,fvl[:,i1:i2,:])
        
    # back to linear domain
    if logy:
        fsmh = np.exp(fsmh)

    # save and output
    fsmh=fsmh.transpose([1,0,2])
    fsm[iok,:,:] = fsmh

    # get the selection
    iokd,=np.where(iok)
    iok,=np.where(~iok)
    ixd = np.append(iok,iokd[ixd])
    ixd = np.unique(ixd)
    ixd.sort()
    
    fsm = fsm[ixd,:,:]
    freqsm = freqo[ixd]

    if Ndim==1:
        fsm = fsm.flatten()
    elif Ndim==2:
        fsm = fsm.reshape([fsm.shape[0],fsm.shape[1]])

    return fsm,freqsm


def tomatrix(xy,vls):
    """
    :param    xy: N x 2 array of x-y values for each row
    :param   vls: N x ? array of values to put in matrix
    """
    
    # need a matrix
    if vls.ndim==1:
        vls=vls.reshape([vls.size,1])

    # needs to be 2-D
    sz = list(vls.shape)
    sz2 = [sz[0],np.prod(sz[1:])]
    if vls.ndim>2:
        vls=vls.reshape(sz2)

    # all the values
    x=np.unique(xy[:,0])
    y=np.unique(xy[:,1])
    
    # which indices
    ix=closest(x,xy[:,0])
    iy=closest(y,xy[:,1])
    ii=ix+iy*x.size

    # create grid
    grd=np.ma.masked_array(np.ndarray([x.size*y.size,sz2[1]]),
                           mask=True)
    grd.mask[ii,:]=False
    grd[ii,:]=vls

    # to gridded shape
    grd=grd.reshape([y.size,x.size]+sz[1:])

    return grd,x,y

def modhalf(a,b):
    """
    :param   a:  numerator
    :param   b:  denominator
    :return  x:  a % b, but always between -b/2 and b/2
    """

    x = a % b
    x[x>b/2.]=x[x>b/2.]-b

    return x
    
