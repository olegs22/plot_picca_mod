### Python lib
import fitsio
import scipy as sp
import copy
import matplotlib.pyplot as plt
import scipy.constants
from . import utils, constants

raw_dic_class = {
    "correlation"             : "",
    "f1"                      : "",
    "f2"                      : "",
    "o1"                      : "",
    "o2"                      : "",
    "l1"                      : "",
    "l2"                      : "",
    "path"                    : None,
    "title"                   : "",
    "nside"                   : None,
}

class Correlation3D_angl:

    def __init__(self,dic=None):

        if (dic is None):
            dic = copy.deepcopy(raw_dic_class)

        ### info from dic
        self._correlation = dic['correlation']
        self._f1 = dic["f1"]
        self._f2 = dic["f2"]
        self._o1 = dic["o1"]
        self._o2 = dic["o2"]
        self._l1 = dic["l1"]
        self._l2 = dic["l2"]
        self._title = dic["title"]
        if "nside" not in dic.keys():
            dic["nside"] = 16
        self._nside = dic["nside"]

        ### bin size (only square)
        self._rpmin    = None
        self._rpmax    = None
        self._binSizeP = None
        self._rtmin    = None
        self._rtmax    = None
        self._binSizeT = None

        ### Grid
        self._nt = None
        self._np = None
        self._rp = None
        self._rt = None
        self._r  = None
        self._z  = None
        self._nb = None
        self._we = None
        self._co = None
        self._dm = None

        ### Correlation
        self._da = None
        self._er = None


        self.read_from_do_cor(dic["path"])

        return

    def __add__(self,other):

        self._rp = self._rp*self._we
        self._rt = self._rt*self._we
        self._z  = self._z*self._we
        self._da = self._da*self._we

        self._nb += other._nb
        self._we += other._we
        self._rp += other._rp*other._we
        self._rt += other._rt*other._we
        self._z  += other._z*other._we
        self._da += other._da*other._we

        cut = (self._we > 0.)
        self._rp[cut] /= self._we[cut]
        self._rt[cut] /= self._we[cut]
        self._z[cut]  /= self._we[cut]
        self._da[cut] /= self._we[cut]

        return self
    def __sub__(self,other):

        assert(self._da.size==other._da.size)
        self._da -= other._da

        return self

    def multiply(self,scalar):
        self._da *= scalar

        return

    def read_from_do_cor(self,path):

        vac = fitsio.FITS(path)

        head = vac[1].read_header()

        ### Grid
        self._nt = head['NT']
        self._np = head['NP']
        self._rt_min = 0.
        self._rt_max = head['RTMAX']
        self._rp_min = head['RPMIN']
        self._rp_max = head['RPMAX']
        self._binSizeP = (self._rp_max-self._rp_min) / self._np
        self._binSizeT = (self._rt_max-self._rt_min) / self._nt

        self._rp = vac[1]['RP'][:]
        self._rt = vac[1]['RT'][:]
        self._z  = vac[1]['Z'][:]
        self._nb = vac[1]['NB'][:]

        ### Correlation
        we  = vac[2]['WE'][:]
        da  = vac[2]['DA'][:]
        self._we = we.sum(axis=0)
        cut = (self._we>0.)
        self._da       = (da*we).sum(axis=0)
        self._da[cut] /= self._we[cut]

        vac.close()

        return
    def plot_2d(self,log=False):
        crt = 1./scipy.constants.degree

        if ((self._we>0.).sum()==0):
            print("no data")
            return

        origin='lower'
        extent=[crt*self._rt_min, crt*self._rt_max, self._rp_min, self._rp_max]
        if (self._correlation=='o_f' or self._correlation=='f_f2'):
            origin='upper'
            extent=[crt*self._rt_min, crt*self._rt_max, self._rp_max, self._rp_min]

        yyy = sp.copy(self._da)
        w = (self._we>0.) & (self._nb>10.)
        yyy[sp.logical_not(w)] = float('nan')
        if log:
            yyy[w] = sp.log10( sp.absolute(yyy[w]))
        yyy = utils.convert1DTo2D(yyy,self._np,self._nt)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks([ i for i in sp.arange(crt*self._rt_min, crt*self._rt_max,crt*self._binSizeT*10) ])
        ax.set_yticks([ i for i in sp.arange(self._rp_min, self._rp_max,self._binSizeP*10) ])

        plt.imshow(yyy, origin=origin, extent=extent, interpolation='nearest', aspect='auto')
        cbar = plt.colorbar()

        if not log:
            cbar.set_label(r'$\xi(\lambda_{1}/\lambda_{2},\theta)$',size=40)
        else:
            cbar.set_label(r'$ \log10 \, |\xi(\lambda_{1}/\lambda_{2},\theta)| $',size=40)

        plt.xlabel(r'$\theta \, [\mathrm{deg}]$', fontsize=40)
        plt.ylabel(r'$\lambda_{1}/\lambda_{2}$', fontsize=40)
        plt.grid(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        plt.show()

        return
    def plot_slice_2d(self,sliceX=None,sliceY=None, other=None,coefX=1.):

        crt = 1./scipy.constants.degree
        if not other is None:
            list_corr = [self] + other

        for el in list_corr:

            if (sliceX is not None):
                cut  = (el._rt>=el._rt_min+el._binSizeT*sliceX) & (el._rt<el._rt_min+el._binSizeT*(sliceX+1))
                cut &= (el._we>0.)
                cut &= (el._nb>10)
                xxx  = el._rp[cut]
            if (sliceY is not None):
                cut  = (el._rp>=el._rp_min+el._binSizeP*sliceY) & (el._rp<el._rp_min+el._binSizeP*(sliceY+1))
                cut &= (el._we>0.)
                cut &= (el._nb>10)
                xxx  = crt*el._rt[cut]
            yyy = el._da[cut]
            if not el._er is None:
                yer = el._er[cut]

            if el._er is None:
                plt.errorbar(coefX*xxx,yyy,linewidth=4,label=r'$'+el._title+'$')
            else:
                plt.errorbar(coefX*xxx,yyy,yerr=yer,linewidth=4,label=r'$'+el._title+'$')
            #minY = el._rp_min+el._binSizeP*sliceY
            #maxY = el._rp_min+el._binSizeP*(sliceY+1)
            #print str(minY)+" < \lambda_{1}/\lambda_{2} < "+str(maxY)

        if (sliceX is not None):
            minX = el._rt_min+el._binSizeT*sliceX
            maxX = el._rt_min+el._binSizeT*(sliceX+1)
            plt.title(r"$"+str(minX)+" < \\theta < "+str(maxX)+"$",fontsize=30)
            plt.xlabel(r'$\lambda_{1}/\lambda_{2}$',fontsize=30)
        if (sliceY is not None):
            #minY = el._rp_min+el._binSizeP*sliceY
            #maxY = el._rp_min+el._binSizeP*(sliceY+1)
            #plt.title(r"$"+str(minY)+" < \lambda_{1}/\lambda_{2} < "+str(maxY)+"$",fontsize=30)
            plt.xlabel(r'$\theta \, [\mathrm{deg}]$',fontsize=30)

        for l in list(constants.absorber_IGM.keys()):
            l = constants.absorber_IGM[l]
            plt.plot( [l,l], [-1.,1.], color='black' )

        plt.ylabel(r'$\xi$',fontsize=30)
        #plt.legend(fontsize=20, numpoints=1,ncol=2, loc=1)
        plt.grid()
        plt.show()

        return
