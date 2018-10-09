### Python lib
import fitsio
import scipy as sp
import copy
import matplotlib.pyplot as plt

from . import utils
from . import constants

raw_dic_class = {
    "correlation"             : None,
    "f1"                      : None,
    "f2"                      : None,
    "lmin1"                   : None,
    "lmin2"                   : None,
    "path"                    : None,
    "title"                   : None,
}

class Correlation1D:

    def __init__(self,dic=None):

        if (dic is None):
            dic = copy.deepcopy(raw_dic_class)

        ### info from dic
        self._correlation = dic['correlation']
        self._f1 = dic["f1"]
        self._f2 = dic["f2"]
        self._lmin1 = dic["lmin1"]
        self._lmin2 = dic["lmin2"]
        self._title = dic["title"]

        ### List obsorber
        self._listAbs1 = None
        if not self._lmin1 is None:
            self._listAbs1 = {}
            for a, l in constants.absorber_IGM.items():
                if l>self._lmin1:
                    self._listAbs1[a] = l
        self._listAbs2 = None
        if not self._lmin2 is None:
            self._listAbs2 = {}
            for a, l in constants.absorber_IGM.items():
                if l>self._lmin2:
                    self._listAbs2[a] = l

        ###
        self._llmin = None
        self._llmax = None
        self._dll   = None
        self._n1d   = None
        self._var   = None
        self._cor   = None
        self._mat   = None
        self.read_from_do_cor(dic["path"])

        return

    def read_from_do_cor(self,path):

        vac = fitsio.FITS(path)
        head  = vac[1].read_header()
        self._llmin = head['LLMIN']
        self._llmax = head['LLMAX']
        self._dll   = head['DLL']
        self._n1d   = int((self._llmax-self._llmin)/self._dll+1)

        ### All Matrix
        self._mat = {}
        self._mat["DA"] = vac[2]['DA'][:]
        self._mat["WE"] = vac[2]['WE'][:]
        self._mat["NB"] = vac[2]['NB'][:]

        ### Variance
        self._var = sp.zeros( (self._n1d,4) )
        self._var[:,0] = 10.**( sp.arange(self._n1d)*self._dll+self._llmin )
        self._var[:,1] = sp.diag(self._mat["DA"])
        self._var[:,2] = sp.diag(self._mat["WE"])
        self._var[:,3] = sp.diag(self._mat["NB"])

        ### Correlation
        self._cor = sp.zeros( (self._n1d,4) )
        self._cor[:,0] = 10.**( sp.arange(self._n1d)*self._dll )

        inDown = False
        upperTri = sp.triu(self._mat["NB"])
        if (upperTri>0.).sum()==0:
            inDown=True
            self._cor[:,0] = self._cor[:,0][::-1]

        norm=1.
        if sp.trace(self._mat["NB"])>0:
            norm = sp.sum(sp.diag(self._mat["DA"],k=0)*sp.diag(self._mat["WE"],k=0))/sp.trace(self._mat["WE"],offset=0)

        for i in range(self._n1d):
            d = i
            if inDown:
                d = 1+i-self._n1d
            tda = sp.diag(self._mat["DA"],k=d)
            twe = sp.diag(self._mat["WE"],k=d)
            tnb = sp.trace(self._mat["NB"],offset=d)
            if tnb>0:
                self._cor[i,1] = sp.sum(tda*twe)/sp.sum(twe)/norm
                self._cor[i,2] = sp.sum(twe)/norm
                self._cor[i,3] = tnb

        vac.close()

        return
    def plot_var(self,other=None,redshiftLine=None):

        if not other is None:
            lst_corr = [self] + other

        for c in lst_corr:
            x = c._var[:,0]
            y = c._var[:,1]
            w = (c._var[:,2]>0.) & (c._var[:,3]>10)
            x = x[w]
            y = y[w]
            if x.size==0: continue
            if not redshiftLine is None:
                x = x/redshiftLine-1.
            if c._title is not None:
                plt.plot(x,y,linewidth=4,label=r"$"+c._title+"$")
            else:plt.plot(x,y,linewidth=4)


        if not redshiftLine is None:
            plt.xlabel(r'$z$',fontsize=30)
        else:
            plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\mathrm{\AA{}}]$',fontsize=30)
        plt.ylabel(r'$\sigma^{2}(\lambda_{\mathrm{Obs.}})$',fontsize=30)
        plt.legend(fontsize=20, numpoints=1,ncol=2, loc=1)
        plt.grid()
        plt.show()

        return
    def plot_cor(self,other=None,lines=False,lineToShow=None,redshiftLine=None):

        if not other is None:
            lst_corr = [self] + other

        ###
        minY = None
        maxY = None
        for c in lst_corr:
            x = c._cor[:,0]
            y = c._cor[:,1]
            w = (c._cor[:,2]>0.) & (c._cor[:,3]>10)
            x = x[w]
            y = y[w]
            if x.size==0: continue
            if not redshiftLine is None:
                x = utils.dist_lines_Obs(lObs1=redshiftLine[0],lObs2=redshiftLine[0]/x,lRF=redshiftLine[1])
            if c._title is not None:
                plt.plot(x,y,linewidth=4,label=r"$"+c._title+"$",marker='o')
            else:
                plt.plot(x,y,linewidth=4,marker='o')

            if minY is None:
                minY = y.min()
            else:
                minY = min(minY,y.min())
            if maxY is None:
                maxY = y.max()
            else:
                maxY = max(maxY,y.max())
            maxY = 0.3

        for c in lst_corr[:1]:

            x = c._cor[:,0]
            y = c._cor[:,1]
            w = (c._cor[:,2]>0.) & (c._cor[:,3]>10)
            x = x[w]
            y = y[w]
            if x.size==0: continue

            ### lines
            if lines and c._listAbs1 is not None:

                la1 = c._listAbs1
                if c._listAbs2 is not None:
                    la2 = c._listAbs2
                else:
                    la2 = c._listAbs1

                lst_lines = []
                for a1,l1 in la1.items():
                    for a2,l2 in la2.items():
                        if (a1==a2) or (a1+"__"+a2 in lst_lines) or (a2+"__"+a1 in lst_lines):
                            continue
                        if (lineToShow is not None) and (a1 not in lineToShow) and (a2 not in lineToShow):
                            continue
                        q = l1/l2
                        if q<x.min() or q>x.max(): q = 1./q
                        if q<x.min() or q>x.max(): continue
                        lst_lines += [a1+"__"+a2]
                        if not redshiftLine is None:
                            q = utils.dist_lines_Obs(lObs1=redshiftLine[0],lObs2=redshiftLine[0]/q,lRF=redshiftLine[1])
                        plt.plot( [q,q], [minY,maxY], color="black")
                        if l1<l2: name = a1+"\,/\,"+a2
                        else: name = a2+"\,/\,"+a1
                        plt.text( q, 0.95*maxY, s=r"$\mathrm{"+name+"}$", rotation='vertical', fontsize=15)

        if not redshiftLine is None:
            plt.xlabel(r'$\Delta r_{\parallel} \, [\mathrm{Mpc \, h^{-1}}]$',fontsize=30)
        else:
            plt.xlabel(r'$\lambda_{1}/\lambda_{2}$',fontsize=30)
        plt.ylabel(r'$\xi^{1D}(\lambda_{1}/\lambda_{2})/\sqrt{\xi^{1D}(\lambda_{1})\xi^{1D}(\lambda_{2})}$',fontsize=30)
        plt.legend(fontsize=20, numpoints=1,ncol=2, loc=1)
        plt.grid()
        #plt.tight_layout()
        #plt.savefig("fig.png")
        #plt.clf()
        plt.show()

        return
    def plot_mat(self):

        ###
        da = sp.copy(self._mat["DA"])
        if sp.trace(da)!=0.:
            da = utils.getCorrelationMatrix(da)
        w = (self._mat["WE"]>0.) & (self._mat["NB"]>10)
        da[ sp.logical_not(w) ] = sp.nan

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(da,origin="lower",interpolation='nearest')
        cbar = plt.colorbar()
        plt.grid(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.show()

        ###
        for k in range(5):
            x = sp.arange(sp.diag(self._mat["DA"],k=k).size)
            y = sp.diag(self._mat["DA"],k=k)
            w = (sp.diag(self._mat["WE"],k=k)>0.) & (sp.diag(self._mat["NB"],k=k)>10.)
            x = x[w]
            y = y[w]
            plt.plot(x,y,linewidth=4,alpha=0.7)
        plt.grid()
        plt.show()

        return
