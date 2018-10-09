### Python lib
from __future__ import print_function
import scipy as sp
import scipy.stats
import copy
import matplotlib.pyplot as plt
import h5py
import os.path

import utils
import constants

raw_dic_class = {
    'path' :'',
    'title':''
}

class Fit:

    def __init__(self,dic=None):

        if (dic is None):
            dic = copy.deepcopy(raw_dic_class)

        self._title = dic['title']
        self.read_fit_results(dic['path'])

        return

    def read_fit_results(self,path):

        path = os.path.expandvars(path)
        f = h5py.File(path,'r')

        ### Parameters
        self._param    = {}
        self._fitAtrrs = {}
        self._fit      = {}
        lst_forFit = ['cov[','ndata', 'npar', 'list of free pars', 'list of fixed pars',
                    'hesse_failed', 'has_reached_call_limit', 'has_accurate_covar', 'has_posdef_covar',
                    'up', 'fval', 'is_valid', 'is_above_max_edm', 'has_covariance', 'has_made_posdef_covar',
                    'has_valid_parameters', 'edm', 'nfcn', 'zeff']
        for el in f['best fit'].attrs:
            if any( str(ell) in el for ell in lst_forFit):
                if str(el)=='list of free pars' or str(el)=='list of fixed pars':
                    self._fitAtrrs[str(el)]=sp.array([ ell for ell in f['best fit'].attrs[el]]).astype(str)
                else:
                    self._fitAtrrs[str(el)]=f['best fit'].attrs[el]
            else:
                self._param[str(el)] = {}
                self._param[str(el)]['value'] = f['best fit'].attrs[el][0]
                self._param[str(el)]['error'] = f['best fit'].attrs[el][1]
                if str(el) in list(constants.latex_name.keys()):
                    self._param[str(el)]['name'] = constants.latex_name[str(el)]
                else:
                    self._param[str(el)]['name'] = str(el)

        ### Set errors to zero for unfitted param
        for el in self._fitAtrrs['list of fixed pars']:
            self._param[el]['error'] = 0.

        ### Convert from bias*f/beta to bias
        #for p in list(self._param.keys()):
        #    if len(p)>5 and p[:5]=='bias_':
        #        if self._param['beta_'+p[5:]]['error']!=0. or self._param['growth_rate']['error']!=0.:
        #            print("Can not correct bias measurement")
        #            sys.exit()
        #        coef = self._param['growth_rate']['value']/self._param['beta_'+p[5:]]['value']
        #        self._param[p]['value'] *= coef
        #        self._param[p]['error'] *= coef
        #        print(p, self._param[p]['value'], self._param[p]['error'])

        ### Set proba
        self._fitAtrrs['proba'] = 1.-sp.stats.chi2.cdf(self._fitAtrrs['fval'],self._fitAtrrs['ndata']-self._fitAtrrs['npar'])

        ### Best fit
        self._data = {}
        for d in f.keys():
            if d in ['best fit','fast mc','minos','chi2 scan']: continue

            dic = {}
            for item, value in f[d].attrs.items():
                dic[str(item)] = value
            dic['fit'] = f[d]['fit'].value
            self._data[str(d)] = dic

        ### minos
        if 'minos' in [ el for el in list(f.keys())]:
            self.minos_sigma = f['minos'].attrs.values()
            self.minos = {}
            for p in f['minos'].keys():
                dic = {}
                for item, value in f['minos'][p].attrs.items():
                    dic[str(item)] = value
                self.minos[str(p)] = dic

        ### chi2 scan
        if 'chi2 scan' in [ el for el in list(f.keys())]:
            self.chi2scan = {}
            self.chi2scan_result = {}
            for p in f['chi2 scan'].keys():
                if p!='result':
                    dic = {}
                    for item, value in f['chi2 scan'][p].attrs.items():
                        dic[str(item)] = value
                    self.chi2scan[str(p)] = dic
                else:
                    self.chi2scan_result['parameters'] = {}
                    for item, value in f['chi2 scan'][p].attrs.items():
                        self.chi2scan_result['parameters'][str(item)] = value
                    self.chi2scan_result['values'] = f['chi2 scan']['result']['values'].value

        ### fast mc
        if 'fast mc' in [ el for el in list(f.keys())]:
            self.fastmc = {}
            self.fastmc['niterations'] = f['fast mc'].attrs['niterations']
            self.fastmc['seed'] = f['fast mc'].attrs['seed']
            self.fastmc['covscaling'] = f['fast mc'].attrs['covscaling']
            self.fastmc['chi2'] = f['fast mc/chi2'].value
            for p in self._param:
                strp = str(p)
                self.fastmc[strp] = {}
                self.fastmc[strp]['values'] = f['fast mc/'+p+'/values'].value
                self.fastmc[strp]['errors'] = f['fast mc/'+p+'/errors'].value
                self.fastmc[strp]['expected'] = self._param[strp]['value']
                self.fastmc[strp]['fixed'] = strp in self._fitAtrrs['list of fixed pars']
                if self.fastmc[strp]['fixed']:
                    self.fastmc[strp]['errors'][:] = 0.
            for p in f['fast mc'].attrs['list of fiducial pars']:
                self.fastmc[str(p)]['expected'] = float(f['fast mc'].attrs['fiducial['+p+']'][0])
                self.fastmc[str(p)]['fixed'] = f['fast mc'].attrs['fiducial['+p+']'][1]=='fixed'
                if self.fastmc[str(p)]['fixed']:
                    self.fastmc[str(p)]['errors'][:] = 0.
        f.close()

        return
    def plot_chi2scan(self,deltachi2=True,sigmas=True):

        if deltachi2:
            zlabel = '\Delta \chi^{2}'
        else:
            zlabel = '\chi^{2}'

        dim = len(self.chi2scan)
        parameters = self.chi2scan_result['parameters']
        values     = self.chi2scan_result['values']

        if dim==1:
            par = self.chi2scan.keys()[0]
            xxx = values[:,parameters[par]]
            zzz = values[:,parameters['fval']]
            if deltachi2:
                zzz -= self._fitAtrrs['fval']

            plt.plot(xxx,zzz,linewidth=4)
            if sigmas:
                for i in range(1,4):
                    plt.plot(xxx,i**2*sp.ones(xxx.size),'--',linewidth=4,color='grey')
            plt.xlabel(r'$'+constants.latex_name[par]+'$',fontsize=30)
            plt.ylabel(r'$'+zlabel+'$',fontsize=30)
            plt.grid()
            plt.show()
        elif dim==2:
            par1 = self.chi2scan.keys()[0]
            par2 = self.chi2scan.keys()[1]
            nb1  = self.chi2scan[par1]['nb_bin']
            nb2  = self.chi2scan[par2]['nb_bin']
            zzz = values[:,parameters['fval']]
            zzz = utils.convert1DTo2D(zzz,nb1,nb2)
            extent = [self.chi2scan[par2]['min'],self.chi2scan[par2]['max'],self.chi2scan[par1]['min'],self.chi2scan[par1]['max']]
            if deltachi2:
                zzz -= self._fitAtrrs['fval']

            plt.imshow(zzz,extent=extent,origin='lower',interpolation='nearest')
            cbar = plt.colorbar()
            plt.xlabel(r'$'+constants.latex_name[par2]+'$', fontsize=30)
            plt.ylabel(r'$'+constants.latex_name[par1]+'$', fontsize=30)
            cbar.set_label(r'$'+zlabel+'$',size=30)
            plt.grid(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            plt.show()

        return
    def plot_fastMC(self, par):

        ###
        if par=='chi2':
            value = self.fastmc['chi2']
            errors = sp.zeros(value.size)
            expected = self._fitAtrrs['ndata']-self._fitAtrrs['npar']
            pull = sp.zeros(value.size)
            data = self._fitAtrrs['fval']
            error_data = sp.zeros(value.size)
            pull_data = sp.zeros(value.size)
            name = '\\chi^{2}'
        else:
            value = self.fastmc[par]['values']
            errors = self.fastmc[par]['errors']
            expected = float(self.fastmc[par]['expected'])
            pull = (value-expected)/errors
            data = self._param[par]['value']
            error_data = self._param[par]['error']
            pull_data = (data-expected)/error_data
            name = self._param[par]['name']

        ###
        print(' parameter = ', name)
        print(' expected  = ', expected)
        print(' mean      = ', value.mean() )
        print(' variance  = ', value.var(ddof=1) )
        print(' error     = ', value.var()/sp.sqrt(value.size-1) )
        print(' var with respect to expected = ', sp.sqrt( sp.mean((value-expected)**2 )) )
        print(' mean pull      = ', pull.mean() )
        print(' variance pull  = ', pull.var(ddof=1) )

        ### histo value
        plt.hist(value,bins=10)
        plt.plot([expected,expected],[0.,value.size], color='red',linewidth=4)
        plt.plot([data,data],[0.,value.size],'--', color='black',linewidth=4)
        plt.xlabel(r'$'+name+'$',fontsize=20)
        plt.ylabel(r'$\#$',fontsize=20)
        plt.grid()
        plt.show()

        ### histo pull
        plt.hist(pull,bins=10)
        plt.plot([0.,0.],[0.,value.size], color='red',linewidth=4)
        plt.plot([pull_data,pull_data],[0.,value.size],'--', color='black',linewidth=4)
        plt.xlabel(r'$('+name+'-exp)/err$',fontsize=20)
        plt.ylabel(r'$\#$',fontsize=20)
        plt.grid()
        plt.show()

        ### histo error
        plt.hist(errors,bins=10, histtype='step', label=r'$\mathrm{'+self._title+'}$',color='blue')
        plt.plot([errors.mean(),errors.mean()],[0.,value.size],'--', color='blue',linewidth=2)
        plt.plot([error_data,error_data],[0.,value.size],'--', color='black',linewidth=2,label=r'$\mathrm{Data}$')
        plt.xlabel(r'$\sigma('+name+')$',fontsize=40)
        plt.ylabel(r'$\#$',fontsize=40)
        plt.legend(fontsize=40)
        plt.grid()
        plt.show()

        return
    def print_fitted_par(self,lst=None,coeffBias=1.,header=True,latex=False,redshift=False):

        if not latex:
            deb  = ' || '
            end  = ' || '
            sep  = ' || '
            pm   = ' +/- '
            math = ''
        else:
            deb  = ''
            end  = ' \\\\ '
            sep  = ' & '
            pm   = ' \pm '
            math = '$'

        if lst is None:
            lst = self._fitAtrrs['list of free pars']

        if header:
            ###
            to_print0  = deb + ''.ljust(20) + sep
            to_print1  = deb + ''.ljust(20) + sep
            if redshift:
                to_print0 += sep+math+' z_{\mathrm{eff}} '+math+sep
                to_print1 += sep+sep
            for p in lst:
                to_print0 += self._param[p]['name'].ljust(20)
                to_print0 += sep
                to_print1 += ''.ljust(20)
                to_print1 += sep
            to_print0 += ''.ljust(20) + end
            to_print1 += ''.ljust(20) + end
            print(to_print0)
            print(to_print1)
        ###
        to_print  = deb + math+self._title.ljust(20)+math + sep

        val = utils.format_number_with_precision(self._fitAtrrs['zeff'],0.1)
        to_print += math+ val +math+sep

        for p in lst:

            if p in self._param.keys():
                val = self._param[p]['value']
                err = self._param[p]['error']
            else:
                val = ''
                err = 0.

            if len(p)>len('bias') and p[:4]=='bias' and p in self._param.keys():
                val *= coeffBias
                err *= coeffBias

            if err==0.:
                val = str(val)
                err = str(err)
            else:
                val = utils.format_number_with_precision(val,err)
                err = utils.format_number_with_precision(err,err)

            to_print += (math+val+pm+err+math).ljust(20)
            to_print += sep

        val = self._fitAtrrs['fval']
        err = 0.1
        val = utils.format_number_with_precision(val,err)
        nbBin   = str(self._fitAtrrs['ndata'])
        nbParam = str(self._fitAtrrs['npar'])
        proba = self._fitAtrrs['proba']
        if proba==0.:
            proba = '0'
        else:
            proba = utils.format_number_with_precision(proba,proba)
        s       = math+val + ' / (' + nbBin + '-' + nbParam + '),  p = ' + proba+math
        to_print  += s.ljust(20) + end

        print(to_print)

        return
    def print_contribution_chi2(self):

        def print_chi2(chi2,nbBin,nbParam):

            val = chi2
            err = 0.1
            val = utils.format_number_with_precision(val,err)
            proba = 1.-sp.stats.chi2.cdf(chi2,nbBin-nbParam)
            proba = utils.format_number_with_precision(proba,proba)
            s = val + ' / (' + str(nbBin) + '-' + str(nbParam) + '),  p = ' + proba

            return s

        data = self._fitAtrrs
        npar = data['npar']
        chi2 = print_chi2(data['fval'],data['ndata'],npar)
        print('all',chi2)
        for d in sorted(self._data.keys()):
            data = self._data[d]
            chi2 = print_chi2(data['chi2'],data['ndata'],npar)
            print(d,chi2)

        return
