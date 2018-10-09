import scipy as sp
import scipy.constants
import constants


def croom(x):
    '''
        Croom et al. Law
    '''

    r = 0.53 + 0.289*(1.+x)**2

    return r
def bias_vs_z_std(z, zref, alpha):

    r = ((1.+z)/(1+zref))**alpha

    return r
def growthStructure(z, omega_M_0, unnormed=False):

    omega_m = 1./(1.+(1.-omega_M_0)/(omega_M_0*(1.+z)**3.))
    omega_l = 1.-omega_m
    a = 1./(1.+z)

    if unnormed:
        norm = 1.
    else:
        norm = 1./growthStructure(0.,omega_M_0, unnormed=True)
    g = norm*(5./2.)*a*omega_m/( omega_m**(4./7.)-omega_l+(1.+omega_m/2.)*(1.+omega_l/70.) )

    return g
def growthStructureSimple(z, omega_M_0):

    g  = 1./(1.+z)
    g /= 1./(1.+100)/growthStructure(100, omega_M_0)

    return g
def growthRateStructure(z, omega_M_0):

    omega_m = omega_M_0*(1.+z)**3 / ( omega_M_0*(1.+z)**3+(1.-omega_M_0))
    f = sp.power(omega_m,0.55)

    return f

def convert1DTo2D(array1D,nbX,nbY):
    '''
        convert a 1D array to a 2D array
    '''

    array2D = sp.zeros(shape=(nbX,nbY))

    for k in range(array1D.size):
        i = k//nbY
        j = k%nbY

        array2D[i][j] = array1D[k]

    return array2D
def getCorrelationMatrix(cov):
    '''
        Get the correlation matrix from a covaraince matrix
    '''

    ### Get normalisation factor
    invSqrtDiag = sp.zeros(sp.diag(cov).size)
    w = (sp.diag(cov)>0.)
    invSqrtDiag[w] = 1./sp.sqrt(sp.diag(cov)[w])

    ### Normalize
    cor = sp.array(cov)
    for i in range(cov[:,0].size):
        cor[:,i] *= invSqrtDiag[i]
        cor[i,:] *= invSqrtDiag[i]
        if invSqrtDiag[i]>0.: cor[i,i] = 1.

    return cor
def get_precision(error,nb_diggit=2):

    precision = int( nb_diggit -1 -sp.floor( sp.log10(error) ) )

    return precision
def precision_and_scale(x):

    '''
    http://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
    '''

    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(sp.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)

    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(sp.log10(frac_digits))

    return (magnitude + scale, scale)
def format_number_with_precision(number,error,number_of_digit=2):

    precision = get_precision(error,number_of_digit)

    string = round(number,precision)
    digit  = precision_and_scale(string)
    string = str(string).ljust(digit[0]-digit[1]+1+precision,'0')

    return string
def dist_lines_Obs(lObs1,lObs2,lRF):

    z1 = lObs1/lRF-1.
    z2 = lObs2/lRF-1.
    z  = (z1+z2)/2.
    dz = z1-z2

    c = scipy.constants.speed_of_light/1000.
    hz = 100*sp.sqrt( constants.omegaM0*(1.+z)**3 + (1.-constants.omegaM0) )
    deltaRP = c*dz/hz

    return deltaRP
def dist_lines_RF(lRF1,lRF2,lObs):

    z1 = lObs/lRF1-1.
    z2 = lObs/lRF2-1.
    z  = (z1+z2)/2.
    dz = z1-z2

    c = scipy.constants.speed_of_light/1000.
    hz = 100.*sp.sqrt( constants.omegaM0*(1.+z)**3 + (1.-constants.omegaM0) )
    deltaRP = c*dz/hz

    return deltaRP
