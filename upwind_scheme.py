import numpy as np
import numexpr as ne
import math as mth
from numba import jit


### simple 1D upwind scheme to model the mid-plane of a circumstellar disc


### GRID and FIELD CONSTRUCTION #####

class grid:
    
    def __init__(self):
        
        self.R = []
        
class field:
    
    def __init__(self):
        
        self.cs = []

def build_grid(Rin,Rmax,NR,Ndust):
    
    agrid = grid()
    bgrid = grid()
    
    ## log spaced grid with two ghost cells
    ## cyclindrical grid (z,R,phi) ala zeus
    Ra = np.linspace(np.log10(Rin),np.log10(Rmax),NR+1)
    dRa_log = np.diff(Ra[:2])
    Ra = (np.append(Ra[0]-dRa_log,np.append(Ra,Ra[-1]+dRa_log)))
    Ra = np.append(Ra[0]-dRa_log,np.append(Ra,Ra[-1]+dRa_log))
    Rb = Ra[:-1] + dRa_log/2.

    agrid.R = 10.**Ra
    agrid.dR = np.append(np.diff(agrid.R),0.)
    agrid.dvR = np.append(np.diff(agrid.R**2.),0.)/2.
    agrid.dvRT = np.append(np.diff(agrid.R**3.),0.)/3. # 1/R^2d/dR required for (div (Tensor))dot e_phi
    agrid.g2 = np.ones(NR+5)
    agrid.g31 = np.ones(NR+5)
    agrid.g32 = agrid.R
    agrid.NR = NR
    agrid.ii = 2
    agrid.io = NR+1
    
    
    bgrid.R = 10.**Rb
    bgrid.dR = np.append(0.,np.diff(bgrid.R))
    bgrid.dvR = np.append (0.,np.diff(bgrid.R**2.))/2.
    bgrid.g2 = np.ones(NR+4)
    bgrid.g31 = np.ones(NR+4)
    bgrid.g32 = bgrid.R
    bgrid.NR = NR
    bgrid.ii = 2
    bgrid.io = NR+1
    
    ## fields
    
    fields = field()
    
    fields.agr = np.zeros(NR+5)
    fields.uR = np.zeros(NR+5)
    fields.uphi = np.zeros(NR+4)
    fields.Sigma_g = np.zeros(NR+4)
    fields.cs = np.zeros(NR+4)
    fields.Pmid = np.zeros(NR+4)
    fields.Omega_Kb = np.zeros(NR+4) # Keplerian angular velocity b grid
    fields.Omega_Ka = np.zeros(NR+5) # Keplerian angular velocity a grid
    fields.alpha = np.zeros(NR+4) # viscous alpha 
    fields.GM = 1.
    fields.Ndust = Ndust

    if (Ndust > 0):
        fields.Sigma_dust = np.zeros((NR+4,Ndust))
        fields.uR_dust = np.zeros((NR+5,Ndust))
        fields.uR_dust_diff = np.zeros((NR+5,Ndust))
        fields.uphi_dust = np.zeros((NR+4,Ndust))
        fields.tstop_a = np.zeros((NR+5,Ndust))
        fields.sizes = np.zeros(Ndust)
        fields.rho_int = np.zeros(Ndust)
        fields.Sc = np.zeros(Ndust) # scmidt number for dust species
    
    return agrid, bgrid, fields

def get_mid_plane_values(fields):
    
    cs = fields.cs
    Sigma = fields.Sigma_g

    Omega_K = fields.Omega_Kb

    fields.Pmid = ne.evaluate("cs * Sigma * Omega_K")
    fields.H    = ne.evaluate("cs / Omega_K")

@jit(nopython=True)
def source_update(GM,ra,drb,g32a,Pmid,ur,uphi,H,Sigma,dt,ii,io,NR):
    
    # body forces
    agr = np.zeros(NR+5)

    for i in range(ii,io+2):
        # centrifugal force, then gravity, then mid-plane pressure
        ## need gas body acceleration seperately to return for dust update
        agr[i] = (0.25*(uphi[i]+uphi[i-1])**2./g32a[i] - GM/ra[i]**2.-((H[i]+H[i-1])/(Sigma[i]+Sigma[i-1]))*(Pmid[i]-Pmid[i-1])/drb[i])
        ur[i] += dt*agr[i]
        
    # effective viscosity
    Ceff=3.
    q2 = np.zeros(NR+5)
    for i in range(ii-1,io+2):
        if  ( (ur[i+1]-ur[i]) < 0.):
            q2[i] = Ceff * Sigma[i] * (ur[i+1]-ur[i])**2.
    
    for i in range(ii,io+2):
        ur[i] += -dt* (q2[i]-q2[i-1])/drb[i] * (2./(Sigma[i]+Sigma[i-1]))
        

    return ur, agr

@jit(nopython = True)
def dust_source_update(GM,ra,g32a,vphi_g,vphi_d,vr_gas,ag_r,vrdust,Omega_a,Omega_b,S_gas,size,rho_int,ii,io,NR,Ndust,dt):

    # use the semi-implicit scheme of Rosotti et al. (2016)
    tstop_a = np.zeros((NR+5,Ndust))
    tstop_b = np.zeros((NR+4,Ndust))
    vr_dust_new = np.zeros((NR+5,Ndust))
    vphi_dust_new = np.zeros((NR+4,Ndust))
    # find stopping time on a grid
    for j in range(Ndust):
        for i in range(ii,io+2):

            tstop_a[i,j] = mth.pi * rho_int[j] * size[j] / (S_gas[i]+S_gas[i-1]) / Omega_a[i]
            tstop_b[i,j] = mth.pi * rho_int[j] * size[j] / (2.*S_gas[i]) / Omega_b[i]

    ## update radial velocity

    for j in range(Ndust):
        for i in range(ii,io+2):
            vr_gas_old = vr_gas[i] - ag_r[i]*dt # gas velocity at start of timestep
            adust_r = 0.25*(vphi_d[i,j]+vphi_d[i-1,j])**2./g32a[i] - GM/ra[i]**2.

            vr_dust_new[i,j] = (vrdust[i,j] * mth.exp(-dt/tstop_a[i,j]) + ag_r[i] * dt -
                                (vr_gas_old+(adust_r - ag_r[i])*tstop_a[i,j])*mth.expm1(-dt/tstop_a[i,j]))

    ## update azimuthal velocity
    for j in range(Ndust):
        for i in range(ii,io+1):
            # uphi for the gas does not update during it's source update, adust and agas = 0 in phi direction
            vphi_dust_new[i,j] = vphi_d[i,j]*mth.exp(-dt/tstop_b[i,j]) - vphi_g[i]*mth.expm1(-dt/tstop_b[i,j])

    return vr_dust_new, vphi_dust_new


@jit(nopython = True)
def disc_viscosity_update(pure_alpha,alpha,cs,Omega_Ka,uphi,ra,rb,drb,dvTa,Sigma,dt,ii,io,NR,pl_index):
    ## does update of angular velocity due to viscosity

    # if pure_alpha is true then use alpha model
    # if false use alpha model at first cell and then extropolate as power-law with pl_index

    Diff_a = np.zeros(NR+5) # diffusion constant on a grid (nu*Sigma*R)

    for i in range(ii,io+2):
        if (pure_alpha):
            Diff_a[i] = (alpha[i]*cs[i]**2.*Sigma[i]+alpha[i-1]*cs[i-1]**2.*Sigma[i-1])*ra[i]/2./Omega_Ka[i]
        else:
            Diff_a[i] = alpha[ii]*cs[ii]**2./Omega_Ka[ii] * (ra[i]/rb[ii])**pl_index * (Sigma[i]+Sigma[i-1])/2. * ra[i]

    for i in range(ii,io+1):
        dOmega_dR_p = (uphi[i+1]/rb[i+1]-uphi[i]/rb[i])/drb[i+1]
        dOmega_dR_m = (uphi[i]/rb[i]-uphi[i-1]/rb[i-1])/drb[i]
        uphi[i] += dt/Sigma[i] * (ra[i+1]**2.*Diff_a[i+1]*dOmega_dR_p-ra[i]**2.*Diff_a[i]*dOmega_dR_m) / dvTa[i]

    return uphi

@jit(nopython=True)
def transport_update(dRa,dva,dRb,dvb,g32a,g32b,Sigma,ur,uphi,dt,ii,io,NR):
    
    # new values
    Sigma_new = np.zeros(NR+4)

    ## density
    # van-leer upwinded value of density
    den_star = get_upwind_values_b_to_a(Sigma,ur,dRb,dRa,ii,io,NR,dt)
    uR_star  = get_upwind_values_a_to_b(ur,ur,dRb,dRa,ii,io,NR,dt)
    uphi_star= get_upwind_values_b_to_a(uphi,ur,dRb,dRa,ii,io,NR,dt)
    
    sr = np.zeros(NR+5)
    for i in range(ii,io+2):
        # momentum flux on a-grid, boundary values no required
        sr[i] = 0.5*(Sigma[i]+Sigma[i-1]) * ur[i]

    sphi = np.zeros(NR+4)
    for i in range(ii-1,io+2):
        sphi[i] = Sigma[i] * uphi[i] * g32b[i] # cell centred

    M_r = den_star * ur # momentum flux in radial direction
    
    for i in range(ii,io+1):
        Sigma_new[i] = Sigma[i] -dt * (M_r[i+1]*g32a[i+1]-M_r[i]*g32a[i])/dva[i]
    
    # radial momentum
    for i in range(ii,io+2):        
        sr[i] += -dt * (0.5*(M_r[i]+M_r[i+1])*uR_star[i]*g32b[i] - 0.5*(M_r[i]+M_r[i-1])*uR_star[i-1]*g32b[i-1]) / dvb[i]    

    # angular momentum
    for i in range(ii,io+1):
        sphi[i] += -dt * (M_r[i+1]*uphi_star[i+1]*g32a[i+1]**2.-M_r[i]*uphi_star[i]*g32a[i]**2.)/dva[i]
    
    return Sigma_new, sr, sphi

@jit(nopython=True)
def momentum_to_velocity(Sr,Sphi,Sigma,g32b,ii,io,NR):

    ur = np.zeros(NR+5)
    uphi = np.zeros(NR+4)

    for i in range(ii,io+2):

        ur[i] = Sr[i] / (0.5 * (Sigma[i-1]+Sigma[i]))

    for i in range(ii,io+1):
        uphi[i] = Sphi[i] / (Sigma[i]*g32b[i])

    return ur, uphi

@jit(nopython=True)
def get_upwind_values_b_to_a(quant,ur,dRb,dRa,ii,io,NR,dt):
    
    # calculates the upwinded quantaties in the radial direction upwinded from b
    # to a grid
    
    qstar = np.zeros(NR+5) # van-leer upwinded value
    dq_2 = np.zeros(NR+5) # van-leer derivatives

    for i in range(ii-1,io+2):
        dqm = (quant[i]-quant[i-1])/dRb[i]
        dqp = (quant[i+1]-quant[i])/dRb[i+1]
        dq_2[i]= max(dqm*dqp,0.) * np.sign(dqm+dqp) / max(np.fabs(dqm+dqp),1e-10)

    for i in range(ii,io+2):
        if (ur[i] > 0.):
            qstar[i] = quant[i-1] + (dRa[i-1]-ur[i]*dt)*dq_2[i-1]
        else:
            qstar[i] = quant[i] - (dRa[i]+ur[i]*dt)*dq_2[i]
            
    return qstar

@jit(nopython=True)    
def get_upwind_values_a_to_b(quant,ur,dRb,dRa,ii,io,NR,dt):
    
    # calculates the upwinded quantaties in the radial direction upwinded from a
    # to b grid
    
    qstar = np.zeros(NR+5) # van-leer upwinded value
    dq_2 = np.zeros(NR+5) # van-leer derivatives
    uR_avg = np.zeros(NR+5) # velocity averaged to b grid

    for i in range(ii-1,io+3):
        dqm = (quant[i]-quant[i-1])/dRa[i-1]
        dqp = (quant[i+1]-quant[i])/dRa[i]
        dq_2[i]= max(dqm*dqp,0.) * np.sign(dqm+dqp) / max(np.fabs(dqm+dqp),1e-10)
        uR_avg[i] = 0.5*(ur[i]+ur[i+1])

    for i in range(ii-1,io+2):
        if (ur[i] > 0.):
            qstar[i] = quant[i] + (dRb[i]-uR_avg[i]*dt)*dq_2[i]
        else:
            qstar[i] = quant[i+1] - (dRb[i+1]+uR_avg[i+1]*dt)*dq_2[i+1]
            
    return qstar
    
@jit(nopython=True)
def dust_diffusion(pure_alpha,alpha,cs,Omega_a,pl_index,Sc,Sigma_g,Sigma_d,drb,ra,rb,ii,io,NR,Ndust):

    # performs the diffusive update to the dust velocity
    vr_dust_diff = np.zeros((NR+5,Ndust))
    for j in range(Ndust):
        for i in range(ii,io+2):
            if (pure_alpha):
                nu_agrid = 0.5*(alpha[i]*cs[i]**2.+alpha[i-1]*cs[i-1]**2.)/Omega_a[i] 
            #else:
                nu_agrid = (alpha[ii]*cs[ii]**2./Omega_a[ii]) * (ra[i]/rb[ii])**(pl_index)

            Fdust = - nu_agrid/Sc[j] * 0.5* (Sigma_g[i]+Sigma_g[i-1])* (Sigma_d[i,j]/Sigma_g[i]-Sigma_d[i-1,j]/Sigma_g[i-1]) / drb[i]

            vr_dust_diff[i,j] = Fdust / (0.5*(Sigma_d[i,j]+Sigma_d[i-1,j]))

    return vr_dust_diff


@jit(nopython=True)
def get_timestep(dra,ur,cs):
    
    dt_cs = np.min(dra[:-1]/cs)
    dt_ur = np.min(np.fabs(dra[:-1]/(ur[:-1]+1e-6*cs)))
    dt_art_vis = np.min(dra[:-1]/(12.*np.diff(ur)+1e-6*cs))
    
    
    dt = 0.35 / np.sqrt(1./dt_cs**2.+1./dt_ur**2.+1./dt_art_vis**2.)
    
    return dt