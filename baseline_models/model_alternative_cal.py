    # Baseline model with alternative calcification equation

# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon, Point, box
from typing import List, Tuple, Dict
from enum import Enum

    # ----- FUNCTIONS -----
# lesion field initialisation
def create_lesion(region_polygon, x, y, origin, alfa, r0, eps):
    c_l = np.ones((x, y)) 
    i0, j0 = origin
    for i in range(x):
        for j in range(y):
            p = Point(i, j)
            if region_polygon is not None and not region_polygon.contains(p):
                c_l[i, j] = 0 # 0 to the outside of the lobule
            else:
                dist = np.sqrt((i - i0)**2 + (j - j0)**2)
                c_l[i, j] = alfa*0.5 * (1 - np.tanh((dist-r0) / eps)) # hyperbolic tangent as the initialisation function
    c_l[c_l < 1e-3] = 0.0
    return c_l

# fibroblast field initialisation
def create_fibroblast(region_polygon, x, y):
    c_f = np.ones((x, y))  # saturated in the septa
    for i in range(x-1):
        for j in range(y-1):
            p = Point(i, j)
            if region_polygon is not None and region_polygon.contains(p):
                c_f[i, j] = 0 # fibroblasts are 0 inside the secondary lobule
    return c_f


    ## ---- DYNAMIC FUNCTIONS ----
# lesion evolution function
# lesion evolution function
def lesion_evolution(x, y, D_l, a, k, dt, dx, c_l, c_f, cal_rate, g):  
    # reaction term
    reaction =  k * c_l * (c_l - a) * g
    
    # diffusion term
    DIF_l = compute_diffusion_flux(x, y, D_l, c_l, g, dx)
    
    # time integration
    c_l_new = c_l + dt * (reaction + DIF_l - cal_rate)
    c_l_new[c_l_new < 1e-4] = 0.0
    c_l_new = np.maximum(0.0, c_l_new)
    return c_l_new, DIF_l


# fibroblasts evolution function
def fibroblast_evolution(x, y, D, dt, dx, c_l, c_f, c_cal, g, xi, xi_tax, k_f):
    gx, gy = np.gradient(c_l, dx) # gradient computation
    grad_l = np.sqrt(gx**2 + gy**2)
    
    # coeficients 
    D_kin = xi * grad_l* g + 1e-12 # chemokinetic coefficient
    D_tax = xi_tax * g # chemotactic coefficient
    A_tax = D_tax * c_f
    
    
    # effective fluxes computation
    DIF_tax = compute_diffusion_flux(x, y, A_tax, c_l, np.ones_like(c_f), dx)  # chemotaktic flux
    DIF_kin = compute_diffusion_flux(x, y, D_kin, c_f, np.ones_like(c_f), dx)  # chemokinetik flux 

    # fibroblast proliferation 
    proliferation = k_f * c_f * grad_l * g # fibroblasts multiply when they are in contact with the lesion, as long as there is available space
    
    # time integration 
    c_f_new = c_f + dt*(DIF_kin - DIF_tax + proliferation)
    c_f_new[c_f_new < 1e-3] = 0.0
    c_f_new = np.maximum(0.0, c_f_new)
    
    return c_f_new, DIF_kin, DIF_tax, proliferation

    
# calcification evolution function
def calcium_evolution(x, y, dt, dx, c_cal, c_l, k_nuc, k_grow, dcal, theta=0.04):
    # slow nucleation
    nucleation = k_nuc * c_l * (1.0 - c_cal)

    # autocatalytic growth
    above_threshold = np.maximum(c_cal - theta, 0.0)
    growth = k_grow * above_threshold * c_l * (1.0 - c_cal)

    # drainage 
    drain = dcal * c_cal * (c_cal > theta).astype(float)

    net_rate = nucleation + growth - drain
    c_cal_new = np.clip(c_cal + dt * net_rate, 0.0, 1.0)

    delta_cal = np.maximum(c_cal_new - c_cal, 0.0)

    return c_cal_new, net_rate, delta_cal


# diffusion flux discretisation
def compute_diffusion_flux(x, y, D, c, g, dx):
    flux_x = np.zeros_like(c)
    flux_y = np.zeros_like(c)
    ix = np.arange(0, x - 1)
    iy = np.arange(0, y - 1)

    if np.isscalar(D):  # constant diffusion coefficient
        D = D * np.ones_like(c) # we convert the scalar coeff into a constant matrix

    # diffusion coefficient between cells n i n+1
    Dx = 0.5 * (D[ix, :] + D[ix+1,:]) # horizontal coordinate
    Dy = 0.5 * (D[:, iy] + D[:,iy+1]) # vertical coordinate
    
    # flux computation
        # horitzontal
    flux_x[ix, :] += Dx * (g[ix, :] * c[ix + 1, :] - g[ix + 1, :] * c[ix, :])
    flux_x[ix+1, :] -= Dx * (g[ix, :] * c[ix + 1, :] - g[ix + 1, :] * c[ix, :])

        # vertical
    flux_y[:, iy] -= Dy * (g[:, iy + 1] * c[:, iy] - g[:, iy] * c[:, iy + 1])
    flux_y[:, iy + 1] += Dy * (g[:, iy + 1] * c[:, iy]  - g[:, iy] * c[:, iy + 1])

    return (flux_x + flux_y)/dx**2


    ## ---- GEOMETRIC FUNCTIONS ----
# lesion radius computation
def calcul_radius(m, dx):  
    return np.sqrt((dx**2)* np.sum(m[m > 0.1])/ (3.14))


## ---- PHASE CLASSIFICATION ----
class LesionPhase(Enum):
    INIT = 0
    I   = 1   # growing: not encapsulated, age >= 14 days, lesion actively expanding
    II  = 2   # encapsulated: encapsulation not yet confirmed, active process. radius has decreased
    III = 3   # calcifying: encapsulated, calcification seeed planted and growing
    IV  = 4   # resolved: calcification dominates, c_cal >> c_l
    ATB = 5

def get_lesion_phase(age, encapsulated, c_cal, c_l, current_radius, current_phase, r0_mm=0.075,
                     atb=False, recent_speed=None, speed_threshold=2e-3, encapsulating_factor=10, cal_dominance_threshold=2.0):

    # disappeared or active lesion??
    if (current_radius + calcul_radius(c_cal, 0.05)) < 0.05 or (np.sum(c_l) + np.sum(c_cal)) < 1e-3:
        return None  # disappeared
    if atb:
        return LesionPhase.ATB # active
    if current_phase == LesionPhase.ATB:
        return LesionPhase.ATB

    # phase computation
    if age < 14.0:
        candidate = LesionPhase.INIT # initialisation phase

    elif not encapsulated: # either I or II
        candidate = LesionPhase.I
        if (recent_speed is not None
                and current_radius > 2 * r0_mm # if the growth speed has slowed down -> encapsulation in process -> phase II
                and np.abs(recent_speed) < encapsulating_factor * speed_threshold):
            candidate = LesionPhase.II

    else:
        total_cal = np.sum(c_cal)
        total_les = np.sum(c_l) + 1e-12

        if total_cal < 1e-6: # if encapsulated but calcification is small -> phase III
            candidate = LesionPhase.III
        else: # if c_cal >> c_l -> phase IV
            r_cal = np.sqrt(total_cal / np.pi)
            r_les = np.sqrt(total_les / np.pi) + 1e-12
            if r_cal >= cal_dominance_threshold * r_les:
                candidate = LesionPhase.IV
            else:
                candidate = LesionPhase.III

    if current_phase is None or current_phase == LesionPhase.INIT:
        return candidate
    return candidate if candidate.value > current_phase.value else current_phase
       
 

     ## ---- MODEL SET UP ----
def model_set_up(central_region, x, y, origin1, alfa, r0, eps):
    # fibroblasts initialisation and boundary conditions
    c_f1 = create_fibroblast(central_region, x, y)
    c_f1[0, :] = c_f1[-1, :] = c_f1[:, 0] = c_f1[:, -1] = 1.0
    c_f10 = c_f1.copy()

    # lesion initialisation and boundary conditions
    c_l1 = create_lesion(central_region, x, y, origin1, alfa, r0, eps)
    c_l1[0, :] = c_l1[-1, :] = c_l1[:, 0] = c_l1[:, -1] = 0.0
    c_l10 = c_l1.copy()

    g10 = 1.0 - c_l10 - c_f10

    return c_f1, c_f10, c_l1, c_l10, g10
    

    ##  ---- MAIN LOOP ----
def model_evolution_loop(iter, params):
       #  parameter definition 
    dt = params["dt"]; dx = params["dx"]; lx = params["lx"]; ly = params["ly"]
    a = params["a"]; age_1 = params["age1"]
    eps = params["eps"]; r0 = params["r0"]; alfa = params["alfa"]; count = params["count"]
    origin1 = params["origin1"]; central_region = params["central_region"]; 
    D = params["D"]; k = params["k"];
    xi = params["xi"]; xi_tax = params["xi_tax"]; k_f = params["k_f"];
    k_nuc = params["k_nuc"]; k_grow = params["k_grow"]; dcal = params["dcal"]

        # variable initialisation
    c_f1, c_f10, c_l_1, c_l10, g10 = model_set_up(central_region, lx, ly, origin1, alfa, r0, eps) 
    

        #  initialization of variables
   # -------------------- General --------------------
    t = np.zeros(iter) # time
    r_1 = np.zeros(iter) # radius of lesion
    r_cell_1 = np.zeros(iter)  # radius of lesion + calcification
    full_r_1 = np.zeros(iter) # radius of lesion + calcification + fibroblast crown
    g1 = np.zeros((lx, ly)) # availability of space, volume conservation
    
    # -------------------- Lesion --------------------
    D_l = D    # lesion diffusion coefficient
    DIF_l_1_vector = np.zeros(iter) # max{∇⋅(g∇c_l)} across the grid in each iter
    max_grad = np.zeros(iter)  # maximum lesion gradient across the grid in each iter
    c_l_lag = np.zeros((lx, ly)) # delayed lesion field
    
        # -------------------- Fibroblast --------------------
    D_kin = np.zeros((lx, ly))    # chemokinetic diffusion coefficient
    D_tax = np.zeros((lx, ly))    # chemotactic diffusion coefficient
    DIF_kin_vector = np.zeros(iter) # max coeff across the grid in each iter
    DIF_tax_vector = np.zeros(iter) # max coeff across the grid in each iter
    prol_vector = np.zeros(iter)    # proliferation coefficient   
    enc_count = 0 
    enc_timer = 0
    
    # -------------------- Calcification --------------------
    c_cal1 = np.zeros((lx, ly))   # calcification initialisation (empty field)
    cal_dif_vec = np.zeros(iter)  # calcification diffusion coefficient
    cal_rate_vec = np.zeros(iter)
    
    # -------------------- Ocupation function --------------------
    g_global = 1.0 - c_l_1 - c_f1  # occupation function
    cfl_vec = np.zeros(iter)
    overshoot_vec = np.zeros(iter)
    n_cells_corrected_vec = np.zeros(iter)


    # we initialise dictionaries
        # state of each lesion and diffusion vector/matrices
    state = {'c_l_1' : c_l_1}   
    diffusion = {'DIF_l_1': np.zeros((lx,ly)), 'DIF_l_1_vector': DIF_l_1_vector}
    min_state = {'c_l_1_min':c_l_1} 
        
        # ages of each lesion
    ages = {'age_1': age_1, }
    
        # origin of each lesion
    origins = {'origin_1' : origin1}
    
        # radius of each lesion
    radius = {'r_1' : r_1}
    
        # radius lesion + calcification
    radius_lc = {'r_lc_1' : r_cell_1}
       
        # radius of lesion + calcification + fibroblast crown 
    full_radius = {'full_r_1':full_r_1}

        # calcification state of each lesion
    calcification = {'c_cal_1' : c_cal1}
    
        # fibroblast crown
    fibros = {'c_f1' : c_f1}

        # encapsulation variable
    encapsulation = {'encap_1':False}
    encap_counter = {'encap_counter_1': enc_count}

        # phase classification 
    phases = {'phase_1': np.zeros(iter, dtype=int)}

        # dictionaries to save our variables at selected timepoints
    c_l_time = {}
    c_cal_time = {} 
    c_f_time = {}

    
    # fibroblast mask for computing the crown (we eliminate fibroblasts at the septa)
    region_mask = np.zeros((lx, ly)) 
    if central_region is not None:
        for ix in range(lx-1):
            for jy in range(ly-1):
                if central_region.contains(Point(ix, jy)):
                    region_mask[ix, jy] = 1.0

# -------------- evolution loop --------------------
    for i in range(iter):
        t[i] = dt * i

        # 1) We compute the available space
        total_c_l = sum(state.values()) # one variable for all LESION FRACTION
        total_c_cal = sum(calcification.values()) # one variable for all CALCIFICATION FRACTION
        g_global = 1.0 - c_f1 - total_c_l - total_c_cal
        g_global = np.clip(g_global, 0.0, 1.0)


        # 2) Let's check for a change in "state" -> if encapsulation is completed / new lesion is seeded
        window_size = 20000
        for e in range(1, count + 1):
            if encapsulation[f'encap_{e}']:
                continue
            if ages[f'age_{e}'] < 15.0:              
                continue
            radius_e = radius[f'r_{e}']
            current_radius = radius_e[i-1]
        
            if current_radius < 0.05 or current_radius > 10.0:
                continue
        
            fin_idx = min(i, window_size)
        
            if fin_idx < 1000:
                continue
        
            recent_radius = radius_e[i - fin_idx : i]
            recent_time = dt * np.arange(len(recent_radius))
        
            mean_t = np.mean(recent_time)
            mean_r = np.mean(recent_radius)
            covariance = np.mean(recent_time * recent_radius) - (mean_t * mean_r)
            variance_t = (np.mean(recent_time**2) - mean_t**2) + 1e-12
            recent_speed = covariance / variance_t
        
            if np.abs(recent_speed) < 2e-4:# if radius growth has slowed down, we activate encapsulation counter
                encap_counter[f'encap_counter_{e}'] += 1
            else:
                encap_counter[f'encap_counter_{e}'] = 0
        
            if encap_counter[f'encap_counter_{e}'] >= 10:  # encapsulation confirmed
                print(f'Encapsulation confirmed for lesion {e} at step {i}, '
                      f'age {ages[f"age_{e}"]:.1f} days')
                encapsulation[f'encap_{e}'] = True


        # 3) Let's integrate aross all variables
            # we'll use temporary dictionaries to store our updated without deleting the ones from the previous step  
        new_state = {}
        new_calcification = {}

        # 3A) let's update the local variables (lesion and calcification)
        current_count = count
        for e in range (1, current_count+1):
            key = f"c_l_{e}"
            tau_lag = 2.0 
            c_l_lag = c_l_lag + dt * (total_c_l - c_l_lag) / tau_lag
            if key not in state:
                continue 
            if state[key].any():                     
                age_e_curr = ages[f'age_{e}']
                # ens guardem els valors actuals de lesió i calcificació
                c_l_e_curr = state[f'c_l_{e}'] 
                c_cal_curr = calcification[f'c_cal_{e}'] 
                
                if (age_e_curr >= 14.0): # we only integrate if the lesion is over 14 days of age                            
                    # let's update calcification (if encapsulation has been completed)
                    c_cal_new, net_rate, delta_cal = calcium_evolution(lx, ly, dt, dx, c_cal_curr, c_l_lag, k_nuc, k_grow, dcal)
                    cal_rate = delta_cal  # actual mass transferred from c_l → c_cal
                    cal_rate_vec[i] = np.max(cal_rate)


                     # let's update lesion and the diffusion vectors
                    c_l_e_new, DIF_l_e = lesion_evolution(lx, ly, D_l, a, k, dt, dx, c_l_e_curr, c_f1, cal_rate, g_global)
                    
                    DIF_l_e_vector = diffusion[f'DIF_l_{e}_vector']; DIF_l_e_vector[i] = np.max(np.abs(DIF_l_e))
                    diffusion[f'DIF_l_{e}_vector'] = DIF_l_e_vector; diffusion[f'DIF_l_{e}'] = DIF_l_e

                    # let's store our updates
                    new_state[f'c_l_{e}'] = c_l_e_new
                    new_calcification[f'c_cal_{e}'] = c_cal_new
                else: 
                    new_state[f'c_l_{e}'] = c_l_e_curr
                    new_calcification[f'c_cal_{e}'] = c_cal_curr
                    
                ages[f'age_{e}'] += dt # age of lesion measured in DAYS
                
        # 3B) let's update fibroblast concentration
        c_f_new, DIF_kin, DIF_tax, proliferation = fibroblast_evolution(lx, ly, D, dt, dx, total_c_l, c_f1, total_c_cal, g_global, xi, xi_tax, k_f)
        
        # we store the distributions from each term
        DIF_kin_vector[i] = np.sum(np.abs(DIF_kin))
        DIF_tax_vector[i] = np.sum(np.abs(DIF_tax))
        prol_vector[i] = np.sum(np.abs(proliferation))


        # 4) Let's update all variables at once
        for e in range(1, count + 1):
            if state[key].any():
                state[f'c_l_{e}'] = new_state[f'c_l_{e}']
                calcification[f'c_cal_{e}'] = new_calcification[f'c_cal_{e}']
        c_f1 = c_f_new

        # 5) Let's revise volume constraints and boundary conditions
        for e in range(1, count + 1):
            l_key = f'c_l_{e}'
            cal_key = f'c_cal_{e}'
            
            state[l_key] = np.maximum(state[l_key], 0.0)
            calcification[cal_key] = np.maximum(calcification[cal_key], 0.0)
        
        c_f1 = np.maximum(c_f1, 0.0)
        
        # Global Proportional Scaling
        new_total_c_l = sum(state.values())
        new_total_c_cal = sum(calcification.values())
        total_mass = new_total_c_l + new_total_c_cal + c_f1       
        overshoot_mask = total_mass > 1.0
        
        if np.any(overshoot_mask):
            scale_factor = np.where(overshoot_mask, 1.0 / total_mass, 1.0)
            for e in range(1, count + 1):
                state[f'c_l_{e}'] *= scale_factor
                calcification[f'c_cal_{e}'] *= scale_factor
            c_f1 *= scale_factor

        # Boundary conditions
        c_f1[0, :] = c_f1[-1, :] = c_f1[:, 0] = c_f1[:, -1] = 1.0
        for e in range(1, count + 1):
            state[f'c_l_{e}'][0, :] = state[f'c_l_{e}'][-1, :] = 0.0
            state[f'c_l_{e}'][:, 0] = state[f'c_l_{e}'][:, -1] = 0.0

        
        # 6) Let's compute the data
            # lesion gradient
        gx, gy = np.gradient(total_c_l, dx)
        gradient_c = np.sqrt(gx**2 + gy**2)
        max_grad[i] = np.max(gradient_c)

            # radii computation
        for e in range(1, count + 1):
            # lesion radius
            radius_e = radius[f'r_{e}']
            radius_e[i] = calcul_radius(state[f'c_l_{e}'], dx)
            radius[f'r_{e}'] = radius_e

            # lesion + calcification radius
            radius_lc_e = radius_lc[f'r_lc_{e}']
            radius_lc_e[i] = calcul_radius(calcification[f'c_cal_{e}'], dx)
            radius_lc[f'r_lc_{e}'] = radius_lc_e

            # lesion + calcification + fibroblast crown radius 
            fibro_mask = region_mask * c_f1
            full_radius_e = full_radius[f'full_r_{e}']
            full_radius_e[i] = calcul_radius(state[f'c_l_{e}'] + fibro_mask + calcification[f'c_cal_{e}'], dx)
            full_radius[f'full_r_{e}'] = full_radius_e

            
         # 7) Let's compute lesion phase
        for e in range(1, count + 1):
            if f'c_l_{e}' not in state:
                continue
        
            current_radius = radius[f'r_{e}'][i]
        
            fin_idx = min(i, 2000)
            if fin_idx >= 1000:
                recent_radius = radius[f'r_{e}'][i - fin_idx : i]
                recent_time = dt * np.arange(len(recent_radius))
                mean_t = np.mean(recent_time)
                mean_r = np.mean(recent_radius)
                covariance = np.mean(recent_time * recent_radius) - (mean_t * mean_r)
                variance_t = (np.mean(recent_time**2) - mean_t**2) + 1e-12
                recent_speed = covariance / variance_t
            else:
                recent_speed = None
            current_phase_val = phases[f'phase_{e}'][i-1] if i > 0 else 0
            current_phase_enum = LesionPhase(current_phase_val)
        
            phase = get_lesion_phase(
                age=ages[f'age_{e}'],
                encapsulated=encapsulation[f'encap_{e}'],
                c_cal=calcification[f'c_cal_{e}'],
                c_l=state[f'c_l_{e}'],
                current_radius=current_radius,
                current_phase = current_phase_enum,
                recent_speed=recent_speed)
        
            if f'phase_{e}' not in phases:
                phases[f'phase_{e}'] = np.zeros(iter, dtype=int)
            phases[f'phase_{e}'][i] = phase.value if phase is not None else 0
        

        # let's save our variables at timepoints of interest
        timesteps = [0, 80000-1, 104000-1, 108000-1, 128000-1, 150000-1, 200000-1] 
        if i in timesteps:
            c_l_time[f'c_l_{i}'] = state['c_l_1'].copy()
            c_cal_time[f'c_cal_{i}'] = calcification['c_cal_1'].copy()
            c_f_time[f'c_f_{i}'] = c_f1.copy()
            
        percent = (i + 1) / iter * 100
        bar = '=' * int(percent // 2)  # Adjust length as needed
        sys.stdout.write(f'\rProgress: [{bar:<50}] {percent:.1f}%')
        sys.stdout.flush()
    
    return state, ages, origins, radius, radius_lc, full_radius, calcification, c_f1, max_grad, DIF_kin_vector, DIF_tax_vector, prol_vector, cal_rate_vec,phases, cfl_vec, overshoot_vec, n_cells_corrected_vec, c_l_time, c_cal_time, c_f_time










