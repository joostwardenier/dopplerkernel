import numpy as np
from scipy import signal, stats
from numba import njit, prange
from matplotlib import pylab as plt

#############################################################################
#############################################################################
@njit(parallel=True, fastmath=True)
def calculate_velocity_field(orbital_phase, v_rot, v_wind, sink_longitude, v_jet, sigma_jet, XX, YY, ZZ, inside):
    """
    The equations in this function are described in Appendix A of Wardenier+ (2025)
    """

    # >>> Planet rotation <<<
    if v_rot > 0:
        vel_rot = v_rot * XX
    else:
        vel_rot = np.zeros_like(XX)

    # >>> Equatorial jet <<<
    if (v_jet != 0) and (sigma_jet > 0):
        exponent = -0.5 * (YY / sigma_jet) ** 2
        vel_jet = v_jet * XX * np.exp(exponent)
    else:
        vel_jet = np.zeros_like(XX)

    # >>> Source-to-sink flow <<<
    if v_wind != 0:

        n_points = XX.size
        vel_wind = np.zeros_like(XX.ravel())

        # Convert degrees to radians (eastward positive)
        lon_source = 0.0
        lon_sink = -np.deg2rad(sink_longitude)

        # Orbital rotation
        theta = 2.0 * np.pi * orbital_phase
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Source and sink on equator (planet frame)
        src_x = np.sin(lon_source)
        src_z = -np.cos(lon_source)

        sink_x = np.sin(lon_sink)
        sink_z = -np.cos(lon_sink)

        # Rotate into observer frame
        src_rx = src_x * cos_t + src_z * sin_t
        src_ry = 0.0
        src_rz = -src_x * sin_t + src_z * cos_t

        sink_rx = sink_x * cos_t + sink_z * sin_t
        sink_ry = 0.0
        sink_rz = -sink_x * sin_t + sink_z * cos_t

        XXf = XX.ravel()
        YYf = YY.ravel()
        ZZf = ZZ.ravel()
        insidef = inside.ravel()

        for idx in prange(n_points):
            if not insidef[idx]:
                continue

            px = XXf[idx]
            py = YYf[idx]
            pz = ZZf[idx]

            # Vector from point to source and sink
            vSS_x = px - src_rx
            vSS_y = py - src_ry
            vSS_z = pz - src_rz

            vAS_x = sink_rx - px
            vAS_y = sink_ry - py
            vAS_z = sink_rz - pz

            # Angular distances
            cos_alpha_SS = px*src_rx + py*src_ry + pz*src_rz
            cos_alpha_AS = px*sink_rx + py*sink_ry + pz*sink_rz

            cos_alpha_SS = min(1.0, max(-1.0, cos_alpha_SS))
            cos_alpha_AS = min(1.0, max(-1.0, cos_alpha_AS))

            alpha_SS = np.arccos(cos_alpha_SS)
            alpha_AS = np.arccos(cos_alpha_AS)

            # Weights
            denom = alpha_SS + alpha_AS + 1e-12
            w_SS = alpha_AS / denom
            w_AS = alpha_SS / denom

            # Weighted flow direction
            vx = w_SS * vSS_x + w_AS * vAS_x
            vy = w_SS * vSS_y + w_AS * vAS_y
            vz = w_SS * vSS_z + w_AS * vAS_z

            norm_v = np.sqrt(vx*vx + vy*vy + vz*vz)
            if norm_v < 1e-12:
                continue
            vx /= norm_v
            vy /= norm_v
            vz /= norm_v

            # Project onto tangent plane
            dot = vx * px + vy * py + vz * pz
            tx = vx - dot*px
            ty = vy - dot*py
            tz = vz - dot*pz

            norm_t = np.sqrt(tx*tx + ty*ty + tz*tz)
            if norm_t < 1e-12:
                continue
            tz /= norm_t

            # LOS velocity (+z observer)
            vel_wind[idx] = -v_wind * tz

        vel_wind = vel_wind.reshape(XX.shape)
    
    else:
        vel_wind = np.zeros_like(XX)

    # Total field
    return vel_rot + vel_jet + vel_wind

#############################################################################
#############################################################################
@njit(parallel=True, fastmath=True)
def calculate_weight_mask(orbital_phase, w_0, peak_shift_longitude, peak_shift_latitude, peak_dropoff, nightside_zero_BOOL, advanced_mask_BOOL, XX, YY, ZZ, inside): 
    """
    The equations in this function are described in Appendix B of Wardenier+ (2025)
    """
    
    # Reset mask whenever function is called
    weight_mask = inside.astype(np.float64)

    theta = 2. * np.pi * orbital_phase

    # Set the nightside weights to zero
    if (nightside_zero_BOOL & (orbital_phase != 0.5)): 

        # surface normal
        #normal_vector = np.stack([XX, YY, ZZ], axis=-1)
        normal_vector = np.empty(XX.shape + (3,), dtype=np.float64)
        normal_vector[:, :, 0] = XX
        normal_vector[:, :, 1] = YY
        normal_vector[:, :, 2] = ZZ

        # surface normal at substellar point
        substellar_vector = np.array([-np.sin(theta), 0.0, -np.cos(theta)])

        #dot = normal_vector @ substellar_vector
        dot = (normal_vector[:, :, 0] * substellar_vector[0] + \
               normal_vector[:, :, 1] * substellar_vector[1] + \
               normal_vector[:, :, 2] * substellar_vector[2])

        #weight_mask[inside] = (dot[inside] > 0).astype(np.float64)
        weight_mask = np.where(inside & (dot > 0), 1.0, 0.0)

    # Apply linear center-to-limb weight function 
    if w_0 != 1.0:
        
        #weight_mask[inside] *= (1.0 + (w_0 - 1.0) * ZZ[inside])
        weight_mask = np.where(inside, weight_mask * (1.0 + (w_0 - 1.0) * ZZ), weight_mask)

    if advanced_mask_BOOL:

        # --- Parameters ---
        peak_shift_rad = np.radians(peak_shift_longitude)  # eastward shift of peak from substellar
        lat_peak_rad   = np.radians(peak_shift_latitude)   # single peak latitude

        # Observer-frame longitude
        lon_obs = np.arctan2(XX, ZZ)                      
        lat     = np.arcsin(np.clip(YY, -1.0, 1.0)) 

        # Substellar longitude (rotates with phase)
        star_lon = (theta + 2*np.pi) % (2*np.pi) - np.pi

        # Longitude in corotating stellar frame
        lon = (lon_obs - star_lon + np.pi) % (2*np.pi) - np.pi

        # Longitude relative to hotspot peak
        lon_rel = lon - peak_shift_rad

        # ===============================
        # Dayside mask
        # ===============================
        dayside = np.abs(lon) <= np.pi/2

        # ===============================
        # Longitude weighting (piecewise cosine)
        # ===============================
        lon_weight = np.zeros_like(lon)

        # WEST of hotspot
        mask_west = dayside & (lon_rel <= 0)
        L_w = peak_shift_rad + np.pi/2
        phase_w = (lon_rel + L_w) / L_w
        lon_weight_west = peak_dropoff * np.cos(0.5*np.pi * (1 - phase_w)) + (1. - peak_dropoff)
        lon_weight = np.where(mask_west, lon_weight_west, lon_weight)

        # EAST of hotspot
        mask_east = dayside & (lon_rel > 0)
        L_e = np.pi/2 - peak_shift_rad
        phase_e = lon_rel / L_e
        lon_weight_east = peak_dropoff * np.cos(0.5*np.pi * phase_e) + (1. - peak_dropoff)
        lon_weight = np.where(mask_east, lon_weight_east, lon_weight)

        # --- Latitude weighting: single peak ---
        lat_weight = np.zeros_like(lat)

        # Northern hemisphere: lat >= lat_peak
        mask_north = lat >= lat_peak_rad
        L_n = np.pi/2 - lat_peak_rad
        frac_n = np.clip((lat - lat_peak_rad) / L_n, 0, 1)
        lat_weight_north = peak_dropoff * np.cos(0.5 * np.pi * frac_n) + (1. - peak_dropoff)
        lat_weight = np.where(mask_north, lat_weight_north, lat_weight)

        # Southern hemisphere: lat < lat_peak
        mask_south = lat < lat_peak_rad
        L_s = np.pi/2 + lat_peak_rad
        frac_s = np.clip((lat_peak_rad - lat) / L_s, 0, 1)
        lat_weight_south = np.cos(0.5 * np.pi * frac_s)
        lat_weight = np.where(mask_south, lat_weight_south, lat_weight)
        
        lat_weight_flipped = lat_weight[::-1,:]
        lat_weight = np.maximum(lat_weight, lat_weight_flipped)

        # --- Combine longitude and latitude ---
        dayside_weight = lon_weight * lat_weight

        # Apply to intensity mask
        #weight_mask[inside] *= dayside_weight[inside]
        weight_mask = np.where(inside, weight_mask * dayside_weight, weight_mask)
    
    return weight_mask
    
#############################################################################
#############################################################################
class DopplerKernel():

    def __init__(self, grid_size=100):

        self.grid_size = grid_size

        coords = np.linspace(-1,1,self.grid_size)
        self.XX, self.YY = np.meshgrid(coords, coords)
        self.R2 = self.XX**2 + self.YY**2

        self.MU = np.zeros_like(self.XX)
        self.inside = self.R2 <=1

        # Cosine angle (1 at center, 0 at limb)
        self.MU[self.inside] = np.sqrt(1.0 - self.R2[self.inside])

        # Default intensity mask (1 inside disk, 0 outside disk)
        self.weight_mask = self.inside.astype(float)

        # Default velocity field (0 everywhere)
        self.velocity_field = np.zeros_like(self.XX)

        # Default center-to-limb weight
        self.w_0 = 0.

        self.c = 299792.458  # km/s
        
    ################################################################
    ################################################################
    def make_velocity_field(self, orbital_phase=0.5, v_rot=5., v_wind=3., sink_longitude=180., v_jet=8., sigma_jet=0.2):

        self.orbital_phase = orbital_phase
        self.v_rot = v_rot
        self.v_wind = v_wind
        self.sink_longitude = sink_longitude
        self.v_jet = v_jet
        self.sigma_jet = sigma_jet

        self.velocity_field = calculate_velocity_field(self.orbital_phase, self.v_rot, self.v_wind, self.sink_longitude, \
                                    self.v_jet, self.sigma_jet, self.XX, self.YY, self.MU, self.inside)

    ################################################################
    ################################################################
    def make_uniform_weight_mask(self):

        self.w_0 = 1.
        self.weight_mask = self.inside.astype(float)

    ################################################################
    ################################################################
    def make_weight_mask(self, orbital_phase=0.5, w_0=1., peak_shift_longitude=15., peak_shift_latitude=30., \
                            peak_dropoff=1., nightside_zero=True, advanced_mask=False):
        
        self.orbital_phase = orbital_phase
        self.w_0 = np.max([0.,w_0])
        self.nightside_zero = nightside_zero
        self.advanced_mask = advanced_mask
        self.peak_shift_longitude = peak_shift_longitude
        self.peak_shift_latitude = peak_shift_latitude
        self.peak_dropoff = peak_dropoff 

        self.weight_mask = calculate_weight_mask(self.orbital_phase, self.w_0, self.peak_shift_longitude, \
                                self.peak_shift_latitude, self.peak_dropoff, self.nightside_zero, self.advanced_mask,
                                self.XX, self.YY, self.MU, self.inside)
    
    ################################################################
    ################################################################
    def plot_velocity_map(self):

        vmax = np.max([abs(self.velocity_field[self.inside].min()), abs(self.velocity_field[self.inside].max())])
        vmin = -vmax
        
        vlos_map = np.ma.masked_where(~self.inside, self.velocity_field) 
        
        plt.figure()
        plt.pcolor(self.XX, self.YY, vlos_map, cmap='RdBu_r', vmin=vmin,vmax=vmax)
        plt.xlabel('x [$R_p$]', fontsize=12)
        plt.ylabel('y [$R_p$]', fontsize=12)
        cbar = plt.colorbar()
        cbar.set_label("Line-of-sight velocity [km/s]", fontsize=12)
        plt.gca().set_aspect("equal")
        plt.show()

    ################################################################
    ################################################################
    def plot_weight_mask(self):
        
        plt.figure()
        plt.pcolor(self.XX, self.YY, self.weight_mask / (self.weight_mask.max() + 1e-12), cmap='binary_r',vmin=0,vmax=1)
        plt.xlabel('x [$R_p$]', fontsize=12)
        plt.ylabel('y [$R_p$]', fontsize=12)
        cbar = plt.colorbar()
        cbar.set_label("Weight value", fontsize=12)
        plt.gca().set_aspect("equal")
        plt.show()

    ################################################################
    ################################################################
    def calculate_doppler_kernel(self, calculate_x_y_arrays=False):

        values = self.velocity_field[self.inside]
        weights = self.weight_mask[self.inside]
        
        self.kernel = stats.gaussian_kde(values, weights=weights)

        if calculate_x_y_arrays:
            
            # Evaluate kernel between -1.5 and +1.5 times the maximum absolute line-of-sight velocity
            vmax = 1.5*np.max([abs(self.velocity_field[self.inside].min()), abs(self.velocity_field[self.inside].max())])
            vmin = -vmax
            
            v_centers = np.linspace(vmin, vmax, 500) 

            K = self.kernel.evaluate(v_centers)
            numerical_K = K / np.trapz(K, v_centers)

            self.kernel_x_values = v_centers
            self.kernel_y_values = numerical_K

    ################################################################
    ################################################################
    def plot_doppler_kernel(self, plot_analytical_rotation_kernel=False):
        
        vmax = 1.5*np.max([abs(self.velocity_field[self.inside].min()), abs(self.velocity_field[self.inside].max())])
        vmin = -vmax

        v_centers = np.linspace(vmin, vmax, 500) 

        # Fetch the kernel values in case it was already calculated before...
        try:
            K = self.kernel.evaluate(v_centers)
        
        # ... if not, calculate the kernel based on the most recently computed velocity field and weight mask
        except AttributeError:
            self.calculate_doppler_kernel()
            K = self.kernel.evaluate(v_centers)
        
        # Normalize the kernel
        numerical_K = K / np.trapz(K, v_centers)
        
        plt.figure()
        plt.plot(v_centers, numerical_K, linewidth=3, color='k', label='numerical kernel')

        # Compare to analytical equation from Gray+ (2005) for pure rotation and a given limb-darkening coefficient 
        if plot_analytical_rotation_kernel:

            velo_mask = abs(v_centers) <= self.v_rot

            if (self.w_0 < 1):
                print('Can only plot analytical rotation kernel for w_0 >= 1 (such that 0 < u_1 <= 1)')

            # Solution without center-to-limb weight variation
            if (self.w_0 == 1):

                 analytical_K = (2 / (np.pi * self.v_rot))*np.sqrt(1 - (v_centers[velo_mask]/self.v_rot)**2)

            # Center-to-limb weight variation
            else:

                mu = v_centers[velo_mask]/self.v_rot

                # Convert w_0 parameter to u_1 limb-darkening coefficient
                u_1 = 1. - 1./self.w_0

                analytical_K =  ( (2 * (1 - u_1) * np.sqrt(1 - mu**2) + 0.5 * np.pi * u_1 * (1 - mu**2)) / 
                                (np.pi * self.v_rot * (1 - u_1/3)) )
            
            plt.plot(v_centers[velo_mask], analytical_K, linewidth=2, color='r', label='analytical kernel')

        plt.xlabel('Radial velocity [km/s]', fontsize=12)
        plt.ylabel('Kernel value', fontsize=12)
        plt.legend(fontsize=12)
        plt.show()

    ################################################################
    ################################################################
    def convolve_with_spectrum(self, wavelength, flux, oversample=4, test_flux_conservation=False):

         # --- 1. log-lambda grid (uniform velocity spacing)
        loglam = np.log(wavelength)
        dloglam = np.min(np.diff(loglam)) / oversample
        loglam_uniform = np.arange(loglam.min(), loglam.max(), dloglam)
        lam_uniform = np.exp(loglam_uniform)

        flux_uniform = np.interp(lam_uniform, wavelength, flux)

        # --- 2. Velocity grid corresponding to pixel spacing
        dv = dloglam * self.c                     # km/s per pixel
        n_points = flux_uniform.size
        v_grid = (np.arange(n_points) - n_points//2) * dv

        vmin = self.velocity_field[self.inside].min()
        vmax = self.velocity_field[self.inside].max()
        diff = vmax - vmin
        vmax += 0.5*diff
        vmin -= 0.5*diff

        # --- 3. Kernel evaluation
        #v_mask = abs(v_grid) < 30 # Only evaluate the kernel at velocities lower than +/- 30 km/s
        v_mask = ((v_grid > vmin) & (v_grid < vmax)) # Evaluate the kernel only between vmin and vmax
        k_values = np.zeros_like(v_grid)

        try:
            k_values[v_mask] = self.kernel.evaluate(v_grid[v_mask])
        except AttributeError:
            self.calculate_doppler_kernel()
            k_values[v_mask] = self.kernel.evaluate(v_grid[v_mask])

        # --- 4. Normalize kernel (important for flux conservation)
        k_values /= np.sum(k_values)

        # --- 5. Convolve in velocity space
        flux_broadened = signal.fftconvolve(flux_uniform, k_values, mode="same")

        # --- 6. Interpolate back to original wavelength grid
        flux_conv = np.interp(wavelength, lam_uniform, flux_broadened)

        if test_flux_conservation:
            
            F_before = np.trapz(flux, wavelength)
            F_after  = np.trapz(flux_conv, wavelength)
            
            print("Relative flux difference convolution:", (F_after - F_before) / F_before)

        return flux_conv
