import streamlit as st
import numpy as np
import pandas as pd

# Cosmology calculator (based on Schombert’s Ned Wright Python code:contentReference[oaicite:1]{index=1})
def cosmocalc(z, H0=70.0, WM=0.3, WV=None):
    """Calculate luminosity distance in Mpc for redshift z."""
    c = 299792.458  # km/s
    if z > 100:  # if input is km/s, convert to z
        z = z / c
    if WV is None:
        WV = 1.0 - WM - 0.4165/(H0*H0)
    h = H0/100.0
    WR = 4.165e-5/(h*h)
    WK = 1 - WM - WV - WR
    az = 1.0/(1.0+z)
    n = 1000
    DCMR = 0.0
    DTT = 0.0
    # Integrate for comoving distance (midpoint rule)
    for i in range(n):
        a = az + (1-az)*(i+0.5)/n
        adot = np.sqrt(WK + WM/a + WR/(a*a) + WV*(a*a))
        DCMR += 1.0/(a*adot)
        DTT  += 1.0/adot
    DCMR = (1.0-az)*DCMR/n
    # Convert to Mpc
    DCMR_Mpc = DCMR * c/H0
    DL_Mpc = (1.0+z) * DCMR_Mpc
    return {'DL_Mpc': DL_Mpc}

def compute_lobe_fields(alpha, gamma1, gamma2, v0, Sv0, vol_kpc3, z, H0=70.0, Om=0.3, Ol=0.7):
    """
    Compute B_min, B_eq, energy densities, luminosity and jet power for one lobe.
    alpha: spectral index; gamma1,gamma2: electron Lorentz factor bounds;
    v0: frequency in MHz, Sv0: flux in Jy; vol_kpc3: volume of lobe in kpc^3;
    z: redshift.
    """
    # Convert volume to cm^3
    vol_cm3 = vol_kpc3 * (3.085677581e21)**3  # 1 kpc = 3.0857e21 cm
    # Luminosity distance
    cosmo = cosmocalc(z, H0=H0, WM=Om, WV=Ol)
    D_L_Mpc = cosmo['DL_Mpc']
    # Compute spectral luminosity at v0: L_ν = 4π D_L^2 S_ν (1+z)^(α-1), S in cgs
    L_nu = 4*np.pi*(D_L_Mpc*3.085677581e24)**2 * (Sv0*1e-23) * (1+z)**(alpha-1)
    # Total luminosity at v0 (erg/s)
    L_total = L_nu * (v0*1e6)
    # Electron energy integral factor: ∫γ^(–p) dγ
    p = 2*alpha + 1
    if abs(p-2.0) > 1e-6:
        int_e = (gamma2**(2-p) - gamma1**(2-p))/(2-p)
    else:
        # p -> 2 (alpha->0.5): use log limit
        int_e = np.log(gamma2/gamma1)
    # Approximate electron energy density (erg/cm^3) ~ L_nu * int_e / V
    u_e = L_nu * int_e / vol_cm3
    # Equipartition B: set magnetic energy density = electron energy density
    B_field = np.sqrt(8*np.pi * u_e)
    B_min = B_field
    B_eq = B_field  # here we assume equipartition ~ minimum energy
    # Magnetic energy density
    u_B = B_field**2/(8*np.pi)
    # Estimate jet power: use p = (u_e+u_B)/3, age ~1e7 yr
    pressure = (u_e + u_B)/3.0
    age_sec = 1e7 * 3.154e7  # 1e7 years in seconds
    Q = 4 * pressure * vol_cm3 / age_sec
    return {
        'B_min': B_min,
        'B_eq': B_eq,
        'u_B': u_B,
        'u_e': u_e,
        'L': L_total,
        'Q': Q
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Lobe Magnetic Field Estimator", layout="wide")
st.title("Lobe Magnetic Field Estimator")

# Cosmology inputs
st.sidebar.header("Cosmology Parameters")
H0 = st.sidebar.number_input("Hubble constant H0 (km/s/Mpc)", 50.0, 90.0, 70.0)
Om = st.sidebar.number_input("Omega_matter", 0.0, 1.0, 0.3)
Ol = st.sidebar.number_input("Omega_lambda", 0.0, 1.0, 0.7)

# Input section
st.header("Input Parameters")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Spectral Parameters")
    alpha = st.number_input("Spectral index α", -5.0, 5.0, 0.7, step=0.01)
    gamma1 = st.number_input("γ_min (electron Lorentz factor)", 1.0, 1e7, 10.0, format="%.1f")
    gamma2 = st.number_input("γ_max", 1.0, 1e9, 1e4, format="%.1f")
    v0 = st.number_input("Obs. frequency ν₀ (MHz)", 0.1, 1e6, 1400.0, format="%.1f")
    Sv0 = st.number_input("Flux S_ν₀ (Jy)", 0.0, 1e6, 1.0, format="%.3f")
with col2:
    st.subheader("Lobe Geometry")
    geom = st.selectbox("Geometry", ["Ellipsoid", "Cylinder"])
    if geom == "Ellipsoid":
        l1 = st.number_input("Lobe 1: Length ℓ₁ (kpc)", 0.0, 1e5, 10.0, format="%.3f")
        b1 = st.number_input("Lobe 1: Breadth b₁ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
        h1 = st.number_input("Lobe 1: Height h₁ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
        l2 = st.number_input("Lobe 2: Length ℓ₂ (kpc)", 0.0, 1e5, 10.0, format="%.3f")
        b2 = st.number_input("Lobe 2: Breadth b₂ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
        h2 = st.number_input("Lobe 2: Height h₂ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
    else:
        r1 = st.number_input("Lobe 1: Radius r₁ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
        h1 = st.number_input("Lobe 1: Height h₁ (kpc)", 0.0, 1e5, 10.0, format="%.3f")
        r2 = st.number_input("Lobe 2: Radius r₂ (kpc)", 0.0, 1e5, 5.0, format="%.3f")
        h2 = st.number_input("Lobe 2: Height h₂ (kpc)", 0.0, 1e5, 10.0, format="%.3f")
    z = st.number_input("Redshift z", 0.0, 10.0, 0.1, step=0.001)

# Compute volumes
if geom == "Ellipsoid":
    vol1 = (4/3)*np.pi*(l1/2)*(b1/2)*(h1/2)
    vol2 = (4/3)*np.pi*(l2/2)*(b2/2)*(h2/2)
else:
    vol1 = np.pi*(r1**2)*h1
    vol2 = np.pi*(r2**2)*h2

# Calculate lobe quantities
res1 = compute_lobe_fields(alpha, gamma1, gamma2, v0, Sv0, vol1, z, H0=H0, Om=Om, Ol=Ol)
res2 = compute_lobe_fields(alpha, gamma1, gamma2, v0, Sv0, vol2, z, H0=H0, Om=Om, Ol=Ol)

# Display results side-by-side
st.header("Results")
colL, colR = st.columns(2)
with colL:
    st.subheader("Left Lobe")
    st.write(f"Volume (kpc³): {vol1:.9f}")
    st.write(f"Luminosity (erg/s): {res1['L']:.9f}")
    st.write(f"B_min (G): {res1['B_min']:.9f}")
    st.write(f"B_eq (G): {res1['B_eq']:.9f}")
    st.write(f"Energy dens. (mag) [erg/cm³]: {res1['u_B']:.9e}")
    st.write(f"Energy dens. (elec) [erg/cm³]: {res1['u_e']:.9e}")
    st.write(f"Jet power (erg/s): {res1['Q']:.9f}")
with colR:
    st.subheader("Right Lobe")
    st.write(f"Volume (kpc³): {vol2:.9f}")
    st.write(f"Luminosity (erg/s): {res2['L']:.9f}")
    st.write(f"B_min (G): {res2['B_min']:.9f}")
    st.write(f"B_eq (G): {res2['B_eq']:.9f}")
    st.write(f"Energy dens. (mag) [erg/cm³]: {res2['u_B']:.9e}")
    st.write(f"Energy dens. (elec) [erg/cm³]: {res2['u_e']:.9e}")
    st.write(f"Jet power (erg/s): {res2['Q']:.9f}")

# Batch upload for multiple sources
st.header("Batch CSV Upload")
uploaded_file = st.file_uploader("Upload CSV/TSV (columns for lobe1 and lobe2 parameters)", type=['csv','tsv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, sep='\t')
    out_rows = []
    for idx, row in df.iterrows():
        a = float(row['alpha'])
        g1 = float(row['gamma1'])
        g2 = float(row['gamma2'])
        v0_i = float(row['v0'])
        S0_i = float(row['s_v0'])
        z_i = float(row.get('z', 0.0))
        # Extract geometry per lobe
        if geom == "Ellipsoid":
            L1 = float(row['l1']); B1 = float(row['b1']); H1 = float(row['h1'])
            L2 = float(row['l2']); B2 = float(row['b2']); H2 = float(row['h2'])
            V1 = (4/3)*np.pi*(L1/2)*(B1/2)*(H1/2)
            V2 = (4/3)*np.pi*(L2/2)*(B2/2)*(H2/2)
        else:
            R1 = float(row['r1']); H1 = float(row['h1'])
            R2 = float(row['r2']); H2 = float(row['h2'])
            V1 = np.pi*(R1**2)*H1
            V2 = np.pi*(R2**2)*H2
        res1_i = compute_lobe_fields(a, g1, g2, v0_i, S0_i, V1, z_i, H0=H0, Om=Om, Ol=Ol)
        res2_i = compute_lobe_fields(a, g1, g2, v0_i, S0_i, V2, z_i, H0=H0, Om=Om, Ol=Ol)
        out_rows.append({
            'Source': row.get('Source', f"src_{idx}"),
            'Vol1 [kpc^3]': f"{V1:.9f}", 'Vol2 [kpc^3]': f"{V2:.9f}",
            'Lum1 [erg/s]': f"{res1_i['L']:.9f}", 'Lum2 [erg/s]': f"{res2_i['L']:.9f}",
            'Bmin1 [G]': f"{res1_i['B_min']:.9f}", 'Bmin2 [G]': f"{res2_i['B_min']:.9f}",
            'Beq1 [G]': f"{res1_i['B_eq']:.9f}", 'Beq2 [G]': f"{res2_i['B_eq']:.9f}",
            'uB1 [erg/cm3]': f"{res1_i['u_B']:.9e}", 'uB2 [erg/cm3]': f"{res2_i['u_B']:.9e}",
            'ue1 [erg/cm3]': f"{res1_i['u_e']:.9e}", 'ue2 [erg/cm3]': f"{res2_i['u_e']:.9e}",
            'Q1 [erg/s]': f"{res1_i['Q']:.9f}", 'Q2 [erg/s]': f"{res2_i['Q']:.9f}"
        })
    df_out = pd.DataFrame(out_rows)
    st.dataframe(df_out)
    csv_data = df_out.to_csv(index=False)
    st.download_button("Download Results CSV", data=csv_data, file_name="lobe_results.csv", mime="text/csv")


st.markdown("---")
st.markdown(
    "📌 The cosmology calculator used for this project is based on [James Schombert's python version of the Ned Wright's Cosmology Calculator](https://www.astro.ucla.edu/~wright/CC.python).",
    unsafe_allow_html=True
)
st.markdown(
    "📖 Reference: Wright, E. L. (2006). A Cosmology Calculator for the World Wide Web. *Publications of the Astronomical Society of the Pacific*, 118(850), 1711–1715. [doi:10.1086/510102](https://doi.org/10.1086/510102)",
    unsafe_allow_html=True
)
st.markdown(
    """
    <hr style="margin-top: 3rem; margin-bottom: 1rem;">
    <div style='text-align: center; font-size: 0.9rem; color: gray;'>
        Created by <b>Arnav Sharma</b><br>
        Under the Guidance of <b>Dr. Chiranjib Konar</b>
    </div>
    """,
    unsafe_allow_html=True
)
