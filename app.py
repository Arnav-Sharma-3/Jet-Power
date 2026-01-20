# ============================================================
# Lobe Magnetic Field Estimator v3.0
# Dual-lobe geometry + jet power
# ============================================================

import streamlit as st
import pandas as pd
import math
from math import sqrt, exp, sin

# ============================================================
# Constants (CGS)
# ============================================================
CGS_KPC = 3.08567758128e21      # cm
CGS_MPC = 3.08567758128e24      # cm
C1 = 6.266e18
C3 = 2.368e-3
M_E = 9.1093837e-28             # g
C_LIGHT = 2.99792458e10         # cm/s
X_FACTOR = 0.0
SEC_IN_MYR = 1e6 * 365.25 * 24 * 3600

# ============================================================
# Cosmology calculator (unchanged)
# ============================================================
def run_cosmology_calculator(z, H0, WM, WV):
    h = H0 / 100
    WR = 4.165E-5 / (h * h)
    WK = 1 - WM - WR - WV
    az = 1.0 / (1.0 + z)
    c = 299792.458

    n = 1000
    age = 0.0
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age += 1.0 / adot
    zage = az * age / n

    DTT = 0.0
    DCMR = 0.0
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT += 1.0 / adot
        DCMR += 1.0 / (a * adot)

    DCMR = (1 - az) * DCMR / n

    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        ratio = (0.5 * (exp(x) - exp(-x)) / x) if WK > 0 else (sin(x) / x)
    else:
        y = x * x
        ratio = 1. + y / 6. + y * y / 120.

    DA = az * ratio * DCMR
    DA_Mpc = (c / H0) * DA
    DL_Mpc = DA_Mpc / (az * az)
    kpc_DA = DA_Mpc / 206.264806

    return DL_Mpc, DA_Mpc, kpc_DA

# ============================================================
# Geometry volumes
# ============================================================
def volume_ellipsoid(l, b, h, Sf):
    return (4/3) * math.pi * (l*Sf) * (b*Sf) * (h*Sf)

def volume_cylinder(r, h, Sf):
    return math.pi * (r*Sf)**2 * (h*Sf)

# ============================================================
# Core physics
# ============================================================
def compute_source(
    alpha, g1, g2, v0, s_v0, z,
    geometry, dims,
    t_age_myr,
    H0, WM, WV
):
    if abs(alpha - 1.0) < 1e-3:
        raise ValueError("α too close to 1 (divergent integrals).")
    if not (g2 > g1 > 1):
        raise ValueError("Require γ₂ > γ₁ > 1.")

    DL, DA, Sf = run_cosmology_calculator(z, H0, WM, WV)
    DL_cm = DL * CGS_MPC

    # ---- Volume ----
    if geometry == "Ellipsoid":
        V1 = volume_ellipsoid(*dims[0], Sf)
        V2 = volume_ellipsoid(*dims[1], Sf)
    else:
        V1 = volume_cylinder(*dims[0], Sf)
        V2 = volume_cylinder(*dims[1], Sf)

    V_kpc3 = V1 + V2
    V_cm3 = V_kpc3 * CGS_KPC**3

    # ---- Synchrotron ----
    v0_hz = v0 * 1e6
    s_v0_cgs = s_v0 * 1e-23
    p = 2 * alpha + 1

    L1 = 4 * math.pi * DL_cm**2 * s_v0_cgs * v0_hz**alpha

    T3 = (g2-1)**(2-p) - (g1-1)**(2-p)
    T4 = (g2-1)**(2*(1-alpha)) - (g1-1)**(2*(1-alpha))
    T5 = (g2-1)**(3-p) - (g1-1)**(3-p)
    T6 = T3 * T4 / T5

    T1 = 3 * L1 / (2 * C3 * (M_E * C_LIGHT**2)**(2*alpha - 1))
    T2 = (1 + X_FACTOR)/(1-alpha) * (3-p)/(2-p) * (sqrt(2/3)*C1)**(1-alpha)
    A = T1 * T2 * T6

    L = L1/(1-alpha) * (sqrt(2/3)*C1*(M_E*C_LIGHT**2)**2)**(1-alpha) * T4

    B_eq = ((4*math.pi*(1+alpha)*A)/V_cm3)**(1/(3+alpha)) \
           * (2/(1+alpha))**(1/(3+alpha))

    u_b = B_eq**2 / (8*math.pi)
    u_p = (alpha*A*L*B_eq**(-3/2)) / V_cm3
    u_eq = u_b + u_p

    U_eq = u_eq * V_cm3
    P_jet = U_eq / (t_age_myr * SEC_IN_MYR)

    return {
        "B_eq (μG)": B_eq * 1e6,
        "u_eq (erg/cm³)": u_eq,
        "U_eq (erg)": U_eq,
        "P_jet (erg/s)": P_jet,
        "Volume (kpc³)": V_kpc3,
        "D_L (Mpc)": DL,
        "D_A (Mpc)": DA,
        "Scale (kpc/\")": Sf
    }

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config("Lobe Magnetic Field Estimator v3", "🌌", "wide")
st.title("🌌 Lobe Magnetic Field Estimator v3")

with st.sidebar:
    st.header("Cosmology")
    H0 = st.number_input("H₀", 69.6)
    WM = st.number_input("Ωₘ", 0.286)
    WV = st.number_input("Ω_Λ", 0.714)

    st.header("Spectral Parameters")
    alpha = st.number_input("α", 0.8)
    g1 = st.number_input("γ₁", 100.0)
    g2 = st.number_input("γ₂", 1e5)
    v0 = st.number_input("ν₀ (MHz)", 1400.0)
    s_v0 = st.number_input("S(ν₀) (Jy)", 1.0)
    z = st.number_input("Redshift z", 0.1)

    st.header("Geometry")
    geometry = st.selectbox("Lobe model", ["Ellipsoid", "Cylinder"])

    dims = []
    if geometry == "Ellipsoid":
        dims.append((
            st.number_input("Lobe 1: l₁ (arcsec)", 20.0),
            st.number_input("Lobe 1: b₁ (arcsec)", 8.0),
            st.number_input("Lobe 1: h₁ (arcsec)", 6.0),
        ))
        dims.append((
            st.number_input("Lobe 2: l₂ (arcsec)", 18.0),
            st.number_input("Lobe 2: b₂ (arcsec)", 7.0),
            st.number_input("Lobe 2: h₂ (arcsec)", 5.0),
        ))
    else:
        dims.append((
            st.number_input("Lobe 1: r₁ (arcsec)", 6.0),
            st.number_input("Lobe 1: h₁ (arcsec)", 20.0),
        ))
        dims.append((
            st.number_input("Lobe 2: r₂ (arcsec)", 5.0),
            st.number_input("Lobe 2: h₂ (arcsec)", 18.0),
        ))

    st.header("Dynamics")
    t_age = st.number_input("Lobe age (Myr)", 10.0)

if st.button("Compute"):
    results = compute_source(
        alpha, g1, g2, v0, s_v0, z,
        geometry, dims, t_age,
        H0, WM, WV
    )
    df = pd.DataFrame(results, index=["Value"]).T
    df["Value"] = df["Value"].apply(lambda x: f"{x:.6e}")
    st.dataframe(df)

st.markdown("---")
st.markdown("**Created by Arnav Sharma — v3.0**")
