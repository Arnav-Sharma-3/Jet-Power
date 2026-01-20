# ============================================================
# Lobe Magnetic Field Estimator v3 (Jet Power Enabled)
# Geometry toggle + input table added
# ============================================================

import streamlit as st
import pandas as pd
import math
from math import sqrt, exp, sin

# --------------------------------------------------
# Constants (CGS)
# --------------------------------------------------
CGS_KPC = 3.08567758128e21
CGS_MPC = 3.08567758128e24
C1 = 6.266e18
C3 = 2.368e-3
M_E = 9.1093837139e-28
C_LIGHT = 2.99792458e10
X_FACTOR = 0.0

# --------------------------------------------------
# Cosmology
# --------------------------------------------------
def run_cosmology_calculator(z, H0, WM, WV):
    h = H0 / 100.0
    WR = 4.165e-5 / (h * h)
    WK = 1.0 - WM - WR - WV
    az = 1.0 / (1.0 + z)
    c = 299792.458

    n = 1000
    DCMR = 0.0
    for i in range(n):
        a = az + (1.0 - az) * (i + 0.5) / n
        adot = sqrt(WK + WM / a + WR / (a * a) + WV * a * a)
        DCMR += 1.0 / (a * adot)
    DCMR *= (1.0 - az) / n

    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        ratio = (0.5 * (exp(x) - exp(-x)) / x) if WK > 0 else sin(x) / x
    else:
        ratio = 1.0 + x * x / 6.0 + x**4 / 120.0

    DA = az * ratio * DCMR
    DA_Mpc = (c / H0) * DA
    DL_Mpc = DA_Mpc / (az * az)
    kpc_DA = DA_Mpc / 206.264806

    return dict(DL_Mpc=DL_Mpc, DA_Mpc=DA_Mpc, kpc_DA=kpc_DA)

# --------------------------------------------------
# Physics core (two-lobe geometry)
# --------------------------------------------------
def compute_fields_lobes(alpha, g1, g2, v0, s_v0, z, t_age, geometry,
                         H0, WM, WV,
                         l1=None, b1=None, h1=None, l2=None, b2=None, h2=None,
                         r1=None, r2=None):

    cosmo = run_cosmology_calculator(z, H0, WM, WV)
    D_l = cosmo["DL_Mpc"]
    D_a = cosmo["DA_Mpc"]
    Sf = cosmo["kpc_DA"]

    # ---------------- Geometry ----------------
    if geometry == "ellipsoid":
        dims = [l1, b1, h1, l2, b2, h2]
        if any(d is None for d in dims):
            raise ValueError("Ellipsoid geometry requires l1 b1 h1 l2 b2 h2")

        l1k, b1k, h1k = l1*Sf, b1*Sf, h1*Sf
        l2k, b2k, h2k = l2*Sf, b2*Sf, h2*Sf

        V_kpc3 = (4/3)*math.pi*(l1k*b1k*h1k + l2k*b2k*h2k)
        V_cm3 = V_kpc3 * (CGS_KPC**3)

        length = l1k + l2k
        breadth = max(b1k, b2k)
        width = max(h1k, h2k)

    elif geometry == "cylinder":
        dims = [r1, r2, h1, h2]
        if any(d is None for d in dims):
            raise ValueError("Cylinder geometry requires r1 h1 r2 h2")

        r1k, h1k = r1*Sf, h1*Sf
        r2k, h2k = r2*Sf, h2*Sf

        V_kpc3 = math.pi*(r1k**2*h1k + r2k**2*h2k)
        V_cm3 = V_kpc3 * (CGS_KPC**3)

        length = h1k + h2k
        breadth = max(2*r1k, 2*r2k)
        width = breadth

    else:
        raise ValueError("Invalid geometry")

    # ---------------- Synchrotron ----------------
    v0_hz = v0 * 1e6
    s_v0_cgs = s_v0 * 1e-23
    D_l_cm = D_l * CGS_MPC

    p = 2*alpha + 1
    L1 = 4*math.pi*D_l_cm**2 * s_v0_cgs * v0_hz**alpha

    T3 = (g2-1)**(2-p) - (g1-1)**(2-p)
    T4 = (g2-1)**(2*(1-alpha)) - (g1-1)**(2*(1-alpha))
    T5 = (g2-1)**(3-p) - (g1-1)**(3-p)
    T6 = T3*T4/T5

    T1 = 3*L1 / (2*C3*(M_E*C_LIGHT**2)**(2*alpha-1))
    T2 = ((1+X_FACTOR)/(1-alpha))*((3-p)/(2-p))*(sqrt(2/3)*C1)**(1-alpha)
    A = T1*T2*T6

    L = L1/(1-alpha)*(sqrt(2/3)*C1*(M_E*C_LIGHT**2)**2)**(1-alpha)*T4

    B_min = ((4*math.pi*(1+alpha)*A)/V_cm3)**(1/(3+alpha))
    B_eq = (2/(1+alpha))**(1/(3+alpha))*B_min

    u_B = B_eq**2/(8*math.pi)
    u_p = alpha*A*L*B_eq**(-1.5)/V_cm3
    u_tot = u_p + u_B

    U_eq = u_tot * V_cm3
    P_jet_W = (U_eq / (t_age*3.15576e7)) * 1e-7

    return dict(
        alpha=alpha, Bmin=B_min*1e6, Beq=B_eq*1e6,
        DL=D_l, DA=D_a, Sf=Sf,
        length=length, breadth=breadth, width=width,
        V=V_kpc3, L=L,
        up=u_p, uB=u_B, utot=u_tot,
        t_age=t_age, Pjet=P_jet_W
    )

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config("Jet Power &Lobe Magnetic Field Estimator v3", "🌌", layout="wide")
st.title("🌀 Jet Power & Lobe Magnetic Field Estimator v3")

# Sidebar (LOCKED)
with st.sidebar:
    st.header("Cosmology Parameters")
    H0 = st.number_input("Hubble Constant (H₀)", value=69.6)
    WM = st.number_input("Ω Matter (Ωₘ)", value=0.28600, format="%.5f")
    WV = st.number_input("Ω Vacuum (Ω_Λ)", value=0.71400, format="%.5f")

tab_single, tab_batch = st.tabs(["🔹 Single Source", "📂 Batch (CSV Upload)"])

# ============================================================
# SINGLE SOURCE
# ============================================================
with tab_single:
    st.subheader("Single Source")

    geometry = st.selectbox("Geometry", ["ellipsoid", "cylinder"])
    source = st.text_input("Source")
    alpha = st.number_input("α", value=0.7)
    g1 = st.number_input("γ₁", value=10.0)
    g2 = st.number_input("γ₂", value=1e5)
    v0 = st.number_input("ν₀ (MHz)", value=1400.0)
    s_v0 = st.number_input("S₀ (Jy)", value=1.0)
    z = st.number_input("Redshift (z)", value=0.1)
    t_age = st.number_input("t_age (years)", value=1e7, format="%.3e")

    if geometry == "ellipsoid":
        l1 = st.number_input("l1 (arcsec)")
        b1 = st.number_input("b1 (arcsec)")
        h1 = st.number_input("h1 (arcsec)")
        l2 = st.number_input("l2 (arcsec)")
        b2 = st.number_input("b2 (arcsec)")
        h2 = st.number_input("h2 (arcsec)")
        r1 = r2 = None
    else:
        r1 = st.number_input("r1 (arcsec)")
        h1 = st.number_input("h1 (arcsec)")
        r2 = st.number_input("r2 (arcsec)")
        h2 = st.number_input("h2 (arcsec)")
        l1 = b1 = l2 = b2 = None

    if st.button("Compute"):
        res = compute_fields_lobes(alpha, g1, g2, v0, s_v0, z, t_age, geometry,
                                   H0, WM, WV,
                                   l1, b1, h1, l2, b2, h2, r1, r2)

        # -------- Input table --------
        st.markdown("### 🔢 Input Parameters")
        inp = {
            "Source": source, "Geometry": geometry, "α": alpha,
            "γ₁": g1, "γ₂": g2, "ν₀ (MHz)": v0,
            "S₀ (Jy)": s_v0, "z": z, "t_age (yr)": t_age
        }
        st.dataframe(pd.DataFrame(inp.items(), columns=["Parameter","Value"]))

        # -------- Output table --------
        st.markdown("### 📊 Output Quantities")
        out = {
            "Source": source,
            "Redshift (z)": z,
            "Spectral Index (α)": res["alpha"],
            "B_min (μG)": res["Bmin"],
            "B_eq (μG)": res["Beq"],
            "D_L (Mpc)": res["DL"],
            "D_A (Mpc)": res["DA"],
            "Scale (kpc/\")": res["Sf"],
            "Length (kpc)": res["length"],
            "Breadth (kpc)": res["breadth"],
            "Width (kpc)": res["width"],
            "Volume (kpc³)": f"{res['V']:.8e}",
            "L (erg/s)": f"{res['L']:.8e}",
            "u_p (erg/cm³)": f"{res['up']:.8e}",
            "u_B (erg/cm³)": f"{res['uB']:.8e}",
            "u_total (erg/cm³)": f"{res['utot']:.8e}",
            "t_age (years)": res["t_age"],
            "Jet power (W)": f"{res['Pjet']:.8e}",
        }
        st.dataframe(pd.DataFrame(out.items(), columns=["Quantity","Value"]))

# ============================================================
# BATCH MODE
# ============================================================
with tab_batch:
    st.subheader("Batch Geometry Mode")
    batch_geometry_mode = st.radio(
        "Geometry handling:",
        ["From CSV", "Force Ellipsoid", "Force Cylinder"],
        horizontal=True
    )
st.markdown(
    """
    Upload a CSV/TSV file describing **two-lobed radio sources**.

    Each file must contain the **common columns**:  
    `Source, alpha, gamma1, gamma2, v0, s_v0, z, t_age, geometry`  
    where **v0** is in **MHz**, **s_v0** in **Jy**, **z** is redshift, **t_age** in **years**, 
    and **geometry** specifies how the lobe dimensions are interpreted.

    In addition to the common columns, **each row must include geometry-specific columns**:

    • If `geometry = ellipsoid`, the row must also contain  
    `l1, b1, h1, l2, b2, h2` (all angular dimensions in **arcsec**).

    • If `geometry = cylinder`, the row must also contain  
    `r1, h1, r2, h2` (radius and height in **arcsec**).

    The geometry selector above controls how this column is used:  
    *From CSV* reads geometry per row (mixed geometries allowed), while 
    *Force Ellipsoid* or *Force Cylinder* ignores the CSV geometry column and requires 
    the corresponding dimensions for all rows. Rows missing required columns are skipped.
    """
)
    
    file = st.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"])
    if file:
        sep = "\t" if file.name.endswith(("tsv","txt")) else ","
        df = pd.read_csv(file, sep=sep)

        rows = []
        for _, r in df.iterrows():
            geom = r["geometry"] if batch_geometry_mode=="From CSV" \
                   else ("ellipsoid" if batch_geometry_mode=="Force Ellipsoid" else "cylinder")

            res = compute_fields_lobes(
                r.alpha, r.gamma1, r.gamma2, r.v0, r.s_v0,
                r.z, r.t_age, geom, H0, WM, WV,
                r.get("l1"), r.get("b1"), r.get("h1"),
                r.get("l2"), r.get("b2"), r.get("h2"),
                r.get("r1"), r.get("r2")
            )

            rows.append([
                r.Source, r.z, res["alpha"], res["Bmin"], res["Beq"],
                res["DL"], res["DA"], res["Sf"],
                res["length"], res["breadth"], res["width"],
                f"{res['V']:.8e}", f"{res['L']:.8e}",
                f"{res['up']:.8e}", f"{res['uB']:.8e}",
                f"{res['utot']:.8e}", res["t_age"], f"{res['Pjet']:.8e}"
            ])

        cols = [
            "Source","Redshift (z)","Spectral Index (α)",
            "B_min (μG)","B_eq (μG)",
            "D_L (Mpc)","D_A (Mpc)","Scale (kpc/\")",
            "Length (kpc)","Breadth (kpc)","Width (kpc)",
            "Volume (kpc³)","L (erg/s)",
            "u_p (erg/cm³)","u_B (erg/cm³)","u_total (erg/cm³)",
            "t_age (years)","Jet power (W)"
        ]

        df_out = pd.DataFrame(rows, columns=cols)
        st.dataframe(df_out)

        st.download_button(
            "Download CSV",
            df_out.to_csv(index=False).encode(),
            "magnetic_fields_results.csv",
            "text/csv"
        )
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
