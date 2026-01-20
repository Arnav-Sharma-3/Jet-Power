import streamlit as st
import pandas as pd
import math
from math import sqrt, exp, sin

# --------------------------------------------------
# Constants for CGS conversions and synchrotron math
# --------------------------------------------------
CGS_KPC = 3.08567758128e21    # cm per kiloparsec
CGS_MPC = 3.08567758128e24    # cm per Megaparsec
C1 = 6.266e18                 # synchrotron constant
C3 = 2.368e-3                 # synchrotron constant
M_E = 9.1093837139e-28        # electron mass (g)
C_LIGHT = 2.99792458e10       # speed of light (cm/s)
X_FACTOR = 0.0                # proton/electron energy ratio

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_cosmology_calculator(z, H0, WM, WV):
    """Calculate cosmological distances from redshift"""
    h = H0 / 100.0
    WR = 4.165E-5 / (h * h)
    WK = 1.0 - WM - WR - WV
    az = 1.0 / (1.0 + z)
    c = 299792.458

    n = 1000  # Integration steps
    # Compute comoving radial distance integral
    DCMR = 0.0
    for i in range(n):
        a = az + (1.0 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DCMR += 1.0 / (a * adot)
    DCMR = (1.0 - az) * DCMR / n
    
    # Curvature factor
    x = math.sqrt(abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = 0.5 * (exp(x) - exp(-x)) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        ratio = 1.0 + y / 6.0 + y * y / 120.0
    
    DCMT = ratio * DCMR
    DA = az * DCMT
    c = 299792.458
    DA_Mpc = (c / H0) * DA
    DL = DA / (az * az)
    DL_Mpc = (c / H0) * DL
    kpc_DA = DA_Mpc / 206.264806  # Scale factor (kpc/arcsec)
    
    return {
        'DL_Mpc': DL_Mpc,       # Luminosity distance (Mpc)
        'DA_Mpc': DA_Mpc,       # Angular diameter distance (Mpc)
        'kpc_DA': kpc_DA        # Scale factor (kpc/arcsec)
    }

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_fields_lobes(alpha, g1, g2, v0, s_v0, z, t_age, geometry, H0=69.6, WM=0.286, WV=0.714,
                         l1=None, b1=None, h1=None, l2=None, b2=None, h2=None,
                         r1=None, r2=None):
    """
    Compute magnetic fields and energetics for a source with two lobes.
    geometry = "ellipsoid" or "cylinder".
    If ellipsoid: uses l1,b1,h1 and l2,b2,h2 (semi-axes in arcsec).
    If cylinder: uses r1, h1 and r2, h2 (radius and height in arcsec).
    """
    cosmo = run_cosmology_calculator(z, H0, WM, WV)
    D_l = cosmo['DL_Mpc']       # Luminosity distance (Mpc)
    D_a = cosmo['DA_Mpc']       # Angular diameter distance (Mpc)
    Sf = cosmo['kpc_DA']        # Scale factor (kpc/arcsec)

    if geometry.lower() == "ellipsoid":
        l1_kpc = l1 * Sf
        b1_kpc = b1 * Sf
        h1_kpc = h1 * Sf
        l2_kpc = l2 * Sf
        b2_kpc = b2 * Sf
        h2_kpc = h2 * Sf
        V1_kpc3 = (4.0/3.0) * math.pi * l1_kpc * b1_kpc * h1_kpc
        V2_kpc3 = (4.0/3.0) * math.pi * l2_kpc * b2_kpc * h2_kpc
        V_kpc3 = V1_kpc3 + V2_kpc3

        l1_cm = l1_kpc * CGS_KPC
        b1_cm = b1_kpc * CGS_KPC
        h1_cm = h1_kpc * CGS_KPC
        l2_cm = l2_kpc * CGS_KPC
        b2_cm = b2_kpc * CGS_KPC
        h2_cm = h2_kpc * CGS_KPC
        V1_cm3 = (4.0/3.0) * math.pi * l1_cm * b1_cm * h1_cm
        V2_cm3 = (4.0/3.0) * math.pi * l2_cm * b2_cm * h2_cm
        V_cm3 = V1_cm3 + V2_cm3

        length_kpc = l1_kpc + l2_kpc
        breadth_kpc = max(b1_kpc, b2_kpc)
        width_kpc = max(h1_kpc, h2_kpc)

    elif geometry.lower() == "cylinder":
        r1_kpc = r1 * Sf
        h1_kpc = h1 * Sf
        r2_kpc = r2 * Sf
        h2_kpc = h2 * Sf
        V1_kpc3 = math.pi * (r1_kpc**2) * h1_kpc
        V2_kpc3 = math.pi * (r2_kpc**2) * h2_kpc
        V_kpc3 = V1_kpc3 + V2_kpc3

        r1_cm = r1_kpc * CGS_KPC
        h1_cm = h1_kpc * CGS_KPC
        r2_cm = r2_kpc * CGS_KPC
        h2_cm = h2_kpc * CGS_KPC
        V1_cm3 = math.pi * (r1_cm**2) * h1_cm
        V2_cm3 = math.pi * (r2_cm**2) * h2_cm
        V_cm3 = V1_cm3 + V2_cm3

        length_kpc = h1_kpc + h2_kpc
        breadth_kpc = max(2.0*r1_kpc, 2.0*r2_kpc)
        width_kpc = breadth_kpc

    else:
        raise ValueError("Invalid geometry: choose 'ellipsoid' or 'cylinder'")

    # Convert frequencies and flux
    v0_hz = v0 * 1e6
    s_v0_cgs = s_v0 * 1e-23

    # Synchrotron/equipartition calculations
    p = 2.0 * alpha + 1.0
    L1 = 4.0 * math.pi * (D_l * CGS_MPC)**2 * s_v0_cgs * (v0_hz**alpha)

    T3 = (g2 - 1.0)**(2.0 - p) - (g1 - 1.0)**(2.0 - p)
    T4 = (g2 - 1.0)**(2.0 * (1.0 - alpha)) - (g1 - 1.0)**(2.0 * (1.0 - alpha))
    T5 = (g2 - 1.0)**(3.0 - p) - (g1 - 1.0)**(3.0 - p)
    T6 = T3 * T4 / T5

    T1 = 3.0 * L1 / (2.0 * C3 * (M_E * C_LIGHT**2)**(2.0 * alpha - 1.0))
    T2 = ((1.0 + X_FACTOR) / (1.0 - alpha)) * ((3.0 - p) / (2.0 - p)) * ((math.sqrt(2.0/3.0) * C1)**(1.0 - alpha))
    A = T1 * T2 * T6

    L = L1 / (1.0 - alpha) * ((math.sqrt(2.0/3.0) * C1 * (M_E * C_LIGHT**2)**2)**(1.0 - alpha)) * T4

    B_min = ((4.0 * math.pi * (1.0 + alpha) * A) / V_cm3)**(1.0 / (3.0 + alpha))
    B_eq = (2.0 / (1.0 + alpha))**(1.0 / (3.0 + alpha)) * B_min

    u_B = B_eq**2 / (8.0 * math.pi)
    u_p = (alpha * A * L * B_eq**(-1.5)) / V_cm3
    u_tot = u_p + u_B

    U_eq = u_tot * V_cm3
    age_seconds = t_age * 3.15576e7
    P_erg_s = U_eq / age_seconds if age_seconds > 0 else float('inf')
    P_watt = P_erg_s * 1.0e-7

    return alpha, B_min * 1e6, B_eq * 1e6, L, u_p, u_B, u_tot, D_l, D_a, Sf, length_kpc, breadth_kpc, width_kpc, V_kpc3, t_age, P_watt

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit App Layout
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Lobe Magnetic Field Estimator v3",
    page_icon="🌌",  
    layout="wide"
)
st.title("🌀 Lobe Magnetic Field Estimator v3 (Cosmology Calculator Integrated)")

# Cosmology parameters in sidebar
with st.sidebar:
    st.header("Cosmology Parameters")
    H0 = st.number_input("Hubble Constant (H₀)", value=69.6)
    WM = st.number_input("Ω Matter (Ωₘ)", value=0.28600, format="%.5f")
    WV = st.number_input("Ω Vacuum (Ω_Λ)", value=0.71400, format="%.5f")

# -------------------------------
# TABS: Single Source  |  Batch
# -------------------------------
tab_single, tab_batch = st.tabs(["🔹 Single Source", "📂 Batch (CSV Upload)"])

with tab_single:
    st.subheader("Single Source (enter values)")
    geometry = st.selectbox("Geometry", options=["Ellipsoid", "Cylinder"])
    source = st.text_input("Source")
    alpha = st.number_input("Spectral Index (α)", value=0.5)
    gamma1 = st.number_input("Gamma 1 (γ₁)", value=10.0)
    gamma2 = st.number_input("Gamma 2 (γ₂)", value=10.0)
    v0 = st.number_input("ν₀ (MHz)", value=1400.0)
    s_v0 = st.number_input("S₀ (Jy)", value=1.0)
    z = st.number_input("Redshift (z)", value=0.1)
    t_age = st.number_input("Source age (years)", value=1e7, format="%.6e")
    if geometry == "Ellipsoid":
        l1 = st.number_input("l₁ (arcsec)", value=10.0)
        b1 = st.number_input("b₁ (arcsec)", value=5.0)
        h1 = st.number_input("h₁ (arcsec)", value=5.0)
        l2 = st.number_input("l₂ (arcsec)", value=10.0)
        b2 = st.number_input("b₂ (arcsec)", value=5.0)
        h2 = st.number_input("h₂ (arcsec)", value=5.0)
        r1 = r2 = None
    else:
        r1 = st.number_input("r₁ (arcsec)", value=5.0)
        h1 = st.number_input("h₁ (arcsec)", value=10.0)
        r2 = st.number_input("r₂ (arcsec)", value=5.0)
        h2 = st.number_input("h₂ (arcsec)", value=10.0)
        l1 = b1 = h1 = l2 = b2 = h2 = None

    if st.button("Compute single source"):
        try:
            res = compute_fields_lobes(alpha, gamma1, gamma2, v0, s_v0, z, t_age, geometry, H0, WM, WV,
                                       l1=l1, b1=b1, h1=h1, l2=l2, b2=b2, h2=h2, r1=r1, r2=r2)
            alpha_out, B_min_uG, B_eq_uG, L_val, u_p, u_B, u_tot, D_l, D_a, Sf, length_kpc, breadth_kpc, width_kpc, V_kpc3, t_age_out, P_watt = res

            header = (
                "Source\tRedshift (z)\tSpectral Index (α)\tB_min (μG)\tB_eq (μG)"
                "\tD_L (Mpc)\tD_A (Mpc)\tScale (kpc/\")\tLength (kpc)\tBreadth (kpc)"
                "\tWidth (kpc)\tVolume (kpc³)\tL (erg/s)\tu_p (erg/cm³)\tu_B (erg/cm³)\tu_total (erg/cm³)\tt_age (years)\tJet power (W)"
            )
            row = (
                f"{source}\t"
                f"{z:.8f}\t"
                f"{alpha_out:.8f}\t"
                f"{B_min_uG:.8f}\t"
                f"{B_eq_uG:.8f}\t"
                f"{D_l:.8f}\t"
                f"{D_a:.8f}\t"
                f"{Sf:.8f}\t"
                f"{length_kpc:.8f}\t"
                f"{breadth_kpc:.8f}\t"
                f"{width_kpc:.8f}\t"
                f"{V_kpc3:.8e}\t"
                f"{L_val:.8e}\t"
                f"{u_p:.8e}\t"
                f"{u_B:.8e}\t"
                f"{u_tot:.8e}\t"
                f"{t_age_out:.8f}\t"
                f"{P_watt:.8e}"
            )

            st.code(header, language=None)
            st.code(row, language=None)

            import io
            df_single = pd.read_csv(io.StringIO(header.replace("\t", ",") + "\n" + row.replace("\t", ",")))
            sci_cols = ["Volume (kpc³)", "L (erg/s)", "u_p (erg/cm³)", "u_B (erg/cm³)", "u_total (erg/cm³)", "Jet power (W)"]
            for col in sci_cols:
                if col in df_single.columns:
                    df_single[col] = df_single[col].apply(lambda x: f"{float(x):.8e}")
            st.dataframe(df_single)
        except Exception as e:
            st.error(str(e))

with tab_batch:
    st.markdown(
        """
        Upload a CSV/TSV with columns:  
        `Source, alpha, gamma1, gamma2, v0, s_v0, z, t_age, geometry`  
        — where **geometry** = "ellipsoid" or "cylinder".  
        If `ellipsoid`: include `l1, b1, h1, l2, b2, h2`.  
        If `cylinder`: include `r1, h1, r2, h2`.  
        **l1, b1, h1, l2, b2, h2, r1, r2, h1, h2** are in **arcsec**,  
        **z** is redshift, **ν0** in **MHz**, **S0** in **Jy**, **t_age** in **years**.
        """
    )

    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "tsv", "txt"])
    if uploaded_file:
        sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
        try:
            df = pd.read_csv(uploaded_file, sep=sep, comment="#")
        except Exception as e:
            st.error(f"📂 Could not read file: {e}")
        else:
            required = ["Source","alpha","gamma1","gamma2","v0","s_v0","z","t_age","geometry"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                numeric_cols = ["alpha","gamma1","gamma2","v0","s_v0","z","t_age","l1","b1","h1","l2","b2","h2","r1","r2"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                invalid_rows = []
                for idx, row in df.iterrows():
                    geom = str(row.get("geometry", "")).strip().lower()
                    if geom not in ["ellipsoid", "cylinder"]:
                        invalid_rows.append(idx)
                        continue
                    common_fields = ["alpha","gamma1","gamma2","v0","s_v0","z","t_age"]
                    if any(pd.isna(row.get(c)) for c in common_fields):
                        invalid_rows.append(idx)
                        continue
                    if geom == "ellipsoid":
                        needed = ["l1","b1","h1","l2","b2","h2"]
                    else:
                        needed = ["r1","r2","h1","h2"]
                    if any(pd.isna(row.get(c)) for c in needed):
                        invalid_rows.append(idx)
                if invalid_rows:
                    st.warning(f"⚠️ {len(invalid_rows)} rows contain missing or invalid values and will be skipped")
                    st.dataframe(df.loc[invalid_rows])
                    df = df.drop(index=invalid_rows)
                if df.empty:
                    st.error("❌ No valid data remaining after cleaning. Please check your input file.")
                else:
                    results = []
                    for idx, row in df.iterrows():
                        geom = str(row["geometry"]).strip().lower()
                        try:
                            if geom == "ellipsoid":
                                res = compute_fields_lobes(
                                    row["alpha"], row["gamma1"], row["gamma2"],
                                    row["v0"], row["s_v0"], row["z"], row["t_age"],
                                    "ellipsoid", H0, WM, WV,
                                    l1=row["l1"], b1=row["b1"], h1=row["h1"],
                                    l2=row["l2"], b2=row["b2"], h2=row["h2"]
                                )
                            else:
                                res = compute_fields_lobes(
                                    row["alpha"], row["gamma1"], row["gamma2"],
                                    row["v0"], row["s_v0"], row["z"], row["t_age"],
                                    "cylinder", H0, WM, WV,
                                    r1=row["r1"], r2=row["r2"], h1=row["h1"], h2=row["h2"]
                                )
                            results.append(res)
                        except Exception:
                            continue
                    df = df.reset_index(drop=True)
                    alphas = [r[0] for r in results]
                    Bmins = [r[1] for r in results]
                    Beqs = [r[2] for r in results]
                    Ls = [r[3] for r in results]
                    ups = [r[4] for r in results]
                    uBs = [r[5] for r in results]
                    utots = [r[6] for r in results]
                    DLs = [r[7] for r in results]
                    DAs = [r[8] for r in results]
                    Sfs = [r[9] for r in results]
                    lengths = [r[10] for r in results]
                    breadths = [r[11] for r in results]
                    widths = [r[12] for r in results]
                    Vols = [r[13] for r in results]
                    ages = [r[14] for r in results]
                    Pjets = [r[15] for r in results]

                    df_out = pd.DataFrame({
                        "Source": df["Source"],
                        "Redshift (z)": df["z"].round(8),
                        "Spectral Index (α)": [round(a,8) for a in alphas],
                        "B_min (μG)": [round(b,8) for b in Bmins],
                        "B_eq (μG)": [round(b,8) for b in Beqs],
                        "D_L (Mpc)": [round(d,8) for d in DLs],
                        "D_A (Mpc)": [round(d,8) for d in DAs],
                        "Scale (kpc/\")": [round(s,8) for s in Sfs],
                        "Length (kpc)": [round(l,8) for l in lengths],
                        "Breadth (kpc)": [round(b,8) for b in breadths],
                        "Width (kpc)": [round(w,8) for w in widths],
                        "Volume (kpc³)": [f"{v:.8e}" for v in Vols],
                        "L (erg/s)": [f"{lval:.8e}" for lval in Ls],
                        "u_p (erg/cm³)": [f"{up:.8e}" for up in ups],
                        "u_B (erg/cm³)": [f"{ub:.8e}" for ub in uBs],
                        "u_total (erg/cm³)": [f"{ut:.8e}" for ut in utots],
                        "t_age (years)": [round(age,8) for age in ages],
                        "Jet power (W)": [f"{p:.8e}" for p in Pjets]
                    })

                    st.success("✅ Calculation complete!")
                    st.dataframe(df_out)

                    csv_data = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📅 Download Results (CSV)",
                        data=csv_data,
                        file_name="magnetic_fields_results.csv",
                        mime="text/csv"
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
)
