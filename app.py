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

        st.dataframe(
            pd.DataFrame(
                out.items(),
                columns=["Quantity", "Value"]
            )
        )
# ============================================================
# A: SINGLE SOURCE (emd)
# ============================================================

# ============================================================
# B: MULTI SOURCE (start)
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
        Upload a CSV/TSV file describing **single radio lobes**.

        Each file must contain the **common columns**:  
        Source, alpha, gamma1, gamma2, v0, s_v0, z, t_age, geometry  
        where **v0** is in **MHz**, **s_v0** in **Jy**, **z** is redshift, **t_age** in **years**, 
        and **geometry** specifies how the lobe dimensions are interpreted.

        In addition to the common columns, **each row must include geometry-specific columns**:

        • If geometry = ellipsoid, the row must also contain  
        l, b, h (all angular dimensions in **arcsec**).

        • If geometry = cylinder, the row must also contain  
        r, h (radius and height in **arcsec**).

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
                r.get("l"), r.get("b"), r.get("h"),
                r.get("r")
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
# ============================================================
# B: MULTI SOURCE (END)
# ============================================================

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
    """
    ,
