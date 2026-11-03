"""app.py — NW Camelids Genomics Platform (Dual Mode + 21 onglets)"""
import gzip, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import re

from config_nw import SPECIES_CONFIG, MAP_SOURCES, REFERENCE, get_config
from core      import (parse_map, parse_ped, parse_vcf, load_pheno, match_pheno,
                        rec_fid, _ca, apply_qc, miss_ind, miss_snp,
                        build_plink, build_vcf, impute_mean)
from analyses  import *  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🦙 NW Camelids · VicPac4",
                   page_icon="🦙", layout="wide",
                   initial_sidebar_state="expanded")

_KEYS = [
    "gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
    "gt_pruned", "snp_pruned", "gt_pre", "ind_pre", "snp_pre", "qc_stats",
    "pca_s", "pca_v", "mds_c", "king_K",
    "fst", "fst_M", "fst_pops", "fst_snp_df",
    "admix_Q", "admix_K", "cv_res", "ld_df", "roh_df", "froh", "ne_dict",
    "reyn_D", "reyn_pops", "nj_result", "nj_mid",
    "win_fst", "taj_df", "css_df",
    "pheno_df", "gwas_res", "lambda_gc",
    "gebv_df", "gblup_stats",
    "vep_res", "ncbi_gene_res", "sel_ann", "sel_gff", "ann_df",
    "mapping_df", "pair_res", "current_species", "grouping_mode",
    "ind_df_pays", "ind_df_species",
]
for _k in _KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None

def reset_derived():
    for k in _KEYS:
        if k not in ("gt", "ind_df", "snp_df", "current_species",
                     "grouping_mode", "ind_df_pays", "ind_df_species"):
            st.session_state[k] = None

@st.cache_data(show_spinner=False)
def load_map_uploaded(file_bytes: bytes, fname: str) -> pd.DataFrame:
    raw = gzip.decompress(file_bytes) if fname.endswith(".gz") else file_bytes
    return parse_map(raw)

@st.cache_data(show_spinner=False)
def compute_diversity_metrics(gt, labels):
    pops = np.unique(labels)
    results = []
    for p in pops:
        mask = labels == p
        sub_gt = gt[mask]
        if sub_gt.shape[0] == 0: continue
        n_ind = sub_gt.shape[0]
        p_freq = np.nanmean(sub_gt, axis=0) / 2.0
        maf = np.nanmean(np.minimum(p_freq, 1 - p_freq))
        ho = np.nanmean(np.nanmean(sub_gt == 1, axis=0))
        he = np.nanmean(2 * p_freq * (1 - p_freq))
        results.append({"Population": p, "N": n_ind, "Ho": round(float(ho), 4),
                        "He": round(float(he), 4), "MAF": round(float(maf), 4)})
    return pd.DataFrame(results)

def main():
    with st.sidebar:
        st.title("🦙 NW Camelids")
        st.caption(f"Réf. {REFERENCE['name']} · {REFERENCE['accession']}")

        species_options = ["🌍 Toutes les espèces (Global)"] + list(SPECIES_CONFIG.keys())
        species_key = st.selectbox("🐾 Mode d'analyse", species_options)
        
        if species_key == "🌍 Toutes les espèces (Global)":
            cfg = get_config("🦙 Alpaga (Vicugna pacos)")
            cfg["prefix"] = "nwc_global"
            cfg["autosomes"] = [str(i) for i in range(1, 39)]
            st.info("🌍 Mode Global : Analyse des 374 individus.")
        else:
            cfg = get_config(species_key)
            st.info(f"Mode spécifique : {species_key}")

        st.markdown(f"**Caryotype** : {cfg['karyotype']}")
        st.markdown(f"**Taxon NCBI** : `{cfg['taxon']}`")

        if st.session_state.current_species is None:
            st.session_state.current_species = species_key
        elif st.session_state.current_species != species_key:
            st.session_state.current_species = species_key
            reset_derived()
            st.session_state.gt = st.session_state.ind_df = st.session_state.snp_df = None
            st.warning("🔄 Mode changé — réinitialisé.")

        # ============================================================
        # SÉLECTEUR : PAYS vs ESPÈCE
        # ============================================================
        st.divider()
        st.header("🎯 Mode de regroupement")
        grouping_mode = st.radio(
            "Regrouper les analyses par :",
            ["🌍 Par Pays (ARG, BOL, PER, USA)", "🦙 Par Espèce (Alpaga, Guanaco, ...)"],
            index=0,
            help="Basculez entre les deux niveaux d'analyse sans recharger les fichiers."
        )
        st.session_state.grouping_mode = grouping_mode

        # Appliquer le mode courant
        if grouping_mode.startswith("🌍") and st.session_state.get("ind_df_pays") is not None:
            st.session_state.ind_df = st.session_state.ind_df_pays.copy()
        elif grouping_mode.startswith("🦙") and st.session_state.get("ind_df_species") is not None:
            st.session_state.ind_df = st.session_state.ind_df_species.copy()

        st.divider()
        st.header("📁 1. Annotations SNP")
        f_map = st.file_uploader(".map", type=["map", "gz"], key="map_up")
        if f_map and st.button("📥 Charger .map", use_container_width=True):
            try:
                st.session_state.snp_df = load_map_uploaded(f_map.read(), f_map.name)
                st.success(f"✅ {len(st.session_state.snp_df):,} SNP")
            except Exception as e:
                st.error(f"❌ {e}")

        st.divider()
        st.header("🧬 2. Génotypes")
        gmode = st.radio("Format", ["PED/MAP", "VCF"], horizontal=True)
        if gmode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped", "txt", "gz"], key="ped_up")
            if (ped and st.session_state.snp_df is not None
                    and st.button("📥 Charger PED", use_container_width=True)):
                try:
                    with st.spinner("Parsing…"):
                        gt, ind, a1, a2 = parse_ped(ped.read(), len(st.session_state.snp_df))
                    snp = st.session_state.snp_df.copy()
                    if len(snp) > gt.shape[1]: snp = snp.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(snp) < gt.shape[1]: gt = gt[:, :len(snp)]
                    snp["A1"] = a1; snp["A2"] = a2
                    reset_derived()
                    ind = rec_fid(ind)
                    st.session_state.ind_df_pays = ind.copy()
                    st.session_state.ind_df_species = None
                    st.session_state.update({"gt": gt, "ind_df": ind.copy(), "snp_df": snp})
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            vf = st.file_uploader(".vcf", type=["vcf", "gz"], key="vcf_up")
            if vf and st.button("📥 Charger VCF", use_container_width=True):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    reset_derived()
                    ind = rec_fid(ind)
                    st.session_state.ind_df_pays = ind.copy()
                    st.session_state.update({"gt": gt, "ind_df": ind.copy(), "snp_df": snp})
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")

        # ============================================================
        # METADONNEES EXCEL
        # ============================================================
        st.divider()
        st.header("📋 2.5 Métadonnées (Espèces)")
        st.caption("Uploadez le fichier Excel pour activer le mode 🦙 Par Espèce.")
        meta_file = st.file_uploader("NWC_Meta_Sample_Details_Suheil.xlsx", type=["xlsx", "csv"], key="meta_up")
        
        if meta_file and st.session_state.ind_df_pays is not None and st.button("🔗 Lier Métadonnées", use_container_width=True):
            try:
                if meta_file.name.endswith(".xlsx"):
                    df_excel = pd.read_excel(meta_file, sheet_name="Sample_Details", header=0)
                    meta = df_excel.iloc[:, [8, 17]].copy()
                    meta.columns = ["Sample_Filename", "Breed"]
                    meta = meta.dropna(subset=["Sample_Filename"])
                    meta = meta[meta["Sample_Filename"].astype(str).str.contains(".CEL", na=False)]

                    def map_breed_to_species(breed):
                        b = str(breed).upper()
                        if "HUACAYA" in b or "SURI" in b or "ALPACA" in b: return "Alpaga"
                        if "GUANACO" in b: return "Guanaco"
                        if "LLAMA" in b: return "Lama"
                        if "V.MENSALIS" in b or "MENSALIS" in b: return "Vigogne (mensalis)"
                        if "V.VICUGNA" in b or "VICUGNA VICUGNA" in b: return "Vigogne (vicugna)"
                        if "VICUGNA" in b or "V.SP" in b: return "Vigogne (sp.)"
                        return "Autre"

                    meta["Species"] = meta["Breed"].apply(map_breed_to_species)
                    meta_df = meta[["Sample_Filename", "Species"]].copy()
                else:
                    meta_df = pd.read_csv(meta_file)

                ind_pays = st.session_state.ind_df_pays.copy()
                ind_species = ind_pays.merge(
                    meta_df[["Sample_Filename", "Species"]],
                    left_on="IID", right_on="Sample_Filename", how="left"
                )
                ind_species["FID"] = ind_species["Species"].fillna(ind_species["FID"])
                ind_species = ind_species.drop(columns=["Species", "Sample_Filename"], errors="ignore")
                st.session_state.ind_df_species = ind_species
                
                if grouping_mode.startswith("🦙"):
                    st.session_state.ind_df = ind_species.copy()

                st.success("✅ Métadonnées liées ! Mode 🦙 Par Espèce disponible.")
                st.write("Répartition par espèce :")
                st.write(ind_species["FID"].value_counts())
                st.dataframe(ind_species[["IID", "FID"]].head(5))
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

        st.divider()
        st.header("⚙️ 3. Seuils QC")
        keep_auto = st.checkbox("Exclure X/Y/MT", value=True)
        rem_out   = st.checkbox("Supprimer outliers hét.", value=True)
        geno      = st.slider("Missing SNP", 0.0, 0.5, 0.05)
        mind      = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t     = st.slider("MAF",         0.0, 0.5, 0.05)
        hwe_t     = st.number_input("HWE p", value=1e-6, format="%.0e")
        het_sd    = st.slider("Het ±σ",     1.0, 5.0, 3.0, 0.1)
        ld_r2     = st.slider("LD r²",       0.05, 0.5, 0.2, 0.05)

    # ============================================================
    # PAGE PRINCIPALE
    # ============================================================
    current_mode = st.session_state.grouping_mode or "🌍 Par Pays"
    st.title(f"🦙 {species_key}")
    st.caption(f"Référence **{REFERENCE['name']}** · Groupement : **{current_mode}**")

    if st.session_state.gt is None:
        st.info("👋 Chargez `.map` et `.ped`, puis le fichier Excel pour activer le mode Espèce.")
        return

    if st.session_state.ind_df is None:
        st.warning("⚠️ Aucun regroupement actif. Rechargez le PED.")
        return

    TABS = st.tabs([
        "🏠 Accueil", "🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture",
        "📈 Démographie", "🌳 Phylogénie", "🔍 Sélection",
        "🧪 GWAS", "🎯 Post-GWAS", "📈 GBLUP", "🧬 Annotation",
        "🧬 VEP & NCBI", "🧬 Structure 3D", "🧬 Alignement",
        "🔬 Méta-analyse", "🧠 ML", "🧬 Haplotypes", "🌍 LiftOver", "📤 Export",
        "📊 Diversité",
    ])

    # --- 0 ACCUEIL ---
    with TABS[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Individus",   st.session_state.gt.shape[0])
        c2.metric("SNP",         st.session_state.gt.shape[1])
        c3.metric("Chromosomes", st.session_state.snp_df["CHR"].nunique())
        c4.metric("Groupes",     st.session_state.ind_df["FID"].nunique())
        ch = st.session_state.snp_df["CHR"].astype(str).value_counts().sort_index()
        st.plotly_chart(px.bar(x=ch.index, y=ch.values, labels={"x": "Chromosome", "y": "Nb SNP"}),
                        use_container_width=True)
        st.subheader(f"📋 Répartition par {current_mode}")
        st.dataframe(st.session_state.ind_df["FID"].value_counts().reset_index(), use_container_width=True)
        st.subheader("📋 Métadonnées")
        st.dataframe(st.session_state.ind_df.head(20), use_container_width=True)

    # --- 1 QC ---
    with TABS[1]:
        if st.button("▶ Lancer QC", type="primary", key="qc_btn"):
            try:
                params = {"geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t, "het_sd": het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(
                    st.session_state.gt, st.session_state.ind_df,
                    st.session_state.snp_df, params, cfg["autosomes"],
                    keep_autosomes=keep_auto, rem_out=rem_out)
                st.session_state.update({"gt_filt": gf, "ind_filt": inf, "snp_filt": snf, "qc_stats": qs})
                st.success(f"✅ {gf.shape[0]} × {gf.shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind. finaux", s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNP finaux",  s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Outliers",    s["trace"].get("het", 0))
            c4.metric("Exclus X/Y/MT", s["trace"].get("autosomes", 0))
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt), miss_snp(st.session_state.gt)),
                            use_container_width=True)
            trace_df = pd.DataFrame(list(s["trace"].items()), columns=["Étape", "Exclus"])
            st.dataframe(trace_df, use_container_width=True)
            st.download_button("⬇ QC rapport", trace_df.to_csv(index=False).encode(),
                               "qc_report.csv", "text/csv")

    # --- 2 LD ---
    with TABS[2]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ LD Pruning", key="ld_btn"):
                keep = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.update({
                    "gt_pruned": st.session_state.gt_filt[:, keep],
                    "snp_pruned": st.session_state.snp_filt[keep].reset_index(drop=True)})
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNP")
            if st.session_state.gt_pruned is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Avant", st.session_state.gt_filt.shape[1])
                c2.metric("Après", st.session_state.gt_pruned.shape[1])
                c3.metric("Réduction",
                          f"{100*(1-st.session_state.gt_pruned.shape[1]/st.session_state.gt_filt.shape[1]):.1f}%")
                if st.button("▶ LD Decay", key="ld_decay_btn"):
                    st.session_state.ld_df = ld_decay(
                        st.session_state.gt_pruned, st.session_state.snp_pruned)
                if st.session_state.ld_df is not None:
                    f = plot_ld(st.session_state.ld_df)
                    if f: st.plotly_chart(f, use_container_width=True)
                    st.dataframe(st.session_state.ld_df, use_container_width=True)

    # --- 3 STRUCTURE ---
    with TABS[3]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS", key="struct_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                s, v = pca_analysis(gu, 10)
                st.session_state.update({"pca_s": s, "pca_v": v})
                st.session_state.mds_c = mds_analysis(gu, 5)
                st.session_state.king_K = king_mat(gu)
                st.success("✅")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v,
                                      st.session_state.ind_filt["FID"].values),
                            use_container_width=True)
            st.plotly_chart(plot_mds(st.session_state.mds_c,
                                      st.session_state.ind_filt["FID"].values),
                            use_container_width=True)

    # --- 4 ADMIXTURE ---
    with TABS[4]:
        if st.session_state.gt_filt is not None:
            c1, c2 = st.columns(2)
            kmin = c1.slider("K min", 2, 5, 2)
            kmax = c2.slider("K max", 4, 10, 8)
            if st.button("📈 Évaluer tous les K", key="eval_k"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                st.session_state.cv_res = choose_best_k(gu, kmin, kmax)
            if st.session_state.cv_res is not None:
                st.dataframe(st.session_state.cv_res, use_container_width=True)
                st.plotly_chart(px.line(st.session_state.cv_res, x="K", y="RMSE", markers=True),
                                use_container_width=True)
            K = st.slider("K choisi", 2, 10, 4)
            if st.button("▶ Lancer NMF", key="admix_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    Q, _ = admixture_nmf(gu, K=K)
                    st.session_state.update({"admix_Q": Q, "admix_K": K})
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                f, qdf = plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values,
                                    st.session_state.ind_filt["FID"].values, st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True)
                st.dataframe(qdf.head(20), use_container_width=True)

    # --- 5 DÉMOGRAPHIE ---
    with TABS[5]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ ROH + Ne", key="dem_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    roh, fr = detect_roh(gu, su[["CHR", "BP"]].to_json())
                    st.session_state.update({"roh_df": roh, "froh": fr})
                    ne = {}
                    for f in st.session_state.ind_filt["FID"].unique():
                        m = (st.session_state.ind_filt["FID"] == f).values
                        if m.sum() >= 5:
                            ne[str(f)] = ne_historical(gu[m], su)
                    st.session_state.ne_dict = ne
                    st.success(f"✅ {len(roh)} ROH")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.froh is not None:
                df_f = pd.DataFrame({"FROH": st.session_state.froh,
                                     "Pop": st.session_state.ind_filt["FID"].values})
                st.plotly_chart(px.histogram(df_f, x="FROH", color="Pop", nbins=50),
                                use_container_width=True)
                st.dataframe(df_f.groupby("Pop")["FROH"].agg(
                    ["mean", "std", "count"]).round(4), use_container_width=True)
                if st.session_state.ne_dict:
                    fig = go.Figure()
                    for pop, ne_df in st.session_state.ne_dict.items():
                        if ne_df is not None and not ne_df.empty:
                            fig.add_trace(go.Scatter(x=ne_df["GenAgo"], y=ne_df["Ne"],
                                                      mode="lines", name=pop))
                    fig.update_layout(xaxis_title="Générations (ago)",
                                      yaxis_title="Ne", xaxis_type="log",
                                      yaxis_type="log", height=500)
                    st.plotly_chart(fig, use_container_width=True)

    # --- 6 PHYLOGÉNIE ---
    with TABS[6]:
        if st.session_state.gt_filt is not None:
            nb = st.slider("Bootstrap", 10, 500, 100, 10, key="nj_boot")
            if st.button("▶ UPGMA", key="phylo_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    lab = st.session_state.ind_filt["FID"].astype(str).values
                    with st.spinner(f"UPGMA + {nb} boot…"):
                        r = upgma_bootstrap(gu, su, lab, n_boot=int(nb))
                    st.session_state.nj_result = r
                    st.session_state.nj_mid = midpoint_info(r["D"], r["pops"])
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.nj_result is not None:
                r = st.session_state.nj_result
                mid = st.session_state.nj_mid
                st.plotly_chart(
                    plot_tree(r["Z"], r["pops"], r["support"],
                              title=f"UPGMA ({nb} boot)", mid=mid),
                    use_container_width=True)
                st.code(r["newick"], language="text")

    # --- 7 SÉLECTION ---
    with TABS[7]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            lab = st.session_state.ind_filt["FID"].astype(str).values
            use_wc = st.checkbox("Weir & Cockerham", value=True)
            st.subheader("🧬 FST")
            if st.button("▶ FST SNP + Pairwise", key="fst_btn"):
                try:
                    st.session_state.fst = fst_snp_wc(gu, lab)
                    st.session_state.fst_snp_df = su.copy()
                    M, pops = fst_pairwise(gu, lab, use_wc84=use_wc)
                    st.session_state.update({"fst_M": M, "fst_pops": pops})
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.fst is not None:
                sf = st.session_state.fst_snp_df
                n = min(len(st.session_state.fst), len(sf))
                dff = pd.DataFrame({
                    "SNP": sf["SNP"].values[:n], "CHR": sf["CHR"].values[:n],
                    "BP": sf["BP"].values[:n], "FST": st.session_state.fst[:n]})
                st.plotly_chart(plot_manhattan(dff, yc="FST", title="FST"),
                                use_container_width=True)
                st.dataframe(dff.sort_values("FST", ascending=False).head(30),
                             use_container_width=True)
                if st.session_state.fst_M is not None:
                    mdf = pd.DataFrame(st.session_state.fst_M,
                                       index=st.session_state.fst_pops,
                                       columns=st.session_state.fst_pops)
                    st.plotly_chart(px.imshow(mdf.round(4), text_auto=True,
                                              color_continuous_scale="Reds"),
                                    use_container_width=True)
            st.divider()
            st.subheader("🧬 Fenêtres FST")
            c1, c2, c3 = st.columns(3)
            ws = c1.slider("Fenêtre", 10, 200, 50)
            sp = c2.slider("Pas", 5, 100, 25)
            zt = c3.slider("Seuil z", 2.0, 5.0, 3.0, 0.1)
            if st.button("▶ Scan FST fenêtré", key="wf_btn"):
                w = windowed_fst(gu, su, lab, ws, sp, use_wc84=use_wc)
                st.session_state.win_fst = detect_outliers(w, zt)
                st.success(f"✅ {len(w)} fenêtres")
            if st.session_state.win_fst is not None:
                st.plotly_chart(
                    plot_win_manh(st.session_state.win_fst, "MEAN_FST",
                                  "FST fenêtré", sigma=zt),
                    use_container_width=True)
                st.dataframe(st.session_state.win_fst.head(30), use_container_width=True)
            st.divider()
            st.subheader("🧬 Tajima's D")
            race = st.selectbox("Population", sorted(np.unique(lab)), key="taj_race")
            wt = st.slider("Fenêtre Tajima", 10, 100, 25)
            if st.button(f"▶ Tajima — {race}", key="taj_btn"):
                mask = lab == race
                if mask.sum() >= 5:
                    st.session_state.taj_df = tajimas_scan(gu[mask], su, wt, wt // 2)
                    st.success(f"✅ {len(st.session_state.taj_df)} fenêtres")
            if st.session_state.taj_df is not None:
                st.plotly_chart(
                    plot_win_manh(st.session_state.taj_df, "TAJIMA_D", "Tajima D"),
                    use_container_width=True)
                st.dataframe(st.session_state.taj_df.sort_values("TAJIMA_D").head(30),
                             use_container_width=True)
            st.divider()
            st.subheader("🧬 Composite Selection Score")
            if st.button("▶ CSS", key="css_btn"):
                try:
                    st.session_state.css_df = composite_score(gu, su, lab)
                    st.success(f"✅ {len(st.session_state.css_df)} SNP")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.css_df is not None:
                cdf = st.session_state.css_df
                cp = cdf.copy(); cp["P_pseudo"] = np.exp(-cp["CSS"])
                st.plotly_chart(
                    plot_manhattan(cp.rename(columns={"P_pseudo": "P"}),
                                   yc="P", title="CSS"),
                    use_container_width=True)
                st.dataframe(cdf.head(30), use_container_width=True)

    # --- 8 GWAS ---
    with TABS[8]:
        pf = st.file_uploader("Phénotypes", type=["csv", "txt", "gz"], key="ph_up")
        c1, c2 = st.columns(2)
        with c1:
            if pf and st.button("📥 Charger", use_container_width=True):
                try:
                    mg, _ = match_pheno(st.session_state.ind_filt,
                                        load_pheno(pf.read(), pf.name))
                    st.session_state.pheno_df = mg
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
        with c2:
            if st.button("🎲 Simuler", use_container_width=True):
                n = len(st.session_state.ind_filt)
                rng = np.random.default_rng(42)
                st.session_state.pheno_df = pd.DataFrame({
                    "FID": st.session_state.ind_filt["FID"].values,
                    "IID": st.session_state.ind_filt["IID"].values,
                    "trait": rng.normal(0, 1, n)})
                st.success("✅")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols)
            is_bin = st.checkbox("Binaire")
            if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    cv = (st.session_state.pca_s[:, :5]
                          if st.session_state.pca_s is not None
                          else np.empty((gu.shape[0], 0)))
                    with st.spinner("GWAS LMM…"):
                        r = run_gwas(gu, ph.values, st.session_state.king_K, cv, is_bin=is_bin)
                        r = annotate_gwas(r, su)
                        r = postprocess_gwas(r)
                    st.session_state.gwas_res = r
                    st.session_state.lambda_gc = lambda_gc(r["P"].values)
                    st.success(f"✅ {len(r)} SNP · λGC = {st.session_state.lambda_gc:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.gwas_res is not None:
                r = st.session_state.gwas_res
                c1, c2, c3 = st.columns(3)
                c1.metric("SNP", len(r))
                c2.metric("FDR < 0.05", int((r["FDR"] < 0.05).sum()))
                c3.metric("λGC", f"{st.session_state.lambda_gc:.3f}")
                st.plotly_chart(plot_manhattan(r, yc="P", title="Manhattan"),
                                use_container_width=True)
                st.dataframe(r.head(30), use_container_width=True)
                st.download_button("⬇ GWAS", r.to_csv(index=False).encode(),
                                   "gwas.csv", "text/csv")

    # --- 9 POST-GWAS ---
    with TABS[9]:
        if st.session_state.gwas_res is not None:
            pip = fine_mapping(st.session_state.gwas_res)
            st.plotly_chart(plot_pip(pip), use_container_width=True)
            st.dataframe(pip[["SNP", "CHR", "BP", "P", "PIP"]].head(30),
                         use_container_width=True)
        else:
            st.warning("Lancez GWAS d'abord.")

    # --- 10 GBLUP ---
    with TABS[10]:
        if st.session_state.pheno_df is None:
            st.warning("Chargez phénotypes (GWAS).")
        else:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols, key="gb_pc")
            if st.button("▶ GBLUP", key="gb_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    with st.spinner("REML…"):
                        df, stats_ = run_gblup(gu, ph.values)
                    st.session_state.gebv_df = df
                    st.session_state.gblup_stats = stats_
                    st.success(f"✅ h² = {stats_['h2']:.3f} · r² = {stats_['r2']:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.gebv_df is not None:
                st.plotly_chart(plot_gblup(st.session_state.gebv_df),
                                use_container_width=True)
                st.dataframe(st.session_state.gebv_df.head(30), use_container_width=True)

    # --- 11 ANNOTATION ---
    with TABS[11]:
        gff_f = st.file_uploader("GFF3/GTF", type=["gff", "gff3", "gtf", "gz"], key="ann_gff")
        src = st.radio("Source", ["Sélection", "GWAS"], horizontal=True)
        wk = st.slider("Fenêtre kb", 1, 500, 50)
        if gff_f:
            try:
                with st.spinner("GFF…"):
                    gff = parse_gff3(gff_f.read())
                st.session_state.sel_gff = gff
                st.success(f"✅ {len(gff)} gènes")
                snps = None
                if src == "Sélection" and st.session_state.win_fst is not None:
                    snps = st.session_state.win_fst.head(20).rename(
                        columns={"START": "BP", "TOP_SNP": "SNP"})[["SNP", "CHR", "BP"]]
                elif src == "GWAS" and st.session_state.gwas_res is not None:
                    snps = st.session_state.gwas_res.sort_values("P").head(20)[["SNP", "CHR", "BP"]]
                if snps is not None and st.button("🔍 Annoter", key="ann_btn"):
                    st.session_state.ann_df = annotate_genes(snps, gff, wk)
                    st.success(f"✅ {len(st.session_state.ann_df)} associations")
                if st.session_state.ann_df is not None:
                    st.dataframe(st.session_state.ann_df, use_container_width=True)
                    st.download_button("⬇ Annotation",
                                       st.session_state.ann_df.to_csv(index=False).encode(),
                                       "annotation.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ {e}")

    # --- 12 VEP & NCBI ---
    with TABS[12]:
        snp_list = []
        if st.session_state.gwas_res is not None:
            snp_list = st.session_state.gwas_res.sort_values("P")["SNP"].head(20).tolist()
        elif st.session_state.win_fst is not None:
            snp_list = st.session_state.win_fst["TOP_SNP"].head(20).tolist()
        elif st.session_state.css_df is not None:
            snp_list = st.session_state.css_df["SNP"].head(20).tolist()
        if snp_list:
            st.info(f"{len(snp_list)} SNP à annoter via VEP")
            if st.button("🔬 Lancer VEP", key="vep_btn"):
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                inputs = []
                for s in snp_list:
                    row = su[su["SNP"].astype(str) == str(s)]
                    if not row.empty:
                        r = row.iloc[0]
                        a, b = _ca(r.get("A1")), _ca(r.get("A2"))
                        inputs.append(f"{r['CHR']}:{int(r['BP'])}:{a}:{b}"
                                      if a != "0" and b != "0"
                                      else f"{r['CHR']}:{int(r['BP'])}-{int(r['BP'])}")
                    else:
                        inputs.append(str(s))
                with st.spinner("VEP…"):
                    vres = run_vep_api(inputs, cfg["vep_species"])
                if not vres.empty:
                    vres["LOCALISATION"] = vres["CONSEQUENCE"].apply(traduire_consequence_vep)
                    st.session_state.vep_res = vres
                    st.success("✅")
        else:
            st.warning("Lancez GWAS, FST ou CSS d'abord.")
        if st.session_state.vep_res is not None:
            st.dataframe(st.session_state.vep_res, use_container_width=True)
            st.download_button("⬇ VEP",
                               st.session_state.vep_res.to_csv(index=False).encode(),
                               "vep.csv", "text/csv")
        st.divider()
        st.subheader("🏛️ NCBI Gene")
        genes = []
        if st.session_state.vep_res is not None:
            genes = [g for g in st.session_state.vep_res["GENE"].dropna().unique()
                     if g not in ("N/A", "Intergenic/Non-codant", "nan", "")]
        if genes and st.button("🏛️ NCBI", key="ncbi_btn"):
            with st.spinner("NCBI…"):
                st.session_state.ncbi_gene_res = fetch_ncbi_gene_info(genes[:20], cfg["taxon"])
            st.success("✅")
        if st.session_state.ncbi_gene_res is not None:
            st.dataframe(st.session_state.ncbi_gene_res, use_container_width=True)

    # --- 13 STRUCTURE 3D ---
    with TABS[13]:
        genes = []
        for key in ("vep_res", "ann_df", "ncbi_gene_res"):
            v = st.session_state.get(key)
            if v is not None and hasattr(v, "columns") and "GENE" in v.columns and not v.empty:
                genes = [g for g in v["GENE"].unique()
                         if str(g) not in ("N/A", "Intergenic/Non-codant", "nan", "")]
                if genes: break
        if not genes:
            st.warning("Lancez VEP ou Annotation d'abord.")
        else:
            g = st.selectbox("Gène", genes, key="af_gene")
            style = st.selectbox("Style", ["cartoon", "stick", "sphere"], key="af_style")
            if st.button("▶ Charger 3D", key="af_btn"):
                with st.spinner(f"AlphaFold {g}…"):
                    uid = fetch_uniprot_id(g, cfg["taxon"])
                    if uid:
                        pdb = fetch_alphafold_pdb(uid)
                        if pdb: render_3d_structure(pdb, style=style)
                        else: st.error("Structure introuvable.")
                    else: st.error("Gène non trouvé.")

    # --- 14 ALIGNEMENT ---
    with TABS[14]:
        src_align = (st.session_state.pair_res
                     or st.session_state.gwas_res
                     or st.session_state.css_df)
        if src_align is not None and not src_align.empty:
            s = st.selectbox("SNP", src_align["SNP"].tolist(), key="aln_snp")
            if st.button("🔍 Aligner", key="aln_btn"):
                row = src_align[src_align["SNP"] == s].iloc[0]
                chrom, bp = str(row["CHR"]).replace("chr", ""), int(row["BP"])
                start, end = max(1, bp - 25), bp + 25
                with st.spinner(f"CHR{chrom}:{start}-{end}…"):
                    seq = fetch_ensembl_sequence(chrom, start, end, cfg["vep_species"])
                if seq:
                    pos = bp - start
                    f = list(seq); f[pos] = "A" if seq[pos] != "A" else "G"
                    st.markdown(align_and_highlight(seq, "".join(f), seq),
                                unsafe_allow_html=True)
                else: st.error("Séquence non trouvée.")
        else:
            st.warning("Lancez GWAS ou CSS d'abord.")

    # --- 15 MÉTA-ANALYSE ---
    with TABS[15]:
        st.subheader("🔬 Enrichissement fonctionnel")
        genes = []
        if st.session_state.vep_res is not None:
            genes = [g for g in st.session_state.vep_res["GENE"].dropna().unique()
                     if g not in ("N/A", "Intergenic/Non-codant", "nan", "")]
        if genes and st.button("🔬 Lancer enrichissement", key="enr_btn"):
            with st.spinner("Enrichr…"):
                enr = run_enrichment(genes, organism="Human")
            if not enr.empty:
                st.dataframe(enr.head(30), use_container_width=True)
                st.download_button("⬇ Enrichissement",
                                   enr.to_csv(index=False).encode(),
                                   "enrichment.csv", "text/csv")
        else:
            st.info("Lancez VEP d'abord.")
        st.divider()
        st.subheader("🕸️ STRING Network")
        if genes and st.button("🕸️ Réseau STRING", key="string_btn"):
            edges, nodes = query_string_network(genes[:30], cfg["taxon"], 400)
            if not edges.empty:
                fig = plot_string_network(edges, nodes)
                if fig: st.plotly_chart(fig, use_container_width=True)
                st.dataframe(edges, use_container_width=True)
            else:
                st.warning("Aucune interaction.")

    # --- 16 ML ---
    with TABS[16]:
        if st.session_state.pheno_df is None:
            st.warning("Chargez phénotypes (GWAS).")
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols, key="ml_pc")
            if st.button("🚀 Random Forest", key="ml_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    valid = np.isfinite(ph.values)
                    if valid.sum() < 20:
                        st.error("❌ Pas assez d'individus.")
                    else:
                        X = impute_mean(gu[valid]); y = ph.values[valid]
                        with st.spinner("RF…"):
                            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(Xtr, ytr)
                            r2 = r2_score(yte, rf.predict(Xte))
                            imp = rf.feature_importances_
                        st.success(f"✅ R² = {r2:.3f}")
                        imp_df = pd.DataFrame({"SNP": su["SNP"].values,
                                               "Importance": imp}).sort_values("Importance", ascending=False).head(20)
                        st.plotly_chart(px.bar(imp_df, x="Importance", y="SNP", orientation="h"),
                                        use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")

    # --- 17 HAPLOTYPES ---
    with TABS[17]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            iu = st.session_state.ind_filt
            chroms = sorted(su["CHR"].astype(str).unique())
            sel_chr = st.selectbox("Chromosome", chroms, key="hap_chr")
            sub_idx = su[su["CHR"].astype(str) == sel_chr].index
            if len(sub_idx) > 0:
                if len(sub_idx) > 50: sub_idx = sub_idx[:50]
                snps_names = su.loc[sub_idx, "SNP"].values
                gt_sub = gu[:, sub_idx]
                groups = iu["FID"].astype(str).values
                sort_idx = np.argsort(groups)
                fig = px.imshow(gt_sub[sort_idx],
                                labels=dict(x="SNP", y="Individu", color="Génotype"),
                                x=snps_names, y=groups[sort_idx],
                                color_continuous_scale=["#F0F0F0", "#3498db", "#e74c3c"],
                                title=f"Haplotypes CHR{sel_chr}", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

    # --- 18 LIFTOVER ---
    with TABS[18]:
        if st.session_state.snp_filt is not None:
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            s = st.selectbox("SNP", su["SNP"].tolist(), key="lift_snp")
            if st.button("🔄 Convertir", key="lift_btn"):
                row = su[su["SNP"] == s].iloc[0]
                chrom, bp = str(row["CHR"]), int(row["BP"])
                with st.spinner(f"LiftOver {chrom}:{bp}…"):
                    hc, hb = lift_over(chrom, bp)
                if hc: st.success(f"✅ Homme : chr{hc}:{hb}")
                else: st.error("Aucune correspondance.")

    # --- 19 EXPORT ---
    with TABS[19]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📦 PLINK", use_container_width=True):
                    try:
                        st.download_button("⬇ PLINK",
                                           build_plink(gu, st.session_state.ind_filt, su, cfg["prefix"]),
                                           f"{cfg['prefix']}.zip", "application/zip")
                    except Exception as e:
                        st.error(f"❌ {e}")
            with c2:
                if st.button("📄 VCF", use_container_width=True):
                    try:
                        st.download_button("⬇ VCF",
                                           build_vcf(gu, st.session_state.ind_filt, su, cfg["prefix"]).encode(),
                                           f"{cfg['prefix']}.vcf", "text/plain")
                    except Exception as e:
                        st.error(f"❌ {e}")
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Individus", gu.shape[0])
            c2.metric("SNP", gu.shape[1])
            c3.metric("Mode", current_mode.split("(")[0].strip())

    # --- 20 DIVERSITÉ ---
    with TABS[20]:
        st.subheader(f"📊 Métriques de Diversité — {current_mode}")
        if st.session_state.gt_filt is not None:
            if st.button("🧮 Calculer les métriques"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    labels = st.session_state.ind_filt["FID"].astype(str).values
                    df_div = compute_diversity_metrics(gu, labels)
                    if st.session_state.froh is not None:
                        df_froh = pd.DataFrame({"Population": labels, "FROH": st.session_state.froh}) \
                                  .groupby("Population")["FROH"].mean().reset_index()
                        df_div = df_div.merge(df_froh, on="Population", how="left")
                        df_div["FROH"] = df_div["FROH"].round(4)
                    st.dataframe(df_div, use_container_width=True)
                    st.download_button("⬇ Télécharger", df_div.to_csv(index=False).encode(),
                                       "diversity_metrics.csv", "text/csv")
                except Exception as e:
                    st.error(f"❌ {e}")

if __name__ == "__main__":
    main()
