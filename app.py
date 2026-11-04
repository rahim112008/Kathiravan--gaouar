"""app.py — NW Camelids Genomics Platform (Version Finale Sans Bug)"""
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
            st.info("🌍 Mode Global")
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

        st.divider()
        st.header("🎯 Mode de regroupement")
        grouping_mode = st.radio(
            "Regrouper les analyses par :",
            ["🌍 Par Pays (ARG, BOL, PER, USA)", "🦙 Par Espèce (Alpaga, Guanaco, ...)"],
            index=0,
        )
        st.session_state.grouping_mode = grouping_mode

        if grouping_mode.startswith("🌍") and st.session_state.get("ind_df_pays") is not None:
            st.session_state.ind_df = st.session_state.ind_df_pays.copy()
        elif grouping_mode.startswith("🦙") and st.session_state.get("ind_df_species") is not None:
            st.session_state.ind_df = st.session_state.ind_df_species.copy()

        st.divider()
        st.header("📁 1. Annotations SNP")
        f_map = st.file_uploader(".map", type=["map", "gz"], key="map_up")
        if f_map and st.button("📥 Charger .map", use_container_width=True):
            try:
                snp_df = load_map_uploaded(f_map.read(), f_map.name)
                # CORRECTION : Forcer le chromosome à être du texte pour éviter le "1.0" et l'axe "0"
                snp_df['CHR'] = snp_df['CHR'].astype(str).str.replace(r'\.0$', '', regex=True)
                st.session_state.snp_df = snp_df
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
        # METADONNEES + EXTRACTION PAR CODE (CORRIGÉ)
        # ============================================================
        st.divider()
        st.header("📋 2.5 Métadonnées (Espèces)")
        meta_file = st.file_uploader("NWC_Meta_Sample_Details_Suheil.xlsx", type=["xlsx", "csv"], key="meta_up")
        
        if meta_file and st.session_state.ind_df_pays is not None:
            try:
                if meta_file.name.endswith(".xlsx"):
                    df_excel = pd.read_excel(meta_file)
                else:
                    df_excel = pd.read_csv(meta_file)
                
                # On récupère la 1ère colonne qui contient les noms (ex: ARG_ALG_ARG01_A1...)
                id_col = df_excel.columns[0]
                df_excel['ID_Clean'] = df_excel[id_col].astype(str).str.strip().str.upper().str.replace('.CEL', '', regex=False)
                
                # On extrait le code pays (ARG) et le code espèce (ALG)
                df_excel['Code_Pays'] = df_excel['ID_Clean'].str.split('_').str[0]
                df_excel['Code_Espece'] = df_excel['ID_Clean'].str.split('_').str[1]
                
                # Dictionnaires de traduction
                dict_especes = {
                    'ALG': 'Alpaga', 'PAH': 'Alpaga', 'BAH': 'Alpaga', 'PAS': 'Alpaga', 'BAS': 'Alpaga',
                    'BLL': 'Lama', 'ALL': 'Lama', 'ULP': 'Lama',
                    'BVM': 'Vigogne', 'AVS': 'Vigogne', 'UVV': 'Vigogne', 'BVV': 'Vigogne', 'BVS': 'Vigogne',
                    'CLG': 'Guanaco'
                }
                dict_pays = {'ARG': 'Argentine', 'BOL': 'Bolivie', 'PER': 'Pérou', 'USA': 'USA', 'CHL': 'Chili'}
                
                df_excel['Espece_Propre'] = df_excel['Code_Espece'].map(dict_especes).fillna('Autre')
                df_excel['Pays_Propre'] = df_excel['Code_Pays'].map(dict_pays).fillna('Autre')
                
                meta = df_excel[['ID_Clean', 'Espece_Propre', 'Pays_Propre']].copy()
                meta = meta.drop_duplicates(subset=['ID_Clean'])

                # --- DIAGNOSTIC ---
                with st.expander("🔍 Voir le diagnostic des IDs", expanded=False):
                    iid_ped = st.session_state.ind_df_pays["IID"].astype(str).str.replace(".CEL", "", regex=False).str.strip().str.upper()
                    iid_excel = meta["ID_Clean"].astype(str)
                    match_exact = len(set(iid_ped).intersection(set(iid_excel)))
                    st.write(f"**Correspondances exactes : {match_exact} / {len(iid_ped)}**")

                if st.button("🔗 Lier Métadonnées", use_container_width=True):
                    ind_pays = st.session_state.ind_df_pays.copy()
                    ind_pays["IID_clean"] = ind_pays["IID"].astype(str).str.replace(".CEL", "", regex=False).str.strip().str.upper()
                    
                    # how='left' garantit qu'on garde EXACTEMENT 374 animaux
                    ind_merged = ind_pays.merge(meta[['ID_Clean', 'Espece_Propre', 'Pays_Propre']], left_on="IID_clean", right_on="ID_Clean", how="left")

                    ind_merged["Espece_Propre"] = ind_merged["Espece_Propre"].fillna("Autre")
                    ind_merged["Pays_Propre"] = ind_merged["Pays_Propre"].fillna("Autre")

                    # Mise à jour du groupe (FID) selon l'espèce
                    ind_species = ind_merged.copy()
                    ind_species["FID"] = ind_species["Espece_Propre"]
                    
                    # Mise à jour du groupe (FID) selon le pays
                    ind_pays_out = ind_merged.copy()
                    ind_pays_out["FID"] = ind_pays_out["Pays_Propre"]

                    st.session_state.ind_df_species = ind_species
                    st.session_state.ind_df_pays = ind_pays_out
                    
                    if grouping_mode.startswith("🦙"):
                        st.session_state.ind_df = ind_species.copy()
                    else:
                        st.session_state.ind_df = ind_pays_out.copy()

                    st.success("✅ Métadonnées liées !")
                    st.write("**Répartition par Espèce :**")
                    st.write(ind_species["Espece_Propre"].value_counts())
                    st.write("**Répartition par Pays :**")
                    st.write(ind_pays_out["Pays_Propre"].value_counts())
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
        st.info("👋 Chargez `.map` et `.ped` pour commencer.")
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
        
        # CORRECTION : Affichage propre du graphique des chromosomes
        ch = st.session_state.snp_df["CHR"].astype(str).value_counts().sort_index()
        fig_ch = px.bar(x=ch.index, y=ch.values, labels={"x": "Chromosome", "y": "Nb SNP"}, title="Nombre de SNP par chromosome")
        fig_ch.update_xaxes(type='category') # Force l'axe X à être catégoriel (1, 2, 3...)
        st.plotly_chart(fig_ch, use_container_width=True)
        
        st.subheader(f"📋 Répartition par {current_mode}")
        st.dataframe(st.session_state.ind_df["FID"].value_counts().reset_index(), use_container_width=True)

    # --- 1 QC ---
    with TABS[1]:
        if st.button("▶ Lancer QC", type="primary", key="qc_btn"):
            try:
                params = {"geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t, "het_sd": het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, params, cfg["autosomes"], keep_autosomes=keep_auto, rem_out=rem_out)
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

    # --- 2 LD ---
    with TABS[2]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ LD Pruning", key="ld_btn"):
                keep = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.update({"gt_pruned": st.session_state.gt_filt[:, keep], "snp_pruned": st.session_state.snp_filt[keep].reset_index(drop=True)})
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNP")
            if st.session_state.gt_pruned is not None:
                if st.button("▶ LD Decay", key="ld_decay_btn"):
                    st.session_state.ld_df = ld_decay(st.session_state.gt_pruned, st.session_state.snp_pruned)
                if st.session_state.ld_df is not None:
                    st.plotly_chart(plot_ld(st.session_state.ld_df), use_container_width=True)

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
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v, st.session_state.ind_filt["FID"].values), use_container_width=True)
            st.plotly_chart(plot_mds(st.session_state.mds_c, st.session_state.ind_filt["FID"].values), use_container_width=True)

    # --- 4 ADMIXTURE ---
    with TABS[4]:
        if st.session_state.gt_filt is not None:
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
                f, _ = plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values, st.session_state.ind_filt["FID"].values, st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True)

    # --- 5 DÉMOGRAPHIE ---
    with TABS[5]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ ROH + Ne", key="dem_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    roh, fr = detect_roh(gu, su[["CHR", "BP"]].to_json())
                    st.session_state.update({"roh_df": roh, "froh": fr})
                    st.success(f"✅ {len(roh)} ROH")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.froh is not None:
                df_f = pd.DataFrame({"FROH": st.session_state.froh, "Pop": st.session_state.ind_filt["FID"].values})
                st.plotly_chart(px.histogram(df_f, x="FROH", color="Pop", nbins=50), use_container_width=True)

    # --- 6 PHYLOGÉNIE ---
    with TABS[6]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ UPGMA", key="phylo_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    lab = st.session_state.ind_filt["FID"].astype(str).values
                    r = upgma_bootstrap(gu, su, lab, n_boot=100)
                    st.session_state.nj_result = r
                    st.session_state.nj_mid = midpoint_info(r["D"], r["pops"])
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.nj_result is not None:
                r = st.session_state.nj_result
                st.plotly_chart(plot_tree(r["Z"], r["pops"], r["support"], title="UPGMA", mid=st.session_state.nj_mid), use_container_width=True)

    # --- 7 SÉLECTION ---
    with TABS[7]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            lab = st.session_state.ind_filt["FID"].astype(str).values
            if st.button("▶ FST Pairwise", key="fst_btn"):
                try:
                    st.session_state.fst = fst_snp_wc(gu, lab)
                    M, pops = fst_pairwise(gu, lab, use_wc84=True)
                    st.session_state.update({"fst_M": M, "fst_pops": pops})
                    st.success("✅")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.fst_M is not None:
                mdf = pd.DataFrame(st.session_state.fst_M, index=st.session_state.fst_pops, columns=st.session_state.fst_pops)
                st.plotly_chart(px.imshow(mdf.round(4), text_auto=True, color_continuous_scale="Reds"), use_container_width=True)
            st.divider()
            st.subheader("FST Fenêtré")
            if st.button("▶ Scan FST fenêtré", key="wf_btn"):
                w = windowed_fst(gu, su, lab, 50, 25)
                st.session_state.win_fst = detect_outliers(w, 3.0)
                st.success(f"✅ {len(w)} fenêtres")
            if st.session_state.win_fst is not None:
                st.plotly_chart(plot_win_manh(st.session_state.win_fst, "MEAN_FST", "FST fenêtré"), use_container_width=True)
            st.divider()
            st.subheader("CSS")
            if st.button("▶ CSS", key="css_btn"):
                st.session_state.css_df = composite_score(gu, su, lab)
                st.success("✅")
            if st.session_state.css_df is not None:
                st.dataframe(st.session_state.css_df.head(20), use_container_width=True)

    # --- 8 GWAS ---
    with TABS[8]:
        pf = st.file_uploader("Phénotypes", type=["csv", "txt", "gz"], key="ph_up")
        if pf and st.button("📥 Charger", key="ph_btn"):
            try:
                mg, _ = match_pheno(st.session_state.ind_filt, load_pheno(pf.read(), pf.name))
                st.session_state.pheno_df = mg
                st.success("✅")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols)
            if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    cv = (st.session_state.pca_s[:, :5] if st.session_state.pca_s is not None else np.empty((gu.shape[0], 0)))
                    r = run_gwas(gu, ph.values, st.session_state.king_K, cv)
                    r = annotate_gwas(r, su); r = postprocess_gwas(r)
                    st.session_state.gwas_res = r
                    st.session_state.lambda_gc = lambda_gc(r["P"].values)
                    st.success(f"✅ {len(r)} SNP · λGC = {st.session_state.lambda_gc:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.gwas_res is not None:
                st.plotly_chart(plot_manhattan(st.session_state.gwas_res, yc="P", title="Manhattan"), use_container_width=True)

    # --- 9 POST-GWAS ---
    with TABS[9]:
        if st.session_state.gwas_res is not None:
            pip = fine_mapping(st.session_state.gwas_res)
            st.plotly_chart(plot_pip(pip), use_container_width=True)
            st.dataframe(pip[["SNP", "CHR", "BP", "P", "PIP"]].head(30), use_container_width=True)

    # --- 10 GBLUP ---
    with TABS[10]:
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols, key="gb_pc")
            if st.button("▶ GBLUP", key="gb_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    df, stats_ = run_gblup(gu, ph.values)
                    st.session_state.gebv_df = df
                    st.session_state.gblup_stats = stats_
                    st.success(f"✅ h² = {stats_['h2']:.3f} · r² = {stats_['r2']:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")

    # --- 11 ANNOTATION ---
    with TABS[11]:
        gff_f = st.file_uploader("GFF3/GTF", type=["gff", "gff3", "gtf", "gz"], key="ann_gff")
        src = st.radio("Source", ["Sélection", "GWAS"], horizontal=True)
        wk = st.slider("Fenêtre kb", 1, 500, 50)
        if gff_f:
            try:
                gff = parse_gff3(gff_f.read())
                st.session_state.sel_gff = gff
                st.success(f"✅ {len(gff)} gènes")
                snps = None
                if src == "Sélection" and st.session_state.win_fst is not None:
                    snps = st.session_state.win_fst.head(20).rename(columns={"START": "BP", "TOP_SNP": "SNP"})[["SNP", "CHR", "BP"]]
                elif src == "GWAS" and st.session_state.gwas_res is not None:
                    snps = st.session_state.gwas_res.sort_values("P").head(20)[["SNP", "CHR", "BP"]]
                if snps is not None and st.button("🔍 Annoter", key="ann_btn"):
                    st.session_state.ann_df = annotate_genes(snps, gff, wk)
                    st.success(f"✅ {len(st.session_state.ann_df)} associations")
                if st.session_state.ann_df is not None:
                    st.dataframe(st.session_state.ann_df, use_container_width=True)
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
        if snp_list and st.button("🔬 Lancer VEP", key="vep_btn"):
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            inputs = []
            for s in snp_list:
                row = su[su["SNP"].astype(str) == str(s)]
                if not row.empty:
                    r = row.iloc[0]; a, b = _ca(r.get("A1")), _ca(r.get("A2"))
                    inputs.append(f"{r['CHR']}:{int(r['BP'])}:{a}:{b}" if a != "0" and b != "0" else f"{r['CHR']}:{int(r['BP'])}-{int(r['BP'])}")
                else:
                    inputs.append(str(s))
            vres = run_vep_api(inputs, cfg["vep_species"])
            if not vres.empty:
                vres["LOCALISATION"] = vres["CONSEQUENCE"].apply(traduire_consequence_vep)
                st.session_state.vep_res = vres
                st.success("✅")
        if st.session_state.vep_res is not None:
            st.dataframe(st.session_state.vep_res, use_container_width=True)

    # --- 13 STRUCTURE 3D ---
    with TABS[13]:
        genes = []
        if st.session_state.vep_res is not None:
            genes = [g for g in st.session_state.vep_res["GENE"].dropna().unique() if g not in ("N/A", "Intergenic/Non-codant", "nan", "")]
        if genes:
            g = st.selectbox("Gène", genes, key="af_gene")
            if st.button("▶ Charger 3D", key="af_btn"):
                uid = fetch_uniprot_id(g, cfg["taxon"])
                if uid:
                    pdb = fetch_alphafold_pdb(uid)
                    if pdb: render_3d_structure(pdb, style="cartoon")
                    else: st.error("Structure introuvable.")
                else: st.error("Gène non trouvé.")

    # --- 14 ALIGNEMENT ---
    with TABS[14]:
        st.info("Alignement disponible via les résultats GWAS/Sélection.")

    # --- 15 MÉTA-ANALYSE ---
    with TABS[15]:
        genes = []
        if st.session_state.vep_res is not None:
            genes = [g for g in st.session_state.vep_res["GENE"].dropna().unique() if g not in ("N/A", "Intergenic/Non-codant", "nan", "")]
        if genes and st.button("🔬 Lancer enrichissement", key="enr_btn"):
            enr = run_enrichment(genes, organism="Human")
            if not enr.empty:
                st.dataframe(enr.head(30), use_container_width=True)

    # --- 16 ML ---
    with TABS[16]:
        if st.session_state.pheno_df is not None:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols, key="ml_pc")
            if st.button("🚀 Random Forest", key="ml_btn"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                valid = np.isfinite(ph.values)
                if valid.sum() >= 20:
                    X = impute_mean(gu[valid]); y = ph.values[valid]
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(Xtr, ytr)
                    r2 = r2_score(yte, rf.predict(Xte))
                    st.success(f"✅ R² = {r2:.3f}")

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
                fig = px.imshow(gt_sub[sort_idx], labels=dict(x="SNP", y="Individu", color="Génotype"), x=snps_names, y=groups[sort_idx], color_continuous_scale=["#F0F0F0", "#3498db", "#e74c3c"], title=f"Haplotypes CHR{sel_chr}", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

    # --- 18 LIFTOVER ---
    with TABS[18]:
        if st.session_state.snp_filt is not None:
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            s = st.selectbox("SNP", su["SNP"].tolist(), key="lift_snp")
            if st.button("🔄 Convertir", key="lift_btn"):
                row = su[su["SNP"] == s].iloc[0]
                hc, hb = lift_over(str(row["CHR"]), int(row["BP"]))
                if hc: st.success(f"✅ Homme : chr{hc}:{hb}")
                else: st.error("Aucune correspondance.")

    # --- 19 EXPORT ---
    with TABS[19]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            if st.button("📦 PLINK", use_container_width=True):
                st.download_button("⬇ PLINK", build_plink(gu, st.session_state.ind_filt, su, cfg["prefix"]), f"{cfg['prefix']}.zip", "application/zip")
            if st.button("📄 VCF", use_container_width=True):
                st.download_button("⬇ VCF", build_vcf(gu, st.session_state.ind_filt, su, cfg["prefix"]).encode(), f"{cfg['prefix']}.vcf", "text/plain")

    # --- 20 DIVERSITÉ ---
    with TABS[20]:
        st.subheader(f"📊 Métriques de Diversité — {current_mode}")
        if st.session_state.gt_filt is not None:
            if st.button("🧮 Calculer les métriques"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                labels = st.session_state.ind_filt["FID"].astype(str).values
                df_div = compute_diversity_metrics(gu, labels)
                if st.session_state.froh is not None:
                    df_froh = pd.DataFrame({"Population": labels, "FROH": st.session_state.froh}).groupby("Population")["FROH"].mean().reset_index()
                    df_div = df_div.merge(df_froh, on="Population", how="left")
                    df_div["FROH"] = df_div["FROH"].round(4)
                st.dataframe(df_div, use_container_width=True)
                st.download_button("⬇ Télécharger", df_div.to_csv(index=False).encode(), "diversity_metrics.csv", "text/csv")

if __name__ == "__main__":
    main()
