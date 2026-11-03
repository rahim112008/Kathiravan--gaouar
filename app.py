"""app.py — NW Camelids Genomics Platform (Excel Direct)"""
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
    "mapping_df", "pair_res", "current_species",
]
for _k in _KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None

def reset_derived():
    for k in _KEYS:
        if k not in ("gt", "ind_df", "snp_df", "current_species"):
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

        if st.session_state.current_species is None:
            st.session_state.current_species = species_key
        elif st.session_state.current_species != species_key:
            st.session_state.current_species = species_key
            reset_derived()
            st.session_state.gt = st.session_state.ind_df = st.session_state.snp_df = None
            st.warning("🔄 Mode changé — réinitialisé.")

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
                    st.session_state.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": snp})
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            vf = st.file_uploader(".vcf", type=["vcf", "gz"], key="vcf_up")
            if vf and st.button("📥 Charger VCF", use_container_width=True):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    reset_derived()
                    st.session_state.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": snp})
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")

        # --- METADONNEES DIRECTEMENT DEPUIS L'EXCEL ---
        st.divider()
        st.header("📋 2.5 Métadonnées")
        st.caption("Uploadez directement le fichier Excel de Kathiravan.")
        meta_file = st.file_uploader("NWC_Meta_Sample_Details_Suheil.xlsx", type=["xlsx", "csv"], key="meta_up")
        
        if meta_file and st.session_state.ind_df is not None and st.button("🔗 Lier Métadonnées", use_container_width=True):
            try:
                if meta_file.name.endswith(".xlsx"):
                    # Lecture directe de l'Excel sur Streamlit
                    df_excel = pd.read_excel(meta_file, sheet_name="Sample_Details", header=0)
                    # Colonne 8 (I) = Nom, Colonne 17 (R) = Breed
                    meta = df_excel.iloc[:, [8, 17]].copy()
                    meta.columns = ["Sample_Filename", "Breed"]
                    meta = meta.dropna(subset=["Sample_Filename"])
                    meta = meta[meta["Sample_Filename"].astype(str).str.contains(".CEL", na=False)]

                    def map_breed_to_species(breed):
                        b = str(breed).upper()
                        if "HUACAYA" in b or "SURI" in b or "ALPACA" in b: return "Alpaga"
                        if "GUANACO" in b: return "Guanaco"
                        if "LLAMA" in b: return "Lama"
                        if "VICUGNA" in b or "V.MENSALIS" in b or "V.SP" in b: return "Vigogne"
                        return "Autre"

                    meta["Species"] = meta["Breed"].apply(map_breed_to_species)
                    meta_df = meta[["Sample_Filename", "Species"]].copy()
                else:
                    # Lecture CSV classique
                    meta_df = pd.read_csv(meta_file)

                # Fusion
                st.session_state.ind_df = st.session_state.ind_df.merge(
                    meta_df[["Sample_Filename", "Species"]],
                    left_on="IID", right_on="Sample_Filename", how="left"
                )
                st.session_state.ind_df["FID"] = st.session_state.ind_df["Species"].fillna(st.session_state.ind_df["FID"])
                
                st.success("✅ Métadonnées liées !")
                st.write("Répartition :")
                st.write(st.session_state.ind_df["FID"].value_counts())
                st.dataframe(st.session_state.ind_df[["IID", "FID"]].head(5))
                
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

    st.title(f"🦙 {species_key}")
    if st.session_state.gt is None:
        st.info("👋 Chargez `.map`, `.ped` et le fichier Excel des métadonnées.")
        return

    TABS = st.tabs([
        "🏠 Accueil", "🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture",
        "📈 Démographie", "🌳 Phylogénie", "🔍 Sélection",
        "🧪 GWAS", "📊 Diversité"
    ])

    with TABS[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Individus",   st.session_state.gt.shape[0])
        c2.metric("SNP",         st.session_state.gt.shape[1])
        c3.metric("Chromosomes", st.session_state.snp_df["CHR"].nunique())
        c4.metric("Groupes",     st.session_state.ind_df["FID"].nunique())
        ch = st.session_state.snp_df["CHR"].astype(str).value_counts().sort_index()
        st.plotly_chart(px.bar(x=ch.index, y=ch.values, labels={"x": "Chromosome", "y": "Nb SNP"}), use_container_width=True)

    with TABS[1]:
        if st.button("▶ Lancer QC", type="primary"):
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

    with TABS[2]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ LD Pruning"):
                keep = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.update({
                    "gt_pruned": st.session_state.gt_filt[:, keep],
                    "snp_pruned": st.session_state.snp_filt[keep].reset_index(drop=True)})
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNP")

    with TABS[3]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS"):
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

    with TABS[4]:
        if st.session_state.gt_filt is not None:
            K = st.slider("K choisi", 2, 10, 4)
            if st.button("▶ Lancer NMF"):
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

    with TABS[5]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ ROH + Ne"):
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

    with TABS[6]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ UPGMA"):
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

    with TABS[7]:
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            lab = st.session_state.ind_filt["FID"].astype(str).values
            if st.button("▶ FST Pairwise"):
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

    with TABS[8]:
        pf = st.file_uploader("Phénotypes", type=["csv", "txt", "gz"])
        if pf and st.button("📥 Charger"):
            try:
                mg, _ = match_pheno(st.session_state.ind_filt, load_pheno(pf.read(), pf.name))
                st.session_state.pheno_df = mg
                st.success("✅")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID")]
            pc = st.selectbox("Phénotype", cols)
            if st.button("🚀 Lancer GWAS"):
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

    with TABS[9]:
        if st.session_state.gt_filt is not None:
            if st.button("🧮 Calculer les métriques"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    labels = st.session_state.ind_filt["FID"].astype(str).values
                    df_div = compute_diversity_metrics(gu, labels)
                    if st.session_state.froh is not None:
                        df_froh = pd.DataFrame({"Population": labels, "FROH": st.session_state.froh}).groupby("Population")["FROH"].mean().reset_index()
                        df_div = df_div.merge(df_froh, on="Population", how="left")
                        df_div["FROH"] = df_div["FROH"].round(4)
                    st.dataframe(df_div, use_container_width=True)
                    st.download_button("⬇ Télécharger", df_div.to_csv(index=False).encode(), "diversity_metrics.csv", "text/csv")
                except Exception as e:
                    st.error(f"❌ {e}")

if __name__ == "__main__":
    main()
