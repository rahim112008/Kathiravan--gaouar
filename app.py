"""app.py — NW Camelids Genomics Platform (Version Finale Fluide — CORRIGÉE)

Corrections majeures (2026-11) :
  1. Autosomes : 36 paires pour les camélidés (2n=74 = 36 paires autosomes + XX/XY).
     -> range(1, 37) et non range(1, 39). X/Y/MT exclus au QC.
  2. Comparaison 2 à 2 : formule |p1 - p2| >= seuil (au lieu de p1>=s & p2<=1-s).
     Défaut du curseur à 0.80 pour reproduire le Tableau 3 du rapport.
  3. Alignement : la séquence de référence est réinitialisée si l'utilisateur
     change de SNP, pour éviter d'aligner la mauvaise séquence.
  4. Annotation : avertissement explicite quand la source retourne 0 SNP.
  5. GWAS : dégradation gracieuse si la matrice KING n'est pas calculée.
  6. Comparaison 2 à 2 : interdiction de comparer un groupe avec lui-même.
"""
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

# Nombre d'autosomes commun à TOUS les camélidés (2n = 74 = 36 paires + XX/XY).
# Référence : rapport p.1 « 38 chromosomes » = 36 autosomes + X + Y (+ MT séparé).
CAMELID_AUTOSOMES = [str(i) for i in range(1, 37)]  # 1..36

# ============================================================
# Fonctions utilitaires Impact Moléculaire
# ============================================================
def fetch_cds_sequence_from_ncbi(chrom, start, end):
    """Récupère la séquence d'ADN d'une région via l'API NCBI."""
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           f"db=nuccore&id={chrom}&rettype=fasta&seq_start={start}"
           f"&seq_stop={end}&retmode=text")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            sequence = "".join([l for l in lines if not l.startswith(">")])
            return sequence.upper()
    except Exception:
        pass
    return None

def translate_codon(codon):
    """Traduit un codon en acide aminé (code génétique standard)."""
    table = {
        'ATA':'I','ATC':'I','ATT':'I','ATG':'M','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
        'AAC':'N','AAT':'N','AAA':'K','AAG':'K','AGC':'S','AGT':'S','AGA':'R','AGG':'R',
        'CTA':'L','CTC':'L','CTG':'L','CTT':'L','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
        'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q','CGA':'R','CGC':'R','CGG':'R','CGT':'R',
        'GTA':'V','GTC':'V','GTG':'V','GTT':'V','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
        'GAC':'D','GAT':'D','GAA':'E','GAG':'E','GGA':'G','GGC':'G','GGG':'G','GGT':'G',
        'TCA':'S','TCC':'S','TCG':'S','TCT':'S','TTC':'F','TTT':'F','TTA':'L','TTG':'L',
        'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*','TGC':'C','TGT':'C','TGA':'*','TGG':'W'
    }
    return table.get(codon, 'X')

# ============================================================
# Session state
# ============================================================
_KEYS = [
    "gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt",
    "gt_pruned","snp_pruned","gt_pre","ind_pre","snp_pre","qc_stats",
    "pca_s","pca_v","mds_c","king_K",
    "fst","fst_M","fst_pops","fst_snp_df",
    "admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict",
    "reyn_D","reyn_pops","nj_result","nj_mid",
    "win_fst","taj_df","css_df",
    "pheno_df","gwas_res","lambda_gc",
    "gebv_df","gblup_stats",
    "vep_res","ncbi_gene_res","sel_ann","sel_gff","ann_df",
    "mapping_df","pair_res","current_species","grouping_mode",
    "snps_specifiques",
    "gt_original","ind_df_original","snp_df_original",
    "last_species_filter","last_grouping_filter",
    "align_ref_seq","align_ref_snp",
]
for _k in _KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None

def reset_derived():
    for k in _KEYS:
        if k not in ("gt","ind_df","snp_df","current_species",
                     "grouping_mode","snps_specifiques",
                     "gt_original","ind_df_original","snp_df_original",
                     "last_species_filter","last_grouping_filter"):
            st.session_state[k] = None

def update_data_filters(species_key, grouping_mode):
    if st.session_state.gt_original is None or st.session_state.ind_df_original is None:
        return

    gt_orig = st.session_state.gt_original
    ind_orig = st.session_state.ind_df_original

    # 1. Filtrage par Espèce
    if species_key != "🌍 Toutes les espèces (Global)":
        target = None
        if "Vicugna pacos" in species_key:   target = "Alpaca"
        elif "Lama glama" in species_key:    target = "Llama"
        elif "Lama guanicoe" in species_key: target = "Guanaco"
        elif "Vicugna vicugna" in species_key: target = "Vicugna"

        if target and "Espece" in ind_orig.columns:
            mask = (ind_orig["Espece"] == target).values
            gt_filtered  = gt_orig[mask, :]
            ind_filtered = ind_orig[mask].copy().reset_index(drop=True)
        else:
            gt_filtered  = gt_orig
            ind_filtered = ind_orig.copy()
    else:
        gt_filtered  = gt_orig
        ind_filtered = ind_orig.copy()

    # 2. Mise à jour du FID selon le mode de regroupement
    if grouping_mode.startswith("🌍") and "Pays" in ind_filtered.columns:
        ind_filtered["FID"] = ind_filtered["Pays"]
    elif grouping_mode.startswith("🦙") and "Espece" in ind_filtered.columns:
        ind_filtered["FID"] = ind_filtered["Espece"]
    elif grouping_mode.startswith("🏷️") and "Race" in ind_filtered.columns:
        ind_filtered["FID"] = ind_filtered["Race"]
    else:
        ind_filtered["FID"] = ind_filtered["IID"]

    st.session_state.gt       = gt_filtered
    st.session_state.ind_df   = ind_filtered
    st.session_state.snp_df   = st.session_state.snp_df_original.copy()

    st.session_state.last_species_filter = species_key
    st.session_state.last_grouping_filter = grouping_mode

@st.cache_data(show_spinner=False)
def load_map_uploaded(file_bytes: bytes, fname: str) -> pd.DataFrame:
    raw = gzip.decompress(file_bytes) if fname.endswith(".gz") else file_bytes
    return parse_map(raw)

def compute_diversity_metrics(gt, labels):
    """Ho, He, MAF par population (SNP-wise puis moyenne)."""
    labels = np.array(labels)
    pops = np.unique(labels)
    results = []
    for p in pops:
        mask = labels == p
        sub_gt = gt[mask]
        if sub_gt.shape[0] == 0:
            continue
        n_ind = sub_gt.shape[0]
        p_freq = np.nanmean(sub_gt, axis=0) / 2.0
        maf = np.nanmean(np.minimum(p_freq, 1 - p_freq))
        ho  = np.nanmean(np.nanmean(sub_gt == 1, axis=0))
        he  = np.nanmean(2 * p_freq * (1 - p_freq))
        results.append({"Population": p, "N": n_ind,
                        "Ho": round(float(ho), 4),
                        "He": round(float(he), 4),
                        "MAF": round(float(maf), 4)})
    return pd.DataFrame(results)

# ============================================================
# Application principale
# ============================================================
def main():
    with st.sidebar:
        st.title("🦙 NW Camelids")
        st.caption(f"Réf. {REFERENCE['name']} · {REFERENCE['accession']}")

        species_options = ["🌍 Toutes les espèces (Global)"] + list(SPECIES_CONFIG.keys())
        species_key = st.selectbox("🐾 Mode d'analyse", species_options)

        if species_key == "🌍 Toutes les espèces (Global)":
            cfg = get_config("🦙 Alpaga (Vicugna pacos)")
            cfg["prefix"] = "nwc_global"
            cfg["autosomes"] = CAMELID_AUTOSOMES[:]     # 1..36
            st.info("🌍 Mode Global")
        else:
            cfg = get_config(species_key)
            # Garde-fou : forcer 36 autosomes si la config est incohérente
            if list(cfg.get("autosomes", [])) != CAMELID_AUTOSOMES:
                st.warning(f"⚠️ Config : {len(cfg.get('autosomes', []))} autosomes. "
                           "Pour les camélidés (2n=74), 36 attendus. Correction appliquée.")
                cfg["autosomes"] = CAMELID_AUTOSOMES[:]
            st.info(f"Mode spécifique : {species_key}")

        st.markdown(f"**Caryotype** : {cfg['karyotype']}")
        st.markdown(f"**Taxon NCBI** : `{cfg['taxon']}`")
        st.caption(f"Autosomes analysés : 1–36 (exclut X, Y, MT)")

        st.divider()
        st.header("🎯 Mode de regroupement")
        grouping_mode = st.radio(
            "Regrouper les analyses par :",
            ["🌍 Par Pays", "🦙 Par Espèce", "🏷️ Par Race (Huacaya, Suri...)"],
            index=0,
        )

        if (st.session_state.gt_original is not None and
            (st.session_state.last_species_filter != species_key or
             st.session_state.last_grouping_filter != grouping_mode)):
            reset_derived()
            update_data_filters(species_key, grouping_mode)

        st.divider()
        st.header("📁 1. Annotations SNP")
        f_map = st.file_uploader(".map", type=["map", "gz"], key="map_up")
        if f_map and st.button("📥 Charger .map", use_container_width=True):
            try:
                raw_bytes = f_map.read()
                if f_map.name.endswith(".gz"):
                    raw_bytes = gzip.decompress(raw_bytes)
                raw_text = raw_bytes.decode('utf-8', errors='ignore')
                data = []
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        data.append(parts[:4])
                if data:
                    map_df = pd.DataFrame(data, columns=['CHR','SNP','CM','BP'])
                    map_df['CHR'] = map_df['CHR'].astype(str).str.replace(r'\.0$', '', regex=True)
                    map_df['BP']  = pd.to_numeric(map_df['BP'], errors='coerce').fillna(0).astype(int)
                    map_df['CM']  = pd.to_numeric(map_df['CM'], errors='coerce').fillna(0.0)
                    st.session_state.snp_df_original = map_df
                    st.session_state.snp_df = map_df.copy()
                    st.success(f"✅ {len(map_df):,} SNP chargés sur {map_df['CHR'].nunique()} chromosomes")
                else:
                    st.error("❌ Aucune donnée valide.")
            except Exception as e:
                st.error(f"❌ {e}")

        st.divider()
        st.header("🧬 2. Génotypes")
        ped = st.file_uploader(".ped", type=["ped","txt","gz"], key="ped_up")
        if ped and st.session_state.snp_df_original is not None and st.button("📥 Charger PED", use_container_width=True):
            try:
                with st.spinner("Parsing…"):
                    gt, ind, a1, a2 = parse_ped(ped.read(), len(st.session_state.snp_df_original))
                snp = st.session_state.snp_df_original.copy()
                if len(snp) > gt.shape[1]:
                    snp = snp.iloc[:gt.shape[1]].reset_index(drop=True)
                elif len(snp) < gt.shape[1]:
                    gt = gt[:, :len(snp)]
                snp["A1"] = a1; snp["A2"] = a2
                ind = rec_fid(ind)
                st.session_state.gt_original     = gt
                st.session_state.ind_df_original = ind.copy()
                st.session_state.snp_df_original = snp
                reset_derived()
                update_data_filters(species_key, grouping_mode)
                st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")

        st.divider()
        st.header("📋 2.5 Métadonnées (Espèces / Races)")
        meta_file = st.file_uploader("NWC_Meta_Sample_Details_Suheil.xlsx",
                                     type=["xlsx","csv"], key="meta_up")
        if meta_file and st.session_state.ind_df_original is not None:
            try:
                df_excel = pd.read_excel(meta_file) if meta_file.name.endswith(".xlsx") else pd.read_csv(meta_file)
                if st.button("🔗 Extraire les Métadonnées", use_container_width=True):
                    ind_orig = st.session_state.ind_df_original.copy()
                    ind_orig["IID_clean"] = (ind_orig["IID"].astype(str)
                                             .str.replace(".CEL", "", regex=False)
                                             .str.strip().str.upper())
                    ind_orig["Code_Pays"]  = ind_orig["IID_clean"].str.split('_').str[0]
                    ind_orig["Code_Race"]  = ind_orig["IID_clean"].str.split('_').str[1]

                    dict_pays = {'ARG':'Argentina','BOL':'Bolivia','PER':'Peru','USA':'USA','CHL':'Chile'}
                    dict_race = {
                        'BAH':('Alpaca','Bolivian Huacaya'), 'BAS':('Alpaca','Bolivian Suri'),
                        'PAH':('Alpaca','Peruvian Huacaya'), 'PAS':('Alpaca','Peruvian Suri'),
                        'ULP':('Alpaca','US Alpaca'),        'ALG':('Guanaco','Argentinian Guanaco'),
                        'CLG':('Guanaco','US Guanaco'),      'ALL':('Llama','Argentinian Llama'),
                        'BLL':('Llama','Bolivian Llama'),    'AVS':('Vicugna','Argentinian Vicugna'),
                        'BVM':('Vicugna','Bolivian V.Mensalis'),
                        'BVS':('Vicugna','Bolivian V.Sp.'),
                        'BVV':('Vicugna','Bolivian V.Vicugna'),
                        'UVV':('Vicugna','US Vicugna')
                    }
                    ind_orig["Pays"]   = ind_orig["Code_Pays"].map(dict_pays).fillna("Inconnu")
                    ind_orig["Espece"] = ind_orig["Code_Race"].map(lambda x: dict_race.get(x,('Inconnu','Inconnu'))[0])
                    ind_orig["Race"]   = ind_orig["Code_Race"].map(lambda x: dict_race.get(x,('Inconnu','Inconnu'))[1])
                    st.session_state.ind_df_original = ind_orig
                    reset_derived()
                    update_data_filters(species_key, grouping_mode)
                    st.success("✅ Métadonnées extraites avec succès !")
                    st.write("**Répartition par Espèce :**")
                    st.dataframe(ind_orig["Espece"].value_counts().reset_index(), use_container_width=True)
                    st.write("**Répartition par Race :**")
                    st.dataframe(ind_orig["Race"].value_counts().reset_index(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur Excel : {e}")

        st.divider()
        st.header("⚙️ 3. Seuils QC")
        keep_auto = st.checkbox("Exclure X/Y/MT", value=True)
        rem_out   = st.checkbox("Supprimer outliers hét.", value=True)
        geno      = st.slider("Missing SNP", 0.0, 0.5, 0.05)
        mind      = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t     = st.slider("MAF",         0.0, 0.5, 0.05)
        hwe_t     = st.number_input("HWE p", value=1e-6, format="%.0e")
        het_sd    = st.slider("Het ±σ",      1.0, 5.0, 3.0, 0.1)
        ld_r2     = st.slider("LD r²",       0.05, 0.5, 0.2, 0.05)

    current_mode = grouping_mode
    st.title(f"🦙 {species_key}")
    st.caption(f"Référence **{REFERENCE['name']}** · Groupement : **{current_mode}**")

    if st.session_state.gt is None:
        st.info("👋 Chargez `.map`, `.ped` et `.xlsx` pour commencer.")
        return
    if st.session_state.ind_df is None:
        st.warning("⚠️ Veuillez extraire les métadonnées (étape 2.5) pour activer les analyses.")
        return

    TABS = st.tabs([
        "🏠 Accueil","🧹 QC","✂️ LD","🧬 Structure","🎨 Admixture",
        "📈 Démographie","🌳 Phylogénie","🔍 Sélection",
        "🧪 GWAS","🎯 Post-GWAS","📈 GBLUP","🧬 Annotation",
        "🧬 VEP & NCBI","🧬 Structure 3D","🧬 Alignement",
        "🔬 Méta-analyse","🧠 ML","🧬 Haplotypes","🌍 LiftOver","📤 Export",
        "📊 Diversité","⚖️ Comparaison 2 à 2","🧬 Blocs de LD",
        "📊 Métriques comparatives","🧬 Impact Moléculaire"
    ])

    # ---------- 0 Accueil ----------
    with TABS[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Individus",    st.session_state.gt.shape[0])
        c2.metric("SNP",          st.session_state.gt.shape[1])
        c3.metric("Chromosomes",  st.session_state.snp_df["CHR"].nunique())
        c4.metric("Groupes",      st.session_state.ind_df["FID"].nunique())
        ch = st.session_state.snp_df["CHR"].astype(str).value_counts()
        ch = ch.reindex(sorted(ch.index, key=lambda x: (len(x), x)))
        fig_ch = px.bar(x=ch.index, y=ch.values,
                        labels={"x":"Chromosome","y":"Nb SNP"},
                        title="SNP par chromosome")
        fig_ch.update_xaxes(type='category')
        st.plotly_chart(fig_ch, use_container_width=True)
        st.subheader(f"📋 Répartition exacte par {current_mode}")
        st.dataframe(st.session_state.ind_df["FID"].value_counts().reset_index(),
                     use_container_width=True)

    # ---------- 1 QC ----------
    with TABS[1]:
        if st.button("▶ Lancer QC", type="primary", key="qc_btn"):
            try:
                params = {"geno":geno, "mind":mind, "maf":maf_t, "hwe":hwe_t, "het_sd":het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(
                    st.session_state.gt, st.session_state.ind_df,
                    st.session_state.snp_df, params,
                    cfg["autosomes"],
                    keep_autosomes=keep_auto, rem_out=rem_out)
                st.session_state.update({"gt_filt":gf, "ind_filt":inf,
                                         "snp_filt":snf, "qc_stats":qs})
                st.success(f"✅ {gf.shape[0]} × {gf.shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind. finaux",   s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNP finaux",    s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Outliers",      s["trace"].get("het", 0))
            c4.metric("Exclus X/Y/MT", s["trace"].get("autosomes", 0))

    # ---------- 2 LD ----------
    with TABS[2]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ LD Pruning", key="ld_btn"):
                keep = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.update({
                    "gt_pruned": st.session_state.gt_filt[:, keep],
                    "snp_pruned": st.session_state.snp_filt[keep].reset_index(drop=True)})
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNP conservés")
            if st.session_state.gt_pruned is not None:
                if st.button("▶ LD Decay", key="ld_decay_btn"):
                    st.session_state.ld_df = ld_decay(st.session_state.gt_pruned,
                                                     st.session_state.snp_pruned)
                if st.session_state.ld_df is not None:
                    st.plotly_chart(plot_ld(st.session_state.ld_df), use_container_width=True)

    # ---------- 3 Structure ----------
    with TABS[3]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS", key="struct_btn"):
            try:
                gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                      else st.session_state.gt_filt)
                s, v = pca_analysis(gu, 10)
                st.session_state.update({"pca_s":s, "pca_v":v})
                st.session_state.mds_c  = mds_analysis(gu, 5)
                st.session_state.king_K = king_mat(gu)
                st.success("✅ Calculs terminés")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v,
                                     st.session_state.ind_filt["FID"].values),
                            use_container_width=True)
            st.plotly_chart(plot_mds(st.session_state.mds_c,
                                     st.session_state.ind_filt["FID"].values),
                            use_container_width=True)

    # ---------- 4 Admixture ----------
    with TABS[4]:
        if st.session_state.gt_filt is not None:
            K = st.slider("K choisi", 2, 10, 4)
            if st.button("▶ Lancer NMF", key="admix_btn"):
                try:
                    gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                          else st.session_state.gt_filt)
                    Q, _ = admixture_nmf(gu, K=K)
                    st.session_state.update({"admix_Q":Q, "admix_K":K})
                    st.success("✅ Admixture calculé")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                f, _ = plot_admix(st.session_state.admix_Q,
                                  st.session_state.ind_filt["IID"].values,
                                  st.session_state.ind_filt["FID"].values,
                                  st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True)

    # ---------- 5 Démographie ----------
    with TABS[5]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ ROH + Ne", key="dem_btn"):
                try:
                    gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                          else st.session_state.gt_filt)
                    su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                          else st.session_state.snp_filt)
                    roh, fr = detect_roh(gu, su[["CHR","BP"]].to_json())
                    st.session_state.update({"roh_df":roh, "froh":fr})
                    st.success(f"✅ {len(roh)} ROH détectés")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.froh is not None:
                df_f = pd.DataFrame({"FROH": st.session_state.froh,
                                     "Pop":  st.session_state.ind_filt["FID"].values})
                st.plotly_chart(px.histogram(df_f, x="FROH", color="Pop", nbins=50),
                                use_container_width=True)

    # ---------- 6 Phylogénie ----------
    with TABS[6]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ UPGMA", key="phylo_btn"):
                try:
                    gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                          else st.session_state.gt_filt)
                    su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                          else st.session_state.snp_filt)
                    lab = st.session_state.ind_filt["FID"].astype(str).values
                    r = upgma_bootstrap(gu, su, lab, n_boot=100)
                    st.session_state.nj_result = r
                    st.session_state.nj_mid = midpoint_info(r["D"], r["pops"])
                    st.success("✅ Arbre généré")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.nj_result is not None:
                r = st.session_state.nj_result
                st.plotly_chart(plot_tree(r["Z"], r["pops"], r["support"],
                                          title="UPGMA",
                                          mid=st.session_state.nj_mid),
                                use_container_width=True)

    # ---------- 7 Sélection ----------
    with TABS[7]:
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                  else st.session_state.snp_filt)
            lab = st.session_state.ind_filt["FID"].astype(str).values
            if st.button("▶ FST Pairwise", key="fst_btn"):
                try:
                    st.session_state.fst = fst_snp_wc(gu, lab)
                    M, pops = fst_pairwise(gu, lab, use_wc84=True)
                    st.session_state.update({"fst_M":M, "fst_pops":pops})
                    st.success("✅ FST calculé")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.fst_M is not None:
                mdf = pd.DataFrame(st.session_state.fst_M,
                                   index=st.session_state.fst_pops,
                                   columns=st.session_state.fst_pops)
                st.plotly_chart(px.imshow(mdf.round(4), text_auto=True,
                                          color_continuous_scale="Reds"),
                                use_container_width=True)
            st.divider()
            st.subheader("FST Fenêtré")
            if st.button("▶ Scan FST fenêtré", key="wf_btn"):
                try:
                    w = windowed_fst(gu, su, lab, 50, 25)
                    if "MEAN_FST" not in w.columns:
                        fst_cols = [c for c in w.columns if "FST" in c.upper()]
                        if fst_cols:
                            w = w.rename(columns={fst_cols[0]:"MEAN_FST"})
                        else:
                            w["MEAN_FST"] = 0.0
                    st.session_state.win_fst = detect_outliers(w, 3.0)
                    st.success(f"✅ {len(w)} fenêtres")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.win_fst is not None:
                fig_win = plot_win_manh(st.session_state.win_fst,
                                        "MEAN_FST", "FST fenêtré")
                if fig_win is not None:
                    st.plotly_chart(fig_win, use_container_width=True)
                else:
                    st.warning("Pas assez de données valides pour le graphique.")
            st.divider()
            st.subheader("CSS")
            if st.button("▶ CSS", key="css_btn"):
                st.session_state.css_df = composite_score(gu, su, lab)
                st.success("✅ CSS calculé")
            if st.session_state.css_df is not None:
                st.dataframe(st.session_state.css_df.head(20), use_container_width=True)

    # ---------- 8 GWAS ----------
    with TABS[8]:
        pf = st.file_uploader("Phénotypes", type=["csv","txt","gz"], key="ph_up")
        if pf and st.button("📥 Charger", key="ph_btn"):
            try:
                mg, _ = match_pheno(st.session_state.ind_filt,
                                    load_pheno(pf.read(), pf.name))
                st.session_state.pheno_df = mg
                st.success("✅ Phénotypes chargés")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID")]
            pc = st.selectbox("Phénotype", cols)
            if st.session_state.king_K is None:
                st.warning("⚠️ Matrice KING non calculée. Lancez d'abord l'onglet « Structure » "
                           "pour un modèle mixte fiable ; sinon GWAS se fera sans random effect.")
            if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                try:
                    gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                          else st.session_state.gt_filt)
                    su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                          else st.session_state.snp_filt)
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    cv = (st.session_state.pca_s[:, :5] if st.session_state.pca_s is not None
                          else np.empty((gu.shape[0], 0)))
                    r = run_gwas(gu, ph.values, st.session_state.king_K, cv)
                    r = annotate_gwas(r, su); r = postprocess_gwas(r)
                    st.session_state.gwas_res  = r
                    st.session_state.lambda_gc = lambda_gc(r["P"].values)
                    st.success(f"✅ {len(r)} SNP · λGC = {st.session_state.lambda_gc:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.gwas_res is not None:
                st.plotly_chart(plot_manhattan(st.session_state.gwas_res,
                                               yc="P", title="Manhattan"),
                                use_container_width=True)

    # ---------- 9 Post-GWAS ----------
    with TABS[9]:
        if st.session_state.gwas_res is not None:
            pip = fine_mapping(st.session_state.gwas_res)
            st.plotly_chart(plot_pip(pip), use_container_width=True)
            st.dataframe(pip[["SNP","CHR","BP","P","PIP"]].head(30),
                         use_container_width=True)

    # ---------- 10 GBLUP ----------
    with TABS[10]:
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID")]
            pc = st.selectbox("Phénotype", cols, key="gb_pc")
            if st.button("▶ GBLUP", key="gb_btn"):
                try:
                    gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                          else st.session_state.gt_filt)
                    ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    df, stats_ = run_gblup(gu, ph.values)
                    st.session_state.gebv_df      = df
                    st.session_state.gblup_stats  = stats_
                    st.success(f"✅ h² = {stats_['h2']:.3f} · r² = {stats_['r2']:.3f}")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ---------- 11 Annotation ----------
    with TABS[11]:
        st.subheader("🧬 Annotation des SNP candidats")
        gff_f = st.file_uploader("GFF3/GTF (VicPac4)",
                                 type=["gff","gff3","gtf","gz"], key="ann_gff")
        col1, col2 = st.columns(2)
        with col1:
            src = st.radio("Source des SNP :",
                           ["FST Fenêtré","CSS","GWAS","Comparaison 2 à 2"],
                           horizontal=True)
        with col2:
            unit = st.radio("Unité d'affichage :",
                            ["Position (bp)","Distance (kb)","Distance (cM)"],
                            horizontal=True)
        wk = st.slider("Fenêtre de recherche (kb)", 1, 1000, 50)

        if gff_f:
            try:
                gff = parse_gff3(gff_f.read())
                st.session_state.sel_gff = gff
                st.success(f"✅ {len(gff)} gènes trouvés")

                snps = None
                if src == "FST Fenêtré" and st.session_state.win_fst is not None:
                    outliers = st.session_state.win_fst[st.session_state.win_fst["OUTLIER"] == True]
                    base = outliers if not outliers.empty else st.session_state.win_fst.head(50)
                    snps = base[["TOP_SNP","CHR","MID"]].rename(
                        columns={"TOP_SNP":"SNP","MID":"BP"})
                elif src == "CSS" and st.session_state.css_df is not None:
                    snps = st.session_state.css_df.head(50)[["SNP","CHR","BP"]]
                elif src == "GWAS" and st.session_state.gwas_res is not None:
                    snps = st.session_state.gwas_res.sort_values("P").head(50)[["SNP","CHR","BP"]]
                elif src == "Comparaison 2 à 2" and st.session_state.get("snps_specifiques") is not None:
                    snps = st.session_state.snps_specifiques

                # CORRECTION : distinguer "source non lancée" et "source vide"
                if snps is None:
                    st.warning(f"Veuillez d'abord lancer l'analyse « {src} ».")
                elif snps.empty:
                    st.info(f"L'analyse « {src} » n'a retourné aucun SNP à annoter.")
                else:
                    if st.button("🔍 Lancer l'annotation", type="primary"):
                        st.session_state.ann_df = annotate_genes(snps, gff, wk)
                        if not st.session_state.ann_df.empty:
                            st.success(f"✅ {len(st.session_state.ann_df)} associations SNP-Gène trouvées.")
                        else:
                            st.warning(f"⚠️ 0 association. Essayez d'augmenter la fenêtre de recherche.")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            df_res = st.session_state.ann_df.copy()
            st.subheader("📋 Tableau des associations")
            st.dataframe(df_res, use_container_width=True)
            st.subheader("🎯 Représentation visuelle")
            snps_uniques = df_res["SNP"].unique().tolist()
            snp_choisi   = st.selectbox("Choisissez le SNP à visualiser :", snps_uniques)
            df_snp       = df_res[df_res["SNP"] == snp_choisi]
            top_assoc    = df_snp.iloc[0]
            fig = plot_gene_snp(top_assoc, top_assoc, unit=unit)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ Télécharger (CSV)",
                               df_res.to_csv(index=False).encode(),
                               "annotation_genes.csv", "text/csv")

    # ---------- 12 VEP & NCBI ----------
    with TABS[12]:
        st.subheader("🔬 VEP & NCBI (VicPac4)")
        st.info("💡 L'API Ensembl n'est pas encore à jour pour VicPac4. "
                "Nous utilisons l'API officielle NCBI (source de VicPac4).")
        genes_to_search = []
        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            genes_to_search = st.session_state.ann_df["GENE"].unique().tolist()
            st.write(f"**Gènes trouvés :** {', '.join(genes_to_search[:10])}...")
        else:
            st.warning("⚠️ Aucune annotation trouvée. Lancez d'abord l'onglet 'Annotation'.")
        if genes_to_search:
            if st.button("🔬 Récupérer les infos NCBI pour ces gènes", key="ncbi_btn"):
                with st.spinner("Connexion à l'API NCBI..."):
                    ncbi_res = fetch_ncbi_gene_info(genes_to_search, taxon=cfg["taxon"])
                if not ncbi_res.empty:
                    st.session_state.ncbi_gene_res = ncbi_res
                    st.success("✅ Informations NCBI récupérées !")
                else:
                    st.error("❌ L'API NCBI n'a pas retourné de résultats.")
        if st.session_state.ncbi_gene_res is not None:
            st.dataframe(st.session_state.ncbi_gene_res, use_container_width=True)
            st.download_button("⬇ Télécharger (CSV)",
                               st.session_state.ncbi_gene_res.to_csv(index=False).encode(),
                               "genes_ncbi.csv", "text/csv")

    # ---------- 13 Structure 3D ----------
    with TABS[13]:
        st.subheader("🧬 Structure 3D (AlphaFold)")
        st.info("💡 Cette analyse utilise les gènes identifiés dans 'Annotation' "
                "pour récupérer leur structure 3D via AlphaFold.")
        genes_list = []
        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            genes_list = st.session_state.ann_df["GENE"].dropna().unique().tolist()
            genes_list = [g for g in genes_list
                          if g not in ("N/A","nan","") and not g.startswith("LOC")]
        elif st.session_state.vep_res is not None:
            genes_list = st.session_state.vep_res["GENE"].dropna().unique().tolist()
            genes_list = [g for g in genes_list
                          if g not in ("N/A","Intergenic/Non-codant","nan","")]

        if genes_list:
            g = st.selectbox("Choisissez un gène à visualiser en 3D :",
                             genes_list, key="af_gene")
            if st.button("▶ Charger la structure 3D", key="af_btn"):
                with st.spinner(f"Recherche UniProt pour {g}..."):
                    uid = fetch_uniprot_id(g, "9839")
                    if uid:
                        st.success(f"✅ UniProt : {uid}")
                        with st.spinner("Téléchargement AlphaFold..."):
                            pdb = fetch_alphafold_pdb(uid)
                            if pdb:
                                render_3d_structure(pdb, style="cartoon")
                            else:
                                st.error("Structure AlphaFold introuvable.")
                    else:
                        st.warning(f"⚠️ Aucun UniProt pour {g} chez l'Alpaga.")
        else:
            st.warning("⚠️ Aucun gène trouvé. Lancez d'abord 'Annotation'.")

    # ---------- 14 Alignement ----------
    with TABS[14]:
        st.subheader("🧬 Alignement de Séquences")
        st.info("💡 Comparez la séquence de référence (Alpaga) autour d'un SNP "
                "avec une séquence d'une autre espèce (ou collée).")
        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            df_ann  = st.session_state.ann_df.copy()
            snp_sel = st.selectbox("Choisissez un SNP :",
                                   df_ann["SNP"].unique().tolist(), key="align_snp")
            row = df_ann[df_ann["SNP"] == snp_sel].iloc[0]
            st.write(f"**Position :** Chr {row['CHR']} : {row['SNP_POS']} bp")

            # CORRECTION : reset si le SNP a changé
            if st.session_state.get("align_ref_snp") != snp_sel:
                st.session_state.align_ref_seq = None

            if st.button("📥 Récupérer la séquence de référence (Alpaga)", key="get_ref_seq"):
                with st.spinner("Téléchargement depuis NCBI..."):
                    start = max(1, int(row["SNP_POS"]) - 100)
                    end   = int(row["SNP_POS"]) + 100
                    ref_seq = fetch_cds_sequence_from_ncbi(row["CHR"], start, end)
                if ref_seq:
                    st.session_state.align_ref_seq = ref_seq
                    st.session_state.align_ref_snp = snp_sel
                    st.success(f"✅ Séquence récupérée ({len(ref_seq)} bp)")
                else:
                    st.error("❌ Impossible de récupérer la séquence. "
                             "Vérifiez le numéro d'accession du chromosome.")

            if (st.session_state.get("align_ref_seq") is not None
                and st.session_state.get("align_ref_snp") == snp_sel):
                st.text_area("Séquence de référence (Alpaga) :",
                             st.session_state.align_ref_seq, height=100)
                st.subheader("🔬 Séquence à comparer")
                seq_to_compare = st.text_area("Collez une séquence (ADN) ici :", height=100)
                if seq_to_compare:
                    seq_to_compare = re.sub(r"[^ATCG]", "", seq_to_compare.upper())
                    st.write(f"Longueur à comparer : {len(seq_to_compare)} bp")
                    if st.button("🚀 Lancer l'alignement", key="run_align"):
                        with st.spinner("Alignement en cours..."):
                            try:
                                from Bio import Align
                                aligner = Align.PairwiseAligner()
                                aligner.mode = "global"
                                alignments = aligner.align(st.session_state.align_ref_seq,
                                                           seq_to_compare)
                                if len(alignments) > 0:
                                    aln = alignments[0]
                                    html = ("<div style='font-family: monospace; "
                                            "font-size: 14px; overflow-x: auto;'>")
                                    html += "<b>Alpaga   :</b> "
                                    for char in aln[0]:
                                        html += f"<span style='color: grey;'>{char}</span>"
                                    html += "<br>"
                                    html += "<b>Comparée :</b> "
                                    for ref_char, query_char in zip(aln[0], aln[1]):
                                        if ref_char == "-" or query_char == "-":
                                            html += (f"<span style='color: orange; "
                                                     f"font-weight: bold;'>{query_char}</span>")
                                        elif ref_char != query_char:
                                            html += (f"<span style='background-color: yellow; "
                                                     f"color: red; font-weight: bold;'>"
                                                     f"{query_char}</span>")
                                        else:
                                            html += f"<span style='color: green;'>{query_char}</span>"
                                    html += "</div>"
                                    st.markdown(html, unsafe_allow_html=True)
                                    matches = sum(1 for a, b in zip(aln[0], aln[1]) if a == b)
                                    identity = (matches / len(aln[0])) * 100
                                    st.metric("Identité de séquence", f"{identity:.2f} %")
                                else:
                                    st.warning("Aucun alignement trouvé.")
                            except Exception as e:
                                st.error(f"❌ Erreur d'alignement : {e}")
        else:
            st.warning("⚠️ Veuillez d'abord lancer une annotation.")

    # ---------- 15 Méta-analyse ----------
    with TABS[15]:
        st.subheader("🔬 Méta-analyse (Enrichissement Fonctionnel)")
        st.info("💡 Utilise les gènes identifiés dans 'Annotation' pour trouver "
                "les voies biologiques (KEGG/GO) enrichies.")
        genes_enr = []
        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            genes_enr = st.session_state.ann_df["GENE"].dropna().unique().tolist()
            genes_enr = [g for g in genes_enr
                         if g not in ("N/A","nan","") and not g.startswith("LOC")]
        elif st.session_state.vep_res is not None:
            genes_enr = st.session_state.vep_res["GENE"].dropna().unique().tolist()
            genes_enr = [g for g in genes_enr
                         if g not in ("N/A","Intergenic/Non-codant","nan","")]

        if genes_enr:
            st.write(f"**Gènes analysés ({len(genes_enr)}) :** {', '.join(genes_enr[:10])}...")
            if st.button("🚀 Lancer l'enrichissement (KEGG & GO)", key="enr_btn"):
                with st.spinner("Interrogation Enrichr (~30 sec)..."):
                    enr = run_enrichment(genes_enr, organism="Human")
                if not enr.empty:
                    st.success("✅ Enrichissement terminé !")
                    st.dataframe(enr.head(30), use_container_width=True)
                    st.download_button("⬇ Télécharger (CSV)",
                                       enr.to_csv(index=False).encode(),
                                       "enrichissement_kegg.csv", "text/csv")
                else:
                    st.warning("⚠️ Aucune voie enrichie trouvée.")
        else:
            st.warning("⚠️ Aucun gène trouvé. Lancez d'abord 'Annotation'.")

    # ---------- 16 ML ----------
    with TABS[16]:
        if st.session_state.pheno_df is not None:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID")]
            pc = st.selectbox("Phénotype", cols, key="ml_pc")
            if st.button("🚀 Random Forest", key="ml_btn"):
                gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                      else st.session_state.gt_filt)
                ph = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                valid = np.isfinite(ph.values)
                if valid.sum() >= 20:
                    X = impute_mean(gu[valid]); y = ph.values[valid]
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(Xtr, ytr)
                    r2 = r2_score(yte, rf.predict(Xte))
                    st.success(f"✅ R² = {r2:.3f}")
                else:
                    st.warning("⚠️ Pas assez d'individus phénotypés (≥20 requis).")

    # ---------- 17 Haplotypes ----------
    with TABS[17]:
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                  else st.session_state.snp_filt)
            iu = st.session_state.ind_filt
            chroms   = sorted(su["CHR"].astype(str).unique())
            sel_chr  = st.selectbox("Chromosome", chroms, key="hap_chr")
            sub_idx  = su[su["CHR"].astype(str) == sel_chr].index
            if len(sub_idx) > 0:
                if len(sub_idx) > 50:
                    sub_idx = sub_idx[:50]
                snps_names = su.loc[sub_idx, "SNP"].values
                gt_sub     = gu[:, sub_idx]
                groups     = iu["FID"].astype(str).values
                sort_idx   = np.argsort(groups)
                fig = px.imshow(gt_sub[sort_idx],
                                labels=dict(x="SNP", y="Individu", color="Génotype"),
                                x=snps_names, y=groups[sort_idx],
                                color_continuous_scale=["#F0F0F0","#3498db","#e74c3c"],
                                title=f"Haplotypes CHR{sel_chr}", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

    # ---------- 18 LiftOver ----------
    with TABS[18]:
        if st.session_state.snp_filt is not None:
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                  else st.session_state.snp_filt)
            s = st.selectbox("SNP", su["SNP"].tolist(), key="lift_snp")
            if st.button("🔄 Convertir", key="lift_btn"):
                row = su[su["SNP"] == s].iloc[0]
                hc, hb = lift_over(str(row["CHR"]), int(row["BP"]))
                if hc:
                    st.success(f"✅ Homme : chr{hc}:{hb}")
                else:
                    st.error("Aucune correspondance.")

    # ---------- 19 Export ----------
    with TABS[19]:
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                  else st.session_state.snp_filt)
            if st.button("📦 PLINK", use_container_width=True):
                st.download_button("⬇ PLINK",
                                   build_plink(gu, st.session_state.ind_filt, su, cfg["prefix"]),
                                   f"{cfg['prefix']}.zip", "application/zip")
            if st.button("📄 VCF", use_container_width=True):
                st.download_button("⬇ VCF",
                                   build_vcf(gu, st.session_state.ind_filt, su, cfg["prefix"]).encode(),
                                   f"{cfg['prefix']}.vcf", "text/plain")

    # ---------- 20 Diversité ----------
    with TABS[20]:
        st.subheader(f"📊 Métriques de Diversité — {current_mode}")
        if st.session_state.gt_filt is not None:
            if st.button("🧮 Calculer les métriques"):
                gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                      else st.session_state.gt_filt)
                labels = st.session_state.ind_filt["FID"].astype(str).values
                df_div = compute_diversity_metrics(gu, labels)
                if st.session_state.froh is not None:
                    df_froh = (pd.DataFrame({"Population":labels,
                                             "FROH": st.session_state.froh})
                               .groupby("Population")["FROH"].mean().reset_index())
                    df_div = df_div.merge(df_froh, on="Population", how="left")
                    df_div["FROH"] = df_div["FROH"].round(4)
                st.dataframe(df_div, use_container_width=True)
                st.download_button("⬇ Télécharger",
                                   df_div.to_csv(index=False).encode(),
                                   "diversity_metrics.csv", "text/csv")

    # ---------- 21 Comparaison 2 à 2 (CORRIGÉE) ----------
    with TABS[21]:
        st.subheader("⚖️ Comparaison directe entre deux groupes")
        st.caption("Formule : |fréquence(groupe 1) − fréquence(groupe 2)| ≥ seuil. "
                   "Défaut 0.80 pour reproduire le Tableau 3 du rapport.")
        if st.session_state.gt_filt is not None:
            pops = st.session_state.ind_filt["FID"].unique().tolist()
            if len(pops) >= 2:
                col1, col2 = st.columns(2)
                pop1 = col1.selectbox("Groupe 1", pops, index=0)
                pop2 = col2.selectbox("Groupe 2", pops, index=1 if len(pops) > 1 else 0)

                # CORRECTION : libellé clair + défaut 0.80 (et non 0.90)
                seuil = st.slider(
                    "Différence minimale de fréquence allélique  |p1 − p2|",
                    0.0, 1.0, 0.80, 0.05)

                if pop1 == pop2:
                    st.error("⚠️ Les deux groupes doivent être différents.")
                elif st.button("🔬 Lancer la comparaison", key="comp_btn"):
                    gu   = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                            else st.session_state.gt_filt)
                    su   = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                            else st.session_state.snp_filt)
                    labs = st.session_state.ind_filt["FID"].astype(str).values

                    m1 = (labs == pop1); m2 = (labs == pop2)
                    if m1.sum() > 0 and m2.sum() > 0:
                        sub_gt  = gu[m1 | m2]
                        sub_lab = labs[m1 | m2]
                        try:
                            fst_m, _ = fst_pairwise(sub_gt, sub_lab)
                            st.metric(f"FST entre {pop1} et {pop2}",
                                      f"{fst_m[0,1]:.4f}")
                        except Exception as e:
                            st.warning(f"FST non calculable : {e}")

                        g1 = gu[m1]; g2 = gu[m2]
                        p1 = np.nanmean(g1, axis=0) / 2.0
                        p2 = np.nanmean(g2, axis=0) / 2.0

                        # CORRECTION : |p1 - p2| >= seuil (et non p1>=s & p2<=1-s)
                        diff      = np.abs(p1 - p2)
                        valid     = np.isfinite(p1) & np.isfinite(p2)
                        diff_haute = valid & (diff >= seuil)

                        n_diff = int(np.sum(diff_haute))
                        st.write(f"**SNP retenus (|Δfréq| ≥ {seuil:.2f}) : {n_diff}**")

                        # Tableau récapitulatif multi-seuils (comme le rapport)
                        st.markdown("**Décompte aux seuils clés du rapport :**")
                        cols_s = st.columns(3)
                        for i, t in enumerate([0.65, 0.80, 0.90]):
                            n_t = int(np.sum(valid & (diff >= t)))
                            cols_s[i].metric(f"Seuil {t:.2f}", f"{n_t} SNP")

                        if n_diff > 0:
                            df_diff = su[diff_haute].copy()
                            df_diff["Freq_" + str(pop1)] = p1[diff_haute]
                            df_diff["Freq_" + str(pop2)] = p2[diff_haute]
                            st.dataframe(
                                df_diff[["CHR","SNP","BP",
                                         "Freq_"+str(pop1),
                                         "Freq_"+str(pop2)]],
                                use_container_width=True)
                            st.session_state.snps_specifiques = df_diff[["SNP","CHR","BP"]].copy()
                            st.success(f"✅ {n_diff} SNP sauvegardés pour l'annotation.")
                        else:
                            st.warning("Aucun SNP ne dépasse ce seuil. "
                                       "Essayez de baisser le curseur (ex: 0.65).")
                    else:
                        st.error("Un des groupes n'a pas d'individus.")

    # ---------- 22 Blocs de LD ----------
    with TABS[22]:
        st.subheader("🧬 Carte de chaleur du Déséquilibre de Liaison (LD)")
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                  else st.session_state.snp_filt)
            chroms  = sorted(su["CHR"].astype(str).unique())
            sel_chr = st.selectbox("Choisir un chromosome", chroms, key="ld_chr")
            idx_chr = su[su["CHR"].astype(str) == sel_chr].index
            if len(idx_chr) > 100:
                idx_chr = idx_chr[:100]
            if len(idx_chr) > 1:
                gt_chr = gu[:, idx_chr]
                X = np.nan_to_num(gt_chr, nan=np.nanmean(gt_chr, axis=0))
                valid_cols = X.std(axis=0) > 1e-9
                X_val = X[:, valid_cols]
                if X_val.shape[1] > 1:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr_matrix = np.corrcoef(X_val.T) ** 2
                    fig = px.imshow(corr_matrix, color_continuous_scale="Reds",
                                    labels=dict(color="r²"),
                                    title=(f"Blocs de LD - Chromosome {sel_chr} "
                                           f"({len(idx_chr)} SNP)"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Pas assez de SNPs variables sur ce chromosome.")
            else:
                st.info("Pas assez de SNP sur ce chromosome.")

    # ---------- 23 Métriques comparatives ----------
    with TABS[23]:
        st.subheader("📊 Tableau comparatif des races/espèces")
        if st.session_state.gt_filt is not None:
            if st.button("Générer le tableau comparatif"):
                gu   = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                        else st.session_state.gt_filt)
                labs = st.session_state.ind_filt["FID"].astype(str).values
                pops = np.unique(labs)
                results = []
                for p in pops:
                    mask   = (labs == p)
                    sub_gt = gu[mask]
                    if sub_gt.shape[0] == 0:
                        continue
                    ho  = np.nanmean(np.nanmean(sub_gt == 1, axis=0))
                    p_f = np.nanmean(sub_gt, axis=0) / 2.0
                    maf = np.nanmean(np.minimum(p_f, 1 - p_f))
                    results.append({"Population":p,
                                    "N": sub_gt.shape[0],
                                    "Ho (Hétérozygotie)": round(float(ho), 4),
                                    "MAF moyen":         round(float(maf), 4)})
                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True)
                st.download_button("⬇ Télécharger (CSV)",
                                   df_res.to_csv(index=False).encode(),
                                   "comparaison_races.csv", "text/csv")

    # ---------- 24 Impact Moléculaire ----------
    with TABS[24]:
        st.subheader("🧬 Analyse de l'Impact Moléculaire d'une Mutation")
        st.info("💡 Analyse la conséquence d'un SNP sur la protéine, puis guide "
                "vers les outils externes pour SIFT/PolyPhen.")
        if st.session_state.ann_df is not None and not st.session_state.ann_df.empty:
            df_ann     = st.session_state.ann_df.copy()
            df_in_gene = df_ann[df_ann["DIST_BP"] == 0]
            if not df_in_gene.empty:
                snp_sel = st.selectbox("Choisissez un SNP (situé dans un gène) :",
                                       df_in_gene["SNP"].unique().tolist())
                row = df_in_gene[df_in_gene["SNP"] == snp_sel].iloc[0]
                st.write(f"**Gène :** {row['GENE']}")
                st.write(f"**Position :** Chr {row['CHR']} : {row['SNP_POS']} bp")
                if st.button("🔬 Analyser la mutation"):
                    with st.spinner("Récupération de la séquence du gène (NCBI)..."):
                        seq = fetch_cds_sequence_from_ncbi(
                            row["CHR"],
                            max(1, int(row["SNP_POS"]) - 10),
                            int(row["SNP_POS"]) + 10)
                    if seq and len(seq) >= 21:
                        st.success(f"✅ Séquence récupérée ({len(seq)} bp)")
                        for frame in range(3):
                            codons = [seq[i:i+3] for i in range(frame, len(seq)-2, 3)]
                            if len(codons) >= 7:
                                central = codons[6]
                                aa = translate_codon(central)
                                if aa not in ('X','*'):
                                    st.write(f"**Cadre {frame+1} :** Codon {central} "
                                             f"→ Acide aminé **{aa}**")
                        st.warning("⚠️ Pour l'exon exact et la position du codon dans la "
                                   "protéine, utilisez VEP (voir ci-dessous).")
                    else:
                        st.error("❌ Impossible de récupérer la séquence.")
                st.divider()
                st.subheader("🎯 Prédiction de l'Impact Fonctionnel (SIFT / PolyPhen-2)")
                st.markdown(f"""
                **Pour obtenir SIFT et PolyPhen-2** :

                1. Allez sur **[Ensembl VEP](https://www.ensembl.org/Tools/VEP)**.
                2. Dans « Variant », entrez : `{row['CHR']} {row['SNP_POS']} . A G` (adaptez les allèles).
                3. Sélectionnez l'espèce **"Alpaca"**.
                4. Cliquez sur "Run". VEP vous donnera :
                    *   Le changement d'acide aminé (ex: `p.Arg273His`).
                    *   Le score **SIFT** (délétère si < 0.05).
                    *   Le score **PolyPhen-2** (probablement délétère si > 0.85).
                """)
            else:
                st.warning("⚠️ Aucun SNP 'Dans le gène' trouvé. Lancez d'abord une annotation.")
        else:
            st.warning("⚠️ Veuillez d'abord lancer une annotation.")

if __name__ == "__main__":
    main()
