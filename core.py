"""core.py — Parsers, QC et export"""
from __future__ import annotations
import gzip, io, gc
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

def _rt(raw: bytes) -> str:
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def _ca(a) -> str:
    s = str(a).strip() if a is not None else "0"
    return "0" if s in ("", ".", "-", "NA", "nan", "None", "0", "00") else s

def _chk(c):
    s = str(c).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _autw(ars, autosomes):
    L = np.array([ars[c] for c in autosomes], dtype=float)
    return L / L.sum()

def rec_fid(ind):
    df = ind.copy()
    if df["FID"].nunique() <= 1:
        df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

def sync_dims(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df))
    m = min(gt.shape[1], len(snp_df))
    return gt[:n, :m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def impute_mean(gt):
    g = gt.astype(np.float32, copy=True)
    cm = np.nanmean(g, axis=0); cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(g)
    if nm.any(): g[nm] = np.take(cm, np.where(nm)[1])
    return g

def parse_ped(b, n_map):
    lines = [l.strip() for l in _rt(b).splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    
    first_line = lines[0]
    
    # Détection du séparateur (virgule, tabulation ou espace)
    if "\t" in first_line: sep = "\t"
    elif "," in first_line: sep = ","
    elif ";" in first_line: sep = ";"
    else: sep = None # split() gère les espaces
    
    p0 = first_line.split(sep) if sep else first_line.split()
    
    # Détection de l'en-tête (Header) à ignorer
    if len(p0) <= 6 or "FID" in p0[0].upper() or "IID" in p0[1].upper() or first_line.startswith("#"):
        lines = lines[1:]
        if not lines: raise ValueError("Fichier vide après l'en-tête.")
        first_line = lines[0]
        p0 = first_line.split(sep) if sep else first_line.split()
        
    n_cols = len(p0)
    
    # Détection du format : .ped classique (2 colonnes/SNP) ou .raw (1 colonne/SNP)
    if n_cols >= (n_map * 2 + 6) - 10:
        is_raw = False
        n_ped = (n_cols - 6) // 2
    elif n_cols >= (n_map + 6) - 10:
        is_raw = True
        n_ped = n_cols - 6
    else:
        raise ValueError(f"Format inconnu ({n_cols} colonnes lues). Aperçu: {first_line[:60]}")
        
    n = min(n_map, n_ped)
    ec = 6 + (n if is_raw else 2 * n)
    
    fids, iids, rows = [], [], []
    for line in lines:
        p = line.split(sep) if sep else line.split()
        if len(p) >= ec:
            fids.append(p[0]); iids.append(p[1])
            rows.append(p[6:ec])
            
    if not iids: raise ValueError("Aucun individu valide (ou lignes trop courtes).")
    
    ac = [Counter() for _ in range(n)]
    for r in rows:
        for j in range(n):
            if is_raw:
                a1 = r[j].strip()
                if a1 not in ("0", "NA", "nan", "-"):
                    ac[j].update([a1])
            else:
                a1, a2 = r[2*j].strip(), r[2*j + 1].strip()
                if a1 not in ("0", "NA", "nan", "-") and a2 not in ("0", "NA", "nan", "-"):
                    ac[j].update([a1, a2])
                    
    alleles_A1, alleles_A2 = [], []
    for c in ac:
        common = c.most_common(2)
        if len(common) >= 2: 
            alleles_A1.append(common[0][0]); alleles_A2.append(common[1][0])
        elif len(common) == 1: 
            alleles_A1.append(common[0][0]); alleles_A2.append(".")
        else: 
            alleles_A1.append("."); alleles_A2.append(".")
            
    minor = [min(c, key=c.get) if len(c) >= 2 else None for c in ac]
    gt = np.full((len(iids), n), np.nan, dtype=np.float32)
    
    for i, r in enumerate(rows):
        for j in range(n):
            if is_raw:
                val = r[j].strip()
                if val not in ("0", "NA", "nan", "-"):
                    try: gt[i, j] = float(val)
                    except: pass
            else:
                if minor[j] and r[2*j] not in ("0", "NA", "-") and r[2*j+1] not in ("0", "NA", "-"):
                    gt[i, j] = (r[2*j] == minor[j]) + (r[2*j+1] == minor[j])
                    
    return gt, pd.DataFrame({"FID": fids, "IID": iids}), alleles_A1, alleles_A2

def parse_vcf(b):
    samples, rows = [], []
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"): samples = line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p = line.split("\t")
            if len(p) >= 10 and "," not in p[4]:
                fmt = p[8].split(":")
                if "GT" not in fmt: continue
                gi = fmt.index("GT")
                gts = np.full(len(samples), np.nan, dtype=np.float32)
                for k, s in enumerate(p[9:]):
                    al = s.split(":")[gi].replace("|", "/").split("/")
                    if not any(a in (".", "") for a in al):
                        try: gts[k] = sum(int(a) for a in al)
                        except ValueError: pass
                try: bp = int(float(p[1]))
                except (ValueError, TypeError): continue
                rows.append({"CHR": str(p[0]), "SNP": p[2] or f"{p[0]}:{p[1]}",
                             "CM": 0.0, "BP": bp, "A1": p[3], "A2": p[4], "GT": gts})
    if not rows: raise ValueError("Aucun variant VCF valide")
    gt = np.zeros((len(samples), len(rows)), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]
        v = gt[:, j][~np.isnan(gt[:, j])]
        if len(v) > 0 and v.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(gt[:, j]), np.nan, 2.0 - gt[:, j])
    df_snp = pd.DataFrame([{k: v for k, v in r.items() if k != "GT"} for r in rows])
    if df_snp["BP"].max() < 10_000: df_snp["BP"] = df_snp["BP"] * 1_000_000
    elif df_snp["BP"].max() < 10_000_000: df_snp["BP"] = df_snp["BP"] * 1_000
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), df_snp

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal", "IID", "id", "ID"] if c in p.columns), None)
        if id_c is None: raise ValueError("Aucune colonne ID dans phénotypes")
        p[id_c] = p[id_c].astype(str)
        map_a = next((c for c in m.columns if c != "IID"), None)
        m[map_a] = m[map_a].astype(str); m["IID"] = m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left")
        ind = ind_df.copy(); ind["_k"] = ind["IID"].astype(str); pm["_k"] = pm["IID"].astype(str)
        return ind[["_k", "FID", "IID"]].merge(pm.drop(columns=["IID"], errors="ignore"),
                                                on="_k", how="left").drop(columns="_k"), id_c
    id_c = next((c for c in ["IID", "id", "ID", "sample", "Animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError("Aucune colonne ID")
    p = pheno_df.copy(); p[id_c] = p[id_c].astype(str)
    ind = ind_df.copy(); ind["_k"] = ind["IID"].astype(str); p["_k"] = p[id_c]
    return ind[["_k", "FID", "IID"]].merge(p.drop(columns=[id_c]), on="_k", how="left").drop(columns="_k"), id_c

def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g):
    p = np.nanmean(g, axis=0) / 2.0
    return np.minimum(p, 1.0 - p)
def het(g): return np.nanmean(g == 1, axis=1)

def hwe_exact(nh, n0, n2):
    n = nh + n0 + n2
    if n == 0 or (nh == 0 and (n0 == 0 or n2 == 0)): return 1.0
    rare = 2 * min(n0, n2) + nh
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    pr = np.zeros(rare + 1); pr[mid] = 1.0; s = 1.0
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        pr[ch + 2] = pr[ch] * 4 * cr * cc / ((ch + 2) * (ch + 1))
        s += pr[ch + 2]; ch += 2; cr -= 1; cc -= 1
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        pr[ch - 2] = pr[ch] * ch * (ch - 1) / (4 * (cr + 1) * (cc + 1))
        s += pr[ch - 2]; ch -= 2; cr += 1; cc += 1
    po = pr[nh] if nh < len(pr) else 0.0
    return float(min(pr[pr <= po + 1e-7].sum() / s, 1.0))

@st.cache_data(show_spinner=False)
def hwe_pvals(g):
    pv = np.full(g.shape[1], np.nan)
    n0 = ((g == 0) & ~np.isnan(g)).sum(0)
    n1 = ((g == 1) & ~np.isnan(g)).sum(0)
    n2 = ((g == 2) & ~np.isnan(g)).sum(0)
    nv = n0 + n1 + n2
    if g.shape[1] <= 20000:
        for j in range(g.shape[1]):
            if nv[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(nv > 0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2.0 * p * (1.0 - p) * nv
            ok = (eh >= 5) & np.isfinite(eh)
            chi2 = (np.abs(n1 - eh) - 0.5) ** 2 / eh
            pv[ok] = 1.0 - stats.chi2.cdf(chi2[ok], df=1)
    return pv

def apply_qc(gt, ind_df, snp_df, params, autosomes, keep_autosomes=True, rem_out=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df)
    n0, m0 = gt.shape; tr = {}
    if keep_autosomes:
        mask = snp_df["CHR"].astype(str).str.upper().isin([a.upper() for a in autosomes])
        gt = gt[:, mask]; snp_df = snp_df[mask].reset_index(drop=True)
        tr["autosomes"] = int(m0 - gt.shape[1])
    k = miss_snp(gt) <= params["geno"]; gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    tr["geno"] = int(m0 - gt.shape[1])
    k = miss_ind(gt) <= params["mind"]; gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True)
    tr["mind"] = int(n0 - gt.shape[0])
    m = maf(gt); k = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    pv = hwe_pvals(gt); k = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus après MAF/HWE")
    gt_pre, ind_pre, snp_pre = gt.copy(), ind_df.copy(), snp_df.copy()
    if rem_out:
        h = het(gt); kp = np.ones(len(h), dtype=bool)
        for f in ind_df["FID"].unique():
            msk = (ind_df["FID"] == f).values; sub = h[msk]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub - np.nanmean(sub)) / np.nanstd(sub)
                idx = np.where(msk)[0]; kp[idx[np.abs(z) > params["het_sd"]]] = False
        gt = gt[kp]; ind_df = ind_df[kp].reset_index(drop=True)
        tr["het"] = int(len(h) - len(ind_df))
    gc.collect()
    return gt, ind_df, snp_df, gt_pre, ind_pre, snp_pre, {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]), "excluded_snp": int(m0 - gt.shape[1]),
        "trace": tr,
    }

def build_plink(gt, ind_df, snp_df, prefix="genome"):
    import zipfile
    from io import BytesIO
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        n, m = gt.shape; nb = (n + 3) // 4
        buf = bytearray([0x6C, 0x1B, 0x01])
        for j in range(m):
            bits = np.zeros(n, dtype=np.uint8)
            bits[np.isnan(gt[:, j])] = 0b01
            bits[gt[:, j] == 0] = 0b11
            bits[gt[:, j] == 1] = 0b10
            bits[gt[:, j] == 2] = 0b00
            pk = np.zeros(nb, dtype=np.uint8)
            for k in range(n): pk[k // 4] |= (bits[k] & 0x03) << (2 * (k % 4))
            buf.extend(pk.tobytes())
        zf.writestr(f"{prefix}.bed", bytes(buf))
        zf.writestr(f"{prefix}.bim", "\n".join(
            f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t{int(r['BP'])}\tA\tG"
            for _, r in snp_df.iterrows()) + "\n")
        zf.writestr(f"{prefix}.fam", "\n".join(
            f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
            for _, r in ind_df.iterrows()) + "\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="NWCamelids"):
    header = [
        "##fileformat=VCFv4.2", f"##source={proj}",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(ind_df["IID"].astype(str)),
    ]
    body = []
    for j in range(gt.shape[1]):
        r = snp_df.iloc[j]
        gts = []
        for i in range(gt.shape[0]):
            v = gt[i, j]
            gts.append("./." if np.isnan(v) else "0/0" if v == 0 else "0/1" if v == 1 else "1/1")
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(header + body) + "\n"
