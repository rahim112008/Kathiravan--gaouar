"""analyses.py — Structure, sélection, GWAS, annotation"""
from __future__ import annotations
import io, json, time
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
from core import impute_mean, _chk

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

@st.cache_data(show_spinner=False)
def pca_analysis(gt, nc=10):
    X = impute_mean(gt); X = X - X.mean(0)
    n = min(nc, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=n)
    return p.fit_transform(X), p.explained_variance_ratio_ * 100.0

@st.cache_data(show_spinner=False)
def mds_analysis(gt, nc=5):
    X = impute_mean(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n): D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=nc, dissimilarity="precomputed",
                      random_state=42, n_init=1, max_iter=300).fit_transform(D)

@st.cache_data(show_spinner=False)
def king_mat(gt):
    h = np.nansum(gt == 1, axis=1).astype(np.float64)
    H = (gt == 1).astype(np.float32)
    BH = H @ H.T
    d = h[:, None] + h[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (d - 2 * BH) / d
    K[~np.isfinite(K)] = 0.0
    np.fill_diagonal(K, 0.5)
    return K

@st.cache_data(show_spinner=False)
def admixture_nmf(gt, K=3, seed=42):
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500)
    W = m.fit_transform(np.clip(impute_mean(gt), 0.0, 2.0))
    s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, m.components_

@st.cache_data(show_spinner=False)
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0); rows = []
    for k in range(k_min, k_max + 1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        W = m.fit_transform(X); H = m.components_
        rmse = float(np.sqrt(np.mean((X - W @ H) ** 2)))
        rows.append({"K": k, "RMSE": rmse})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def reynolds(gt, _labels):
    labels = np.asarray(_labels).astype(str)
    pops = sorted(np.unique(labels)); K = len(pops)
    ph = {p: np.nanmean(gt[labels == p], 0) / 2.0 for p in pops}
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi, pj = ph[pops[i]], ph[pops[j]]
            num = np.nansum((pi - pj) ** 2)
            dn = np.nansum(pi * (1 - pj) + pj * (1 - pi))
            if dn > 0: D[i, j] = D[j, i] = num / dn
    return D, pops

@st.cache_data(show_spinner=False)
def upgma_bootstrap(gt, snp_df, _labels, n_boot=100, seed=42):
    from scipy.cluster.hierarchy import to_tree
    labels = np.asarray(_labels).astype(str)
    pops = sorted(np.unique(labels)); K = len(pops); n_snp = gt.shape[1]
    def cR(g):
        ph = {p: np.nanmean(g[labels == p], 0) / 2.0 for p in pops}
        D = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                pi, pj = ph[pops[i]], ph[pops[j]]
                num = np.nansum((pi - pj) ** 2)
                dn = np.nansum(pi * (1 - pj) + pj * (1 - pi))
                if dn > 0: D[i, j] = D[j, i] = num / dn
        return D
    Do = cR(gt)
    Zo = linkage(squareform(Do, checks=False), method="average")
    nl = K
    def gc(Z):
        cl = {}
        for k, row in enumerate(Z):
            i, j = int(row[0]), int(row[1])
            l = frozenset([i]) if i < nl else cl[i]
            r = frozenset([j]) if j < nl else cl[j]
            cl[nl + k] = frozenset(l | r)
        return cl
    co = gc(Zo); sc = np.zeros(len(Zo))
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.choice(n_snp, n_snp, replace=True)
        Db = cR(gt[:, idx])
        Zb = linkage(squareform(Db, checks=False), method="average")
        cb = set(gc(Zb).values())
        for k, cl in co.items():
            zi = k - nl
            if 0 <= zi < len(Zo) and cl in cb: sc[zi] += 1
    supp = (sc / n_boot * 100).round(0)
    tr = to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l = nw(node.get_left()); r = nw(node.get_right())
        if node.id >= nl: return f"({l},{r}){int(supp[node.id - nl])}:{node.dist:.4f}"
        return f"({l},{r}):{node.dist:.4f}"
    return {"tree": tr, "Z": Zo, "newick": nw(tr) + ";",
            "support": supp, "D": Do, "pops": pops}

@st.cache_data(show_spinner=False)
def midpoint_info(D, labels):
    D = np.asarray(D, dtype=float); n = len(labels)
    D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    im, jm, md = 0, 1, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > md: md = D[i, j]; im, jm = i, j
    return {"leaf_i": labels[im], "leaf_j": labels[jm],
            "max_dist": float(md), "midpoint": float(md / 2)}

def plot_pca(s, v, l):
    return px.scatter(pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": l}),
                      x="PC1", y="PC2", color="Pop",
                      title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)

def plot_mds(m, l):
    return px.scatter(pd.DataFrame({"MDS1": m[:, 0], "MDS2": m[:, 1], "Pop": list(l)}),
                      x="MDS1", y="MDS2", color="Pop", title="MDS Reynolds", height=520)

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labs; df["Pop"] = pops
    df["_d"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po", "_d"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}",
                             marker_color=px.colors.qualitative.Set2[k % 8]))
    fig.update_layout(barmode="stack", title=f"Admixture K={K}", height=500)
    return fig, df

def plot_tree(Z, labels, support=None, title="", mid=None):
    n = len(labels); labels = [str(l) for l in labels]
    lo = leaves_list(Z); ly = {int(l): i for i, l in enumerate(lo)}
    cy = {i: ly[i] for i in range(n)}
    cx = {i: 0.0 for i in range(n)}
    fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2])
        yi, yj = cy[i], cy[j]; xi, xj = cx[i], cx[j]
        if support is not None and k < len(support):
            s = support[k]
            col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        else: col = "#3498db"
        fig.add_trace(go.Scatter(x=[xi, d], y=[yi, yi], mode="lines",
                                 line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[xj, d], y=[yj, yj], mode="lines",
                                 line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[d, d], y=[yi, yj], mode="lines",
                                 line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        ni = n + k; cy[ni] = (yi + yj) / 2; cx[ni] = d
        if support is not None and k < len(support):
            s = support[k]
            fig.add_annotation(x=d, y=(yi + yj) / 2, text=f"<b>{int(s)}</b>",
                               showarrow=False, font=dict(size=10, color=col), yshift=8)
    for leaf, y in ly.items():
        fig.add_annotation(x=0, y=y, text=f"<b>{labels[leaf]}</b>",
                           showarrow=False, xanchor="right", xshift=-8, font=dict(size=12))
    if mid: fig.add_vline(x=mid["midpoint"], line_dash="dot", line_color="purple")
    fig.update_layout(title=title, xaxis_title="Distance de Reynolds",
                      yaxis=dict(showticklabels=False), height=max(500, n * 35),
                      margin=dict(l=120, r=40, t=80, b=40))
    return fig

@st.cache_data(show_spinner=False)
def fst_snp_wc(gt, _labels):
    pops = np.unique(_labels); r = len(pops); n_snp = gt.shape[1]
    fst = np.full(n_snp, np.nan)
    for j in range(n_snp):
        n_i = np.zeros(r); p_i = np.zeros(r); h_i = np.zeros(r)
        for k, p in enumerate(pops):
            g = gt[_labels == p, j]; g = g[~np.isnan(g)]
            if len(g) == 0: continue
            n_i[k] = len(g); p_i[k] = g.mean() / 2.0; h_i[k] = np.mean(g == 1)
        if (n_i >= 2).sum() < 2: continue
        n_bar = n_i.mean()
        if n_bar <= 1: continue
        p_bar = (n_i * p_i).sum() / n_i.sum()
        s2 = (n_i * (p_i - p_bar) ** 2).sum() / (n_bar * (r - 1))
        h_bar = (n_i * h_i).sum() / n_i.sum()
        n_c = (r * n_bar - (n_i ** 2).sum() / (r * n_bar)) / (r - 1)
        if n_c <= 0: continue
        a = (n_bar / n_c) * (s2 - (1 / (n_bar - 1)) * (p_bar * (1 - p_bar)
                              - (r - 1) / r * s2 - h_bar / 4))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2
                                     - (2 * n_bar - 1) / (4 * n_bar) * h_bar)
        c = h_bar / 2.0
        den = a + b + c
        if den > 1e-9: fst[j] = max(0.0, a / den)
    return fst

@st.cache_data(show_spinner=False)
def fst_pairwise(gt, _labels, use_wc84=False):
    labels = np.asarray(_labels).astype(str)
    pops = sorted(np.unique(labels)); K = len(pops)
    M = np.full((K, K), np.nan); np.fill_diagonal(M, 0.0)
    masks = {p: (labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            sub_mask = np.logical_or(masks[pops[i]], masks[pops[j]])
            sub = gt[sub_mask]; sub_lab = labels[sub_mask]
            if use_wc84:
                v = fst_snp_wc(sub, sub_lab)
                M[i, j] = M[j, i] = float(np.nanmean(v))
            else:
                g1 = gt[masks[pops[i]]]; g2 = gt[masks[pops[j]]]
                p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0
                n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
                dn = np.where(n1 + n2 > 0, n1 + n2, np.nan)
                pb = (p1 * n1 + p2 * n2) / dn
                hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / dn
                ht = 2 * pb * (1 - pb)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fj = np.clip((ht - hs) / ht, 0.0, None)
                M[i, j] = M[j, i] = float(np.nanmean(fj))
    return M, pops

@st.cache_data(show_spinner=False)
def windowed_fst(gt, snp_df, _labels, w=50, step=25, use_wc84=False):
    fst = fst_snp_wc(gt, _labels)
    snp = snp_df.copy().reset_index(drop=True)
    n = min(len(snp), len(fst))
    snp = snp.iloc[:n].reset_index(drop=True); fst = fst[:n]
    snp["FST"] = fst; res = []
    for c in snp["CHR"].unique():
        sub = snp[snp["CHR"] == c].sort_values("BP").reset_index(drop=True)
        if len(sub) < w: continue
        for i in range(0, len(sub) - w + 1, step):
            win = sub.iloc[i:i + w]
            if win["FST"].notna().sum() < 5: continue
            res.append({"CHR": str(c), "START": int(win["BP"].min()),
                        "END": int(win["BP"].max()),
                        "MID": int((win["BP"].min() + win["BP"].max()) / 2),
                        "N_SNP": len(win),
                        "MEAN_FST": float(win["FST"].mean()),
                        "MAX_FST": float(win["FST"].max()),
                        "TOP_SNP": win.iloc[win["FST"].argmax()]["SNP"]})
    return pd.DataFrame(res)

@st.cache_data(show_spinner=False)
def detect_outliers(win_df, z_thr=3.0):
    df = win_df.copy()
    mu, sd = df["MEAN_FST"].mean(), df["MEAN_FST"].std()
    df["Z_FST"] = (df["MEAN_FST"] - mu) / sd if sd > 1e-9 else 0.0
    df["OUTLIER"] = df["Z_FST"] >= z_thr
    return df.sort_values("MEAN_FST", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def tajimas_window(genos):
    ni, ns = genos.shape
    if ns < 3 or ni < 4: return np.nan, 0
    S = 0; piv = []
    for j in range(ns):
        col = genos[:, j]; col = col[~np.isnan(col)]
        if len(col) < 2: continue
        p = col.mean() / 2.0
        if 0 < p < 1: S += 1; piv.append(2 * p * (1 - p) * (len(col) / (len(col) - 1)))
    if S == 0 or not piv: return np.nan, int(S)
    pi = np.mean(piv); n = ni * 2
    a1 = np.sum(1.0 / np.arange(1, n))
    a2 = np.sum(1.0 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2)
    e1 = c1 / a1; e2 = c2 / (a1 ** 2 + a2)
    den = np.sqrt(e1 * S + e2 * S * (S - 1))
    if den <= 1e-12: return np.nan, int(S)
    return float((pi - S / a1) / den), int(S)

@st.cache_data(show_spinner=False)
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True)
    n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w: continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx) - w + 1, step):
            wi = idx[i:i + w]
            D, S = tajimas_window(gt[:, wi])
            if not np.isnan(D):
                res.append({"CHR": str(c),
                            "START": int(snp.iloc[wi]["BP"].min()),
                            "END": int(snp.iloc[wi]["BP"].max()),
                            "MID": int((snp.iloc[wi]["BP"].min() + snp.iloc[wi]["BP"].max()) / 2),
                            "N_SNP": len(wi), "S": S, "TAJIMA_D": D})
    return pd.DataFrame(res)

@st.cache_data(show_spinner=False)
def composite_score(gt, snp_df, _labels):
    fst = fst_snp_wc(gt, _labels)
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    maf_v = np.nanmean(gt, axis=0) / 2.0
    maf_v = np.minimum(maf_v, 1 - maf_v)
    df = snp_df.copy().reset_index(drop=True)
    n = min(len(df), len(fst), len(hom), len(maf_v))
    df = df.iloc[:n].reset_index(drop=True)
    df["FST"] = fst[:n]; df["HOM"] = hom[:n]; df["MAF"] = maf_v[:n]
    def norm(x):
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values)
    df["Z_HOM"] = norm(df["HOM"].values)
    df["CSS"] = np.abs(df["Z_FST"] * df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def detect_roh(gt, snp_json, min_snps=30, min_kb=500.0):
    snp = pd.read_json(io.StringIO(snp_json))
    n = min(gt.shape[1], len(snp))
    gt = gt[:, :n]; snp = snp.iloc[:n].reset_index(drop=True)
    ca = snp["CHR"].astype(str).values
    bp = pd.to_numeric(snp["BP"], errors="coerce").fillna(0).astype(np.int64).values
    rohs = []; fb = np.zeros(gt.shape[0]); tot = 0.0
    for c in pd.unique(ca):
        idxs = np.where(ca == c)[0]
        if len(idxs) < min_snps: continue
        idxs = idxs[np.argsort(bp[idxs])]
        sp = bp[idxs[-1]] - bp[idxs[0]]
        if sp < 1: continue
        tot += sp
        for i in range(gt.shape[0]):
            col = gt[i, idxs]; sk = None
            for k in range(len(idxs)):
                if col[k] == 0 or col[k] == 2:
                    if sk is None: sk = k
                elif sk is not None:
                    nr = k - sk
                    lkb = (bp[idxs[k - 1]] - bp[idxs[sk]]) / 1000.0
                    if nr >= min_snps and lkb >= min_kb:
                        rohs.append({"IID_idx": i, "CHR": c,
                                     "start_bp": int(bp[idxs[sk]]), "end_bp": int(bp[idxs[k - 1]]),
                                     "n_snp": int(nr), "length_kb": float(lkb)})
                        fb[i] += bp[idxs[k - 1]] - bp[idxs[sk]]
                    sk = None
            if sk is not None:
                nr = len(idxs) - sk
                lkb = (bp[idxs[-1]] - bp[idxs[sk]]) / 1000.0
                if nr >= min_snps and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": c,
                                 "start_bp": int(bp[idxs[sk]]), "end_bp": int(bp[idxs[-1]]),
                                 "n_snp": int(nr), "length_kb": float(lkb)})
                    fb[i] += bp[idxs[-1]] - bp[idxs[sk]]
    fr = fb / tot if tot > 0 else fb
    return (pd.DataFrame(rohs) if rohs else pd.DataFrame(
        columns=["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"])), fr

@st.cache_data(show_spinner=False)
def ne_historical(gt, snp_df, max_snp=2000):
    bp = snp_df["BP"].astype(float).values
    ca = snp_df["CHR"].astype(str).values
    n = gt.shape[1]
    if n > max_snp:
        idx = np.linspace(0, n - 1, max_snp).astype(int)
        gt = gt[:, idx]; bp = bp[idx]; ca = ca[idx]
    X = impute_mean(gt); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2l, dl = [], []; nn = X.shape[1]
    for i in range(nn - 1):
        for j in range(i + 1, min(i + 50, nn)):
            if ca[i] != ca[j]: continue
            d = abs(bp[j] - bp[i]) / 1e6
            if d < 0.001 or d > 50: continue
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2l.append(r * r); dl.append(d)
    if not dl: return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2 = np.array(r2l); d = np.array(dl)
    bins = np.logspace(np.log10(d.min() + 1e-6), np.log10(d.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([r2[(d >= bins[k]) & (d < bins[k + 1])].mean()
                    if ((d >= bins[k]) & (d < bins[k + 1])).sum() > 0 else np.nan
                    for k in range(len(bins) - 1)])
    m = np.isfinite(r2b)
    if m.sum() < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2_adj = np.clip(r2b - 1.0 / X.shape[0], 1e-6, None)
    ne = np.where(r2_adj[m] > 1e-4, 1.0 / (4 * (bc[m] * 0.01) * r2_adj[m]), np.nan)
    df = pd.DataFrame({"GenAgo": bc[m] * 100, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@st.cache_data(show_spinner=False)
def ld_pruning(gt, w=50, step=5, r2=0.2):
    n = gt.shape[1]
    if n < 3: return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool); X = impute_mean(gt); X = X - X.mean(0)
    i = 0
    while i < n:
        we = min(i + step, n)
        while we < n and (we - i) < w:
            if keep[we] and keep[i] and X[:, i].std() > 1e-8 and X[:, we].std() > 1e-8:
                if np.corrcoef(X[:, i], X[:, we])[0, 1] ** 2 > r2: keep[we] = False
            we += 1
        i += step
    return keep

@st.cache_data(show_spinner=False)
def ld_decay(gt, snp_df, max_kb=1000, max_snp=3000):
    n = gt.shape[1]
    if n < 10: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    df_snp = snp_df.copy()
    df_snp["CHR"] = df_snp["CHR"].astype(str)
    df_snp["BP"] = df_snp["BP"].astype(float)
    df_snp["CHR_num"] = df_snp["CHR"].map(lambda x: _chk(x)[1])
    df_snp = df_snp.sort_values(["CHR_num", "BP"]).reset_index(drop=True)
    if n > max_snp:
        idx = np.linspace(0, n - 1, max_snp).astype(int)
    else:
        idx = np.arange(n)
    gt = gt[:, idx]; df_snp = df_snp.iloc[idx].reset_index(drop=True)
    n = len(idx)
    X = impute_mean(gt); X = X - X.mean(0)
    stds = X.std(0); stds[stds == 0] = 1e-9; X = X / stds
    distances = []; r2_vals = []
    bp = df_snp["BP"].values; chrs = df_snp["CHR"].values
    max_bp = max_kb * 1000
    for i in range(n):
        j_end = min(i + 100, n)
        same_chr = (chrs[i] == chrs[i + 1:j_end])
        if not np.any(same_chr): continue
        dists = np.abs(bp[i] - bp[i + 1:j_end])
        valid = same_chr & (dists <= max_bp) & (dists > 0)
        if np.any(valid):
            r = np.dot(X[:, i], X[:, i + 1:j_end][:, valid]) / X.shape[0]
            distances.extend(dists[valid] / 1000.0); r2_vals.extend(r ** 2)
    if not distances: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    df = pd.DataFrame({"dist_kb": distances, "r2": r2_vals})
    bins = np.linspace(0, max_kb, 30)
    df["bin"] = pd.cut(df["dist_kb"], bins=bins, labels=False, include_lowest=True)
    return df.groupby("bin").agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean"),
                                  count=("r2", "count")).dropna().sort_values("dist_kb").reset_index(drop=True)

def plot_manhattan(df, yc="P", title="", thr=None):
    if yc == "P":
        df = df.copy(); df["_y"] = -np.log10(np.clip(df["P"], 1e-300, 1)); yc = "_y"
    df = df[np.isfinite(df[yc])].copy()
    df = df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k", "BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[yc], mode="markers",
                                 marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
                                 text=sub["SNP"], hoverinfo="text", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Chr", yaxis_title=yc, height=500)
    return fig

def plot_win_manh(df, vc="MEAN_FST", title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df = df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k", "MID"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub["MID"] / 1e6, y=sub[vc], mode="markers",
                                 marker=dict(size=7, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
                                 text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}"
                                       for c, m, v in zip(sub["CHR"], sub["MID"], sub[vc])],
                                 hoverinfo="text", showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std()
    fig.add_hline(y=mu + sigma * sd, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Pos (Mb)", yaxis_title=vc, height=520)
    return fig

def plot_ld(ld, b=20):
    if ld is None or ld.empty: return None
    d = ld.copy()
    if "bin" in d.columns: d = d.drop(columns=["bin"])
    d["bin"] = (d["dist_kb"] // b) * b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(),
                   x="bin", y="r2", title="LD decay", height=450)

def plot_missing(mi, ms):
    from plotly.subplots import make_subplots
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind", "Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False)
    return f

def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError: return beta[-1], None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-9
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1]))
        se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

def _vc_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K)
    d = np.maximum(d, 0); Yt = U.T @ Y; X0t = U.T @ X0
    def nll(dl):
        if dl <= 0: return 1e18
        w = 1.0 / np.sqrt(dl * d + 1.0)
        Yw = Yt * w; Xw = X0t * w[:, None]
        try: b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except: return 1e18
        s2 = float(np.sum((Yw - Xw @ b) ** 2) / n)
        return 1e18 if s2 <= 0 else -(-0.5 * (n * np.log(s2) + np.sum(np.log(dl * d + 1.0))))
    grid = np.logspace(-3, 3, 30)
    lls = np.array([nll(dl) for dl in grid]); ib = int(np.argmin(lls))
    try:
        return max(float(minimize_scalar(nll, bounds=(grid[max(0, ib - 1)],
                                                       grid[min(29, ib + 1)]),
                                          method="bounded").x), 1e-6)
    except: return max(float(grid[ib]), 1e-6)

@st.cache_data(show_spinner=False)
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError("<10 individus valides")
    if K is None or not isinstance(K, np.ndarray) or K.shape[0] != gt.shape[0]:
        K = np.eye(gt.shape[0])
    Y = pheno[valid]; gv = gt[valid]; Kv = K[np.ix_(valid, valid)]
    X0 = np.ones((len(Y), 1)) if covs is None or covs.size == 0 else \
         np.column_stack([np.ones(len(Y)), np.asarray(covs)[valid]])
    X0 = X0[:, np.std(X0, axis=0) > 1e-10]
    if is_bin:
        X0_final, Yt, U, w = X0, Y, None, None
    elif use_lmm:
        dl = _vc_delta(Y, X0, Kv)
        d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(dl * d + 1.0)
        Yt = (U.T @ Y) * w
        X0_final = (U.T @ X0) * w[:, None]
    else:
        X0_final, Yt, U, w = X0, Y, None, None
    rows = []
    for j in range(gv.shape[1]):
        g = gv[:, j].astype(float)
        if np.isnan(g).mean() > miss_max: continue
        g = np.where(np.isnan(g), np.nanmean(g), g)
        p = g.mean() / 2.0; mj = min(p, 1 - p)
        if mj < maf_min: continue
        if is_bin:
            X = np.column_stack([X0_final, g])
            beta_snp, se = _logistic_irls(X, Y)
            if se is None or se <= 0: continue
            tv = beta_snp / se
            pval = float(2 * stats.norm.sf(abs(tv)))
            rows.append({"SNP_idx": j, "MAF": mj, "BETA": beta_snp, "SE": se,
                         "T": float(tv), "P": pval})
        else:
            X = np.column_stack([X0_final, (U.T @ g) * w if use_lmm else g])
            try: b, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            except Exception: continue
            res = Yt - X @ b
            s2 = float(np.sum(res ** 2) / max(len(Y) - X.shape[1], 1))
            if s2 <= 0: continue
            try: se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
            except Exception: continue
            if se <= 0: continue
            tv = b[-1] / se
            pval = float(2 * t_dist.sf(abs(tv), df=max(len(Y) - X.shape[1], 1)))
            rows.append({"SNP_idx": j, "MAF": mj, "BETA": b[-1], "SE": se,
                         "T": float(tv), "P": pval})
    return pd.DataFrame(rows)

def annotate_gwas(r, snp_df):
    if r.empty: return r
    idx = r["SNP_idx"].astype(int).values; r = r.copy()
    for c in ["SNP", "CHR", "BP", "CM"]:
        dt = str if c in ("SNP", "CHR") else float if c == "CM" else int
        r[c] = snp_df[c].astype(dt).values[idx]
    return r[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1))

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty: return r
    r = r.copy()
    if HAS_STATSMODELS:
        try:
            _, fdr, _, _ = multipletests(r["P"].values, method="fdr_bh")
            r["FDR"] = fdr
        except Exception: r["FDR"] = np.nan
    else: r["FDR"] = np.nan
    r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300, 1), df=1) / lam, df=1)
            r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r):
    if r.empty: return r
    df = r.copy().sort_values("P"); v = 0.04
    df["ABF"] = np.exp(0.5 * (df["BETA"] ** 2 / (df["SE"] ** 2 + v))
                       - 0.5 * np.log(1 + v / df["SE"] ** 2))
    df["PIP"] = df["ABF"] / df["ABF"].sum()
    return df.sort_values("PIP", ascending=False)

def run_gblup(gt, pheno, kf=5):
    valid = np.isfinite(pheno)
    if valid.sum() < 20: raise ValueError("<20 phénotypes")
    kf = min(kf, len(pheno) - 1)
    Y = pheno[valid]; X = impute_mean(gt[valid])
    p = X.mean(0) / 2.0; Z = X - 2 * p
    G = (Z @ Z.T) / (2 * np.sum(p * (1 - p))) + np.eye(len(Y)) * 1e-5
    I = np.eye(len(Y))
    bh, bl = 0.5, -np.inf
    for h2 in np.linspace(0.01, 0.99, 20):
        lam = (1 - h2) / h2
        try:
            ll = -0.5 * (np.linalg.slogdet(G + lam * I)[1]
                         + len(Y) * np.log(Y.T @ np.linalg.inv(G + lam * I) @ Y))
            if ll > bl: bl, bh = ll, h2
        except Exception: pass
    lam = (1 - bh) / bh
    gebv = G @ np.linalg.inv(G + lam * I) @ Y
    idx = np.arange(len(Y)); np.random.shuffle(idx)
    folds = np.array_split(idx, kf); yp = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(idx, te)
        Gtt = G[np.ix_(tr, tr)]; Gvt = G[np.ix_(te, tr)]
        yp[te] = Gvt @ np.linalg.inv(Gtt + lam * np.eye(len(tr))) @ Y[tr]
    corr = np.corrcoef(Y, yp)[0, 1] if len(Y) > 2 else 0.0
    return pd.DataFrame({"IID": np.where(valid)[0], "Phenotype": Y, "GEBV": gebv}), \
           {"h2": bh, "corr": corr, "r2": corr ** 2,
            "rmse": float(np.sqrt(np.mean((Y - yp) ** 2)))}

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP")
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1)
            p = np.poly1d(z)
            xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines",
                                     name="OLS", line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig

def plot_pip(df):
    return px.bar(df.head(20), x="SNP", y="PIP", title="PIP",
                  color="PIP", color_continuous_scale="Viridis")

_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if "missense" in c: return "🔴 Exon (Faux-sens)"
    if "synonymous" in c: return "🟡 Exon (Silencieuse)"
    if "stop_gained" in c or "frameshift" in c: return "⛔ Exon (Non-sens)"
    if "splice" in c: return "⚠️ Jonction"
    if "intron" in c: return "🟢 Intron"
    if "utr" in c: return "🔵 UTR"
    if "upstream" in c or "downstream" in c: return "🟣 Régulateur"
    return "⚪ Intergénique"

def run_vep_api(snp_list, species, batch=200):
    if not snp_list: return pd.DataFrame()
    url = f"https://rest.ensembl.org/vep/{species}/region?regulatory=1"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []
    for i in range(0, len(snp_list), batch):
        chunk = snp_list[i:i + batch]
        try:
            r = requests.post(url, headers=headers,
                              data=json.dumps({"variants": chunk}), timeout=60)
            if r.status_code == 200:
                for item in r.json():
                    sid = item.get("input", "N/A")
                    if item.get("transcript_consequences"):
                        cons = sorted(item["transcript_consequences"],
                                      key=lambda x: _SEVERITY.get(x.get("impact", "MODIFIER"), 0),
                                      reverse=True)[0]
                        results.append({"SNP": sid, "GENE": cons.get("gene_symbol", "N/A"),
                                        "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])),
                                        "IMPACT": cons.get("impact", "N/A"),
                                        "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A")})
                    else:
                        results.append({"SNP": sid, "GENE": "Intergenic/Non-codant",
                                        "CONSEQUENCE": "N/A", "IMPACT": "MODIFIER",
                                        "AMINO_ACID_CHANGE": "N/A"})
        except Exception as e:
            st.warning(f"⚠️ VEP batch {i // batch + 1}: {e}")
        time.sleep(0.5)
    return pd.DataFrame(results)

def fetch_ncbi_gene_info(gene_list, taxon):
    if not gene_list: return pd.DataFrame()
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{quote(str(gene))}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json(); reports = data.get("reports") or [{}]
                gi = reports[0].get("gene", {}) if reports else {}
                results.append({"GENE": gene, "Nom complet": gi.get("description", "N/A"),
                                "Chromosome": (gi.get("chromosomes") or ["N/A"])[0],
                                "Résumé": (gi.get("summary", "N/A")[:200] + "...")
                                         if gi.get("summary") else "N/A"})
            else:
                results.append({"GENE": gene, "Nom complet": "Non trouvé",
                                "Chromosome": "N/A", "Résumé": "N/A"})
        except Exception as e:
            results.append({"GENE": gene, "Nom complet": "Erreur API",
                            "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_uniprot_id(gene_name, taxon):
    q = f"gene:{gene_name} AND organism_id:{taxon}"
    url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                return data["results"][0].get("primaryAccession")
    except Exception: pass
    return None

def fetch_alphafold_pdb(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            for rec in r.json():
                if rec.get("uniprotAccession") == uniprot_id:
                    pdb_url = rec.get("pdbUrl")
                    if pdb_url:
                        pr = requests.get(pdb_url, timeout=30)
                        if pr.status_code == 200: return pr.text
    except Exception: pass
    return None

def render_3d_structure(pdb_data, style="cartoon", color_scheme="spectrum", highlight_res=None):
    style_json = json.dumps({style: {"color": color_scheme}})
    pdb_json = json.dumps(pdb_data); hl_js = ""
    if highlight_res and str(highlight_res).strip().isdigit():
        hl_js = f"viewer.addStyle({{resi: '{highlight_res}'}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
    html_code = f"""
    <div id="viewport" style="width: 100%; height: 500px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{backgroundColor: 'white'}});
        viewer.addModel({pdb_json}, "pdb");
        viewer.setStyle({{}}, {style_json});
        {hl_js}
        viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html_code, height=520)

def fetch_ensembl_sequence(chrom, start, end, species):
    url = f"https://rest.ensembl.org/sequence/region/{species}/{chrom}:{start}..{end}:1?content-type=text/plain"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200: return r.text.upper()
    except Exception: pass
    return None

def align_and_highlight(seq_ref, seq_fast, seq_slow):
    if not HAS_BIO: return "<p>⚠️ Biopython requis.</p>"
    aligner = Align.PairwiseAligner(); aligner.mode = "global"
    af = aligner.align(seq_ref, seq_fast)[0]
    asl = aligner.align(seq_ref, seq_slow)[0]
    html = "<div style='font-family: monospace; font-size: 16px;'>"
    def fmt(aln, name, color):
        ref_aln, seq_aln = aln[0], aln[1]
        s = f"<b>{name:10}</b> | "
        for r, q in zip(ref_aln, seq_aln):
            if r == "-" or q == "-":
                s += f"<span style='color: orange; font-weight: bold;'>{q}</span>"
            elif r != q:
                s += f"<span style='color: red; font-weight: bold;'>{q}</span>"
            else:
                s += f"<span style='color: {color};'>{q}</span>"
        return s + "<br>"
    html += fmt(af, "Rapide", "green")
    html += f"<b>{'Référence':10}</b> | " + "".join(
        f"<span style='color: grey;'>{r}</span>" for r in af[0]) + "<br>"
    html += fmt(asl, "Lente", "blue")
    return html + "</div>"

def lift_over(chrom, bp, species_to="human"):
    url = f"https://rest.ensembl.org/map/{species_to}/GRCh38/{chrom}:{bp}-{bp}:1?content-type=application/json"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            m = r.json().get("mappings", [])
            if m:
                mp = m[0].get("mapped", {})
                return mp.get("seq_region_name"), mp.get("start")
    except Exception: pass
    return None, None

def parse_gff3(b):
    from core import _rt as _rt_core
    genes = []
    for line in _rt_core(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split("\t")
        if len(p) >= 9 and p[2] == "gene":
            a = p[8]; name = "Unknown"
            if "Name=" in a: name = a.split("Name=")[1].split(";")[0]
            elif 'gene_name "' in a: name = a.split('gene_name "')[1].split('"')[0]
            elif 'gene_id "' in a: name = a.split('gene_id "')[1].split('"')[0]
            genes.append({"CHR": str(p[0]), "START": int(p[3]),
                          "END": int(p[4]), "GENE": name})
    return pd.DataFrame(genes).drop_duplicates(subset=["CHR", "START", "END", "GENE"])

def annotate_genes(snp_df, gff, wk=50):
    w = wk * 1000; res = []
    for _, s in snp_df.iterrows():
        pos = s.get("BP", s.get("START", 0)); ch = str(s["CHR"])
        msk = (gff["CHR"] == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)
        ng = gff[msk]
        for _, g in ng.iterrows():
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch,
                        "SNP_POS": pos, "GENE": g["GENE"],
                        "GENE_START": g["START"], "GENE_END": g["END"],
                        "DIST_KB": round(abs(g["START"] - pos) / 1000.0, 2)})
    return pd.DataFrame(res)

def query_string_network(genes, species, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([str(g).strip() for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status()
        data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(data)
        edges = df[["preferredName_A", "preferredName_B", "score"]].copy()
        edges.columns = ["SOURCE", "TARGET", "SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
        return edges, nodes
    except Exception: return pd.DataFrame(), pd.DataFrame()

def plot_string_network(edges, nodes):
    if edges is None or edges.empty: return None
    try: import networkx as nx
    except ImportError:
        st.error("⚠️ networkx non installé.")
        return None
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    ex, ey = [], []
    for s, t in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[t]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=0.6, color="#cccccc"),
                             hoverinfo="skip", showlegend=False))
    nx_, ny_ = [], []
    for n in G.nodes():
        x, y = pos[n]; nx_.append(x); ny_.append(y)
    deg = [d for _, d in G.degree()]
    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text",
                             marker=dict(size=[8 + 3 * d for d in deg], color=deg,
                                         colorscale="Viridis",
                                         line=dict(width=1, color="white")),
                             text=list(G.nodes()), textposition="top center",
                             hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list
                      if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None:
            gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets,
                         organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        rename = {"Term": "TERM", "Adjusted P-value": "FDR",
                  "P-value": "PVAL", "Overlap": "OVERLAP",
                  "Genes": "GENES", "Combined Score": "SCORE"}
        for old, new in rename.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Enrichr : {e}")
        return pd.DataFrame()
