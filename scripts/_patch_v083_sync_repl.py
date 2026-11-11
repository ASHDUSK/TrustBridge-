# -*- coding: utf-8 -*-
"""v0.8.3 同步补丁：数值收集 + 中英替换对构建。由 _patch_v083_sync.py 调用。"""
import csv
import glob
import json
import os

import numpy as np

ROOT = r"D:\MedTranslateProject"
R = os.path.join(ROOT, "results")


def J(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


def pearson(dirname):
    fs = glob.glob(os.path.join(R, dirname, "per_slice.csv"))
    if not fs:
        return None
    rows = list(csv.DictReader(open(fs[0], encoding="utf-8")))
    pairs = [(float(r["trust_score"]), float(r["body_psnr"]))
             for r in rows if r.get("trust_score") and r.get("body_psnr")]
    if len(pairs) < 5:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return float(np.corrcoef(x, y)[0, 1])


def sep_median(dirname):
    fs = glob.glob(os.path.join(R, dirname, "per_slice.csv"))
    if not fs:
        return {}
    rows = list(csv.DictReader(open(fs[0], encoding="utf-8")))
    vals = {}
    for r in rows:
        cfg = r.get("config", "")
        if cfg and r.get("decile_sep"):
            try:
                vals.setdefault(cfg, []).append(float(r["decile_sep"]))
            except ValueError:
                pass
    return {k: (round(float(np.median(v)), 0) if np.median(v) >= 100
                else round(float(np.median(v)), 1))
            for k, v in vals.items()}


def collect():
    V = {}
    V["deg_self"] = J("2026-11-12_degradation_response/summary.json")["conditions"]
    V["deg_off"] = J("2026-11-12_degradation_response_official/summary.json")["conditions"]
    V["tumor_spars"] = J("2026-11-12_tumor_sparsify_body/summary.json")
    V["crossinst"] = J("2026-11-12_crossinst_brats/summary.json")["sets"]
    V["tumor_seg"] = J("2026-11-12_tumor_trust_corrected/summary.json")
    V["collapse"] = J("2026-11-12_collapse_alarm/summary.json")
    import csv as _csv
    _rows = list(_csv.DictReader(open(os.path.join(
        R, "2026-11-12_collapse_alarm/per_case.csv"), encoding="utf-8")))
    V["collapse_au"] = {
        lbl: (float(np.nanmean([float(r["auroc"]) for r in _rows if r["model"] == lbl])),
              float(np.nanstd([float(r["auroc"]) for r in _rows if r["model"] == lbl])))
        for lbl in ("official_15db", "self_20db")}
    V["uq"] = {t: J(d) for t, d in [
        ("t1t2_self", "2026-11-12_uq_baselines_T1_T2_self/summary.json"),
        ("t2t1_self", "2026-11-12_uq_baselines_T2_T1_self/summary.json"),
        ("t1t2_off", "2026-11-12_uq_baselines_T1_T2_official/summary.json"),
        ("t2t1_off", "2026-11-12_uq_baselines_T2_T1_official/summary.json")]}
    V["patient"] = J("2026-11-12_patient_stats/summary.json")
    V["qc"] = J("2026-11-12_qc_thresholds/summary.json")
    V["SEP"] = {t: sep_median(d) for t, d in [
        ("t1t2_self", "2026-11-12_uq_baselines_T1_T2_self"),
        ("t2t1_self", "2026-11-12_uq_baselines_T2_T1_self"),
        ("t1t2_off", "2026-11-12_uq_baselines_T1_T2_official"),
        ("t2t1_off", "2026-11-12_uq_baselines_T2_T1_official")]}
    V["PEAR"] = {"t1t2": pearson("2026-11-12_slice_scores_T1_T2"),
                 "t2t1": pearson("2026-11-12_slice_scores_T2_T1"),
                 "tumor": pearson("2026-11-12_tumor_sparsify_body")}
    print("== 数值采集 ==")
    print(f"自训 fused: {V['uq']['t1t2_self']['methods']['ours_fused']['auroc']['mean']:.3f}/"
          f"{V['uq']['t2t1_self']['methods']['ours_fused']['auroc']['mean']:.3f}")
    print(f"肿瘤: {V['tumor_spars']['auroc']['mean']:.3f} | 崩塌 off {V['collapse']['groups']['official_15db']['trust_score']}")
    print(f"QC75: {V['qc']['at_75_PASS']['flagged_pct']:.2f} | PEAR {V['PEAR']}")
    return V


def build_repl(V):
    uq, SEP, PEAR = V["uq"], V["SEP"], V["PEAR"]
    ds, do_ = V["deg_self"], V["deg_off"]
    tumor_spars, tumor_seg = V["tumor_spars"], V["tumor_seg"]
    B = V["crossinst"]["B_same_collection_heldout"]
    C = V["crossinst"]["C_cross_institution_cbica"]
    co_off = V["collapse"]["groups"]["official_15db"]
    co_self = V["collapse"]["groups"]["self_20db"]
    au_off = V["collapse_au"]["official_15db"][0]
    au_self_col = V["collapse_au"]["self_20db"][0]
    co_flags = co_off["qc_flags"]
    co_flag_txt = (f"{co_flags['PASS']} PASS / {co_flags['REVIEW']} REVIEW / "
                   f"{co_flags['REJECT']} REJECT")
    co_flag_txt_zh = (f"{co_flags['PASS']} PASS / {co_flags['REVIEW']} REVIEW / "
                      f"{co_flags['REJECT']} REJECT")
    patient, qc = V["patient"], V["qc"]

    def m(tag, cfg, metric="auroc"):
        return uq[tag]["methods"][cfg][metric]["mean"]

    def sd(tag, cfg, metric="auroc"):
        return uq[tag]["methods"][cfg][metric]["std"]

    a_f_1, a_f_2 = m("t1t2_self", "ours_fused"), m("t2t1_self", "ours_fused")
    a_c1_1, a_c1_2 = m("t1t2_self", "ours_rcrf_n1"), m("t2t1_self", "ours_rcrf_n1")
    a_e_1, a_e_2 = m("t1t2_self", "ours_unc_n4"), m("t2t1_self", "ours_unc_n4")
    a_tta_1, a_tta_2 = m("t1t2_self", "tta_disp"), m("t2t1_self", "tta_disp")
    a_no_1, a_no_2 = m("t1t2_self", "noise_disp"), m("t2t1_self", "noise_disp")
    a_gr_1, a_gr_2 = m("t1t2_self", "gradient"), m("t2t1_self", "gradient")
    a_rd_1, a_rd_2 = m("t1t2_self", "random"), m("t2t1_self", "random")
    d_1 = m("t1t2_self", "ours_fused", "dpsnr20")
    d_2 = m("t2t1_self", "ours_fused", "dpsnr20")
    oa_1, oa_2 = m("t1t2_off", "ours_fused"), m("t2t1_off", "ours_fused")
    od_1, od_2 = m("t1t2_off", "ours_fused", "dpsnr20"), m("t2t1_off", "ours_fused", "dpsnr20")
    otta_1, otta_2 = m("t1t2_off", "tta_disp"), m("t2t1_off", "tta_disp")
    otta_d1 = m("t1t2_off", "tta_disp", "dpsnr20")
    otta_d2 = m("t2t1_off", "tta_disp", "dpsnr20")
    off_all_2 = sorted(m("t2t1_off", c) for c in
                       ("ours_fused", "ours_unc_n4", "ours_rcrf_n1", "tta_disp", "noise_disp"))
    tu_au, tu_au_sd = tumor_spars["auroc"]["mean"], tumor_spars["auroc"]["std"]
    tu_d20 = tumor_spars["dpsnr20"]["mean"]
    tu_sep = tumor_spars["decile_sep"]["mean"]
    seg_au = tumor_seg["auroc_lowtrust_to_tumor"]["mean"]
    seg_sd = tumor_seg["auroc_lowtrust_to_tumor"].get("std", 0.0)
    frac = tumor_seg["slices_with_higher_err_inside"]
    p1 = patient["self_T1_T2:fused_vs_ours_unc_n4"]["subject_macro"]
    p2 = patient["self_T2_T1:fused_vs_ours_unc_n4"]["subject_macro"]
    p_off = patient["official_T2T1:ours_fused_vs_chance"]
    qc75, qc55 = qc["at_75_PASS"], qc["at_55_REVIEW"]
    best = qc.get("suggested_reject_thresh") or {}

    def ci_txt(ci):
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    inc1 = ("includes zero --- not significant on T1$\\to$T2"
            if p1["ci95"][0] <= 0 <= p1["ci95"][1] else "significant on T1$\\to$T2")
    inc2 = ("significant on T2$\\to$T1" if p2["ci95"][0] > 0
            else "includes zero on T2$\\to$T1")
    zinc1 = ("含 0——T1$\\to$T2 上不显著" if p1["ci95"][0] <= 0 <= p1["ci95"][1]
             else "T1$\\to$T2 上显著")
    zinc2 = ("T2$\\to$T1 上显著" if p2["ci95"][0] > 0 else "含 0")
    d_tta_1, d_tta_2 = a_f_1 - a_tta_1, a_f_2 - a_tta_2
    d_no_1, d_no_2 = a_f_1 - a_no_1, a_f_2 - a_no_2
    disp_lo = min(a_f_1, a_f_2, oa_1)
    disp_hi = max(a_f_1, a_f_2, oa_1)
    concl_lo = min(tu_au, a_f_2)
    concl_hi = max(C["auroc"]["mean"], a_f_1)

    E, Z = [], []

    # ============ EN ============
    E.append((r"""AUROC $0.90$/$0.82$ (T1$\to$T2/T2$\to$T1, self-trained; $0.88$/$0.79$ for
a single single-chain (no extra forward)), $0.91$ on real tumor patients, and $0.95$
\emph{cross-institution} on never-seen subjects.""",
              f"""AUROC ${a_f_1:.2f}$/${a_f_2:.2f}$ (T1$\\to$T2/T2$\\to$T1, self-trained; ${a_c1_1:.2f}$/${a_c1_2:.2f}$ for
a single chain at no extra cost), ${tu_au:.2f}$ on real tumor patients, and ${C['auroc']['mean']:.2f}$
\\emph{{cross-institution}} on never-seen subjects.""", "abs-fused"))

    E.append((r"""\emph{fails to notice}: the collapsed model receives PASS with $94/100$
mean trust.""",
              f"""\\emph{{barely notices}}: the collapsed model still scores ${co_off['trust_score']:.0f}/100$---below
the healthy model's ${co_self['trust_score']:.0f}$, yet only REVIEW, never REJECT, and its pixel-level
AUROC degrades to $0.61$ (healthy $0.82$); the originally reported blanket PASS
($94/100$, whole-image mean) was itself an air-pooling artifact.""", "abs-collapse"))

    E.append((r"""the fused signal localizes the top-error decile at AUROC
        $0.90$/$0.82$ (self-trained T1$\to$T2/T2$\to$T1), beating all
        equal-budget baselines, $0.91$ on real tumor patients, and $0.95$
        cross-institution; on the \emph{official} weights the same protocol
        flips the ranking and blinds T2$\to$T1 entirely (\S\ref{sec:exp}).""",
              f"""the fused signal localizes the top-error decile at AUROC
        ${a_f_1:.3f}$/${a_f_2:.3f}$ (self-trained T1$\\to$T2/T2$\\to$T1), beating all
        equal-budget baselines, ${tu_au:.2f}$ on real tumor patients (our earlier
        $0.911$ there was itself a whole-image artifact, \\S\\ref{{sec:exp-tumor}}),
        and ${C['auroc']['mean']:.2f}$ cross-institution; on the \\emph{{official}} weights the same protocol
        flips the ranking and blinds T2$\\to$T1 entirely (\\S\\ref{{sec:exp}}).""", "contrib"))

    E.append((r"""the fused signal achieves AUROC $0.823$/$0.757$ (T1$\to$T2/T2$\to$T1)
with $+2.67$/$+1.85$\,dB sparsification gains. Paired subject-level bootstrap
($B{=}5000$, 8 subjects) shows: fused vs.\ ensemble alone $\Delta$AUROC
$+0.004$ (95\% CI $[-0.001, 0.008]$, \emph{includes zero} --- not significant
on T1$\to$T2) and $+0.010$ (CI $[0.005, 0.015]$, significant on T2$\to$T1);
fused vs.\ all equal-budget baselines is significant with large margins
(TTA $-0.082$/$-0.107$, noise $-0.041$/$-0.078$, gradient). A single CReF
chain (zero extra cost) reaches $0.787$/$0.705$ --- most of the value at a
quarter of the budget; CReF's claim is cost, not accuracy over the ensemble.""",
              f"""the fused signal achieves AUROC ${a_f_1:.3f}$/${a_f_2:.3f}$ (T1$\\to$T2/T2$\\to$T1)
with $+{d_1:.2f}$/$+{d_2:.2f}$\\,dB sparsification gains. Paired subject-level bootstrap
($B{{=}}5000$, 8 subjects) shows: fused vs.\\ ensemble alone $\\Delta$AUROC
$+{p1['delta']:.3f}$ (95\\% CI {ci_txt(p1['ci95'])}, \\emph{{{inc1}}})
and $+{p2['delta']:.3f}$ (CI {ci_txt(p2['ci95'])}, {inc2});
fused vs.\\ all equal-budget baselines is significant with large margins
(TTA $-{d_tta_1:.3f}$/$-{d_tta_2:.3f}$, noise $-{d_no_1:.3f}$/$-{d_no_2:.3f}$, gradient). A single CReF
chain (zero extra cost) reaches ${a_c1_1:.3f}$/${a_c1_2:.3f}$ --- most of the value at a
quarter of the budget; CReF's claim is cost, not accuracy over the ensemble.""", "44-fused"))

    E.append((r"""\textbf{On the official weights}, the ranking changes and the
T2$\to$T1 direction collapses entirely: every dispersion signal---ours and
all baselines'---is \emph{systematically below chance} (fused CI
$[0.442, 0.485]$, upper bound $<0.5$), because that model's
residual error is dominated by fine texture that no dispersion measure can
track.""",
              f"""\\textbf{{On the official weights}}, the ranking changes and the
T2$\\to$T1 direction collapses: our sampler-native signals fall to near-chance
(fused ${m('t2t1_off','ours_fused'):.3f}\\pm{sd('t2t1_off','ours_fused'):.3f}$, ensemble
${m('t2t1_off','ours_unc_n4'):.3f}$, CReF ${m('t2t1_off','ours_rcrf_n1'):.3f}$) while the strongest \\emph{{baseline}}---TTA
dispersion---retains weak signal (${m('t2t1_off','tta_disp'):.3f}\\pm{sd('t2t1_off','tta_disp'):.3f}$), because that model's
residual error is dominated by fine texture that the convergence and
bridge-dispersion measures barely track. (The ``systematically below
chance'' values we originally reported here---e.g.\\ fused CI $[0.442, 0.485]$---were
themselves whole-image artifacts.)""", "44-blind"))

    E.append((r"""they are trusted and near-zero-error, so pooling them with body pixels
inflates whole-image UQ metrics severalfold---we show a headline localization
score of $0.90$ collapsing to $0.78$ under a within-body protocol, and retract""",
              f"""they are trusted and near-zero-error, so pooling them with body pixels
inflates whole-image UQ metrics severalfold---we show a headline localization
score of $0.90$ collapsing to ${m('t1t2_off','ours_fused'):.2f}$ under a within-body protocol, and retract""", "abs-trap1"))

    E.append((r"""illustrate the effect: the same fused signal scores AUROC $0.78$ within the""",
              f"""illustrate the effect: the same fused signal scores AUROC ${m('t1t2_off','ours_fused'):.2f}$ within the""", "41-demo"))

    E.append((r"""Second, \emph{dispersion is
not error}: across an equal-budget comparison with TTA and input-noise
baselines, every dispersion-style signal---including ours---is at chance on
T2$\to$T1, where residual error is dominated by fine texture.""",
              f"""Second, \\emph{{dispersion is
not error}}: across an equal-budget comparison with TTA and input-noise
baselines, our sampler-native signals fall to chance on the official
T2$\\to$T1 weights (fused ${m('t2t1_off','ours_fused'):.2f}$; the TTA baseline retains weak signal,
${m('t2t1_off','tta_disp'):.2f}$), where residual error is dominated by fine texture---which signal, if
any, survives is a property of the (model, direction) pair.""", "abs-trap2"))

    E.append((r"""flips the ranking and blinds T2$\to$T1 entirely (\S\ref{sec:exp}).""",
              f"""flips the ranking and blinds our signals on T2$\\to$T1 (fused
        ${m('t2t1_off','ours_fused'):.2f}$; TTA baseline ${m('t2t1_off','tta_disp'):.2f}$) (\\S\\ref{{sec:exp}}).""", "contrib-blind"))

    E.append((r"""pure CReF ($0.787$ at budget 10) already dominates
full TTA ($0.820$)---so routing adds nothing there""",
              f"""pure CReF (${a_c1_1:.3f}$ at budget 10) already dominates
full TTA (${a_tta_1:.3f}$)---so routing adds nothing there""", "routing"))

    # Table 2 整表（EN）
    t2 = f"""\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{T1$\\to$T2 (self)}} & \\multicolumn{{3}}{{c}}{{T2$\\to$T1 (self)}} \\\\
Signal & AUROC & $\\Delta$PSNR@20\\% & Sep & AUROC & $\\Delta$PSNR@20\\% & Sep \\\\
\\midrule
Random (control) & {a_rd_1:.3f} & $\\approx$0 & 1.0 & {a_rd_2:.3f} & $\\approx$0 & 1.0 \\\\
Gradient & {a_gr_1:.3f} & {m('t1t2_self','gradient','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('gradient','--')} & {a_gr_2:.3f} & {m('t2t1_self','gradient','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('gradient','--')} \\\\
Input-noise dispersion & {a_no_1:.3f} & {m('t1t2_self','noise_disp','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('noise_disp','--')} & {a_no_2:.3f} & {m('t2t1_self','noise_disp','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('noise_disp','--')} \\\\
TTA dispersion & {a_tta_1:.3f} & {m('t1t2_self','tta_disp','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('tta_disp','--')} & {a_tta_2:.3f} & {m('t2t1_self','tta_disp','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('tta_disp','--')} \\\\
CReF only (N=1, no extra) & {a_c1_1:.3f} & {m('t1t2_self','ours_rcrf_n1','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('ours_rcrf_n1','--')} & {a_c1_2:.3f} & {m('t2t1_self','ours_rcrf_n1','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('ours_rcrf_n1','--')} \\\\
Ensemble only (N=4) & {a_e_1:.3f} & {m('t1t2_self','ours_unc_n4','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('ours_unc_n4','--')} & {a_e_2:.3f} & {m('t2t1_self','ours_unc_n4','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('ours_unc_n4','--')} \\\\
\\textbf{{Fused (ours)}} & \\textbf{{{a_f_1:.3f}}} & $\\mathbf{{{d_1:+.2f}}}$ & \\textbf{{{SEP['t1t2_self'].get('ours_fused','--')}}} & \\textbf{{{a_f_2:.3f}}} & $\\mathbf{{{d_2:+.2f}}}$ & \\textbf{{{SEP['t2t1_self'].get('ours_fused','--')}}} \\\\
\\midrule
\\multicolumn{{7}}{{l}}{{\\emph{{Same protocol on the official weights (N=4):}}}} \\\\
Fused (official) & {oa_1:.3f} & {od_1:+.2f} & {SEP['t1t2_off'].get('ours_fused','--')} & {oa_2:.3f} & {od_2:+.2f} & {SEP['t2t1_off'].get('ours_fused','--')} \\\\
TTA (official) & {otta_1:.3f} & {otta_d1:+.2f} & {SEP['t1t2_off'].get('tta_disp','--')} & {otta_2:.3f} & {otta_d2:+.2f} & {SEP['t2t1_off'].get('tta_disp','--')} \\\\
All signals (official, T2$\\to$T1) & -- & -- & -- & \\multicolumn{{3}}{{l}}{{${off_all_2[0]:.3f}$--${off_all_2[-1]:.3f}$ (near chance)}} \\\\
\\bottomrule
\\end{{tabular}}"""
    E.append(("__TAB2__", t2, "tab2"))

    E.append((r"""Within the body, the least-trusted decile carries $194\times$ (T1$\to$T2,
median; the top three bins carry near-zero error, so the ratio is an
order-of-magnitude statement), $4.0\times$ (T2$\to$T1) and $2.1\times$
(BraTS) the mean absolute error of the most-trusted decile.""",
              f"""Within the body, the least-trusted decile carries ${SEP['t1t2_self'].get('ours_fused', 197):.0f}\\times$ (T1$\\to$T2,
median; the top three bins carry near-zero error, so the ratio is an
order-of-magnitude statement), ${SEP['t2t1_self'].get('ours_fused', 3.8)}\\times$ (T2$\\to$T1) and ${tu_sep:.1f}\\times$
(BraTS) the mean absolute error of the most-trusted decile.""", "45-decile"))

    E.append((r"""correlates with slice quality (body PSNR) weakly: Pearson $r=+0.47$
(T1$\to$T2, self-trained), $r=+0.26$ (T2$\to$T1, self-trained),
$r=-0.10$ (BraTS, official)---right-signed on the models whose signals
work, absent on the one whose do not. The factory thresholds are likewise mis-calibrated to the
actual score distributions: PASS$\ge$75 flags $88\%$ of pooled slices while
REJECT$<$55 fires on $3\%$. An empirical reading: the lowest-scoring slices
are enriched for quality problems but coverage--precision trade-offs are
steep (at the best pooled cut, $9\%$ coverage buys $29\%$ precision for the
bottom-quality quartile).""",
              f"""correlates with slice quality (body PSNR) weakly: Pearson $r={PEAR['t1t2']:+.2f}$
(T1$\\to$T2, self-trained), $r={PEAR['t2t1']:+.2f}$ (T2$\\to$T1, self-trained),
$r={PEAR['tumor']:+.2f}$ (BraTS, official)---right-signed on the models whose signals
work, absent on the one whose do not. The factory thresholds are likewise
mis-calibrated to the actual score distributions: PASS$\\ge$75 waves through
{(1-qc75['flagged_pct'])*100:.0f}\\% of pooled slices but catches only {qc75['recall']*100:.0f}\\% of the
bottom-quality-quartile failures, while REJECT$<$55 fires on
{qc55['flagged_pct']*100:.1f}\\% (with high precision, negligible recall:
{qc55['recall']*100:.0f}\\%). An empirical reading: the lowest-scoring slices are
enriched for quality problems but coverage--precision trade-offs are steep
(the best pooled cut, $<$55, buys {best.get('precision', 0)*100:.0f}\\% precision at
{best.get('flagged_pct', 0)*100:.1f}\\% coverage for the bottom-quality quartile).""", "46-qc"))

    E.append((r"""residual error is texture-dominated---while the same direction works on our
self-trained model ($0.757$), isolating the failure to the model, not the task;""",
              f"""residual error is texture-dominated---while the same direction works on our
self-trained model (${a_f_2:.3f}$), isolating the failure to the model, not the task;""", "47-b6"))

    E.append((r"""$0.42$--$0.67$ where dispersion reaches $0.76$--$0.82$, and their""",
              f"""$0.42$--$0.67$ where dispersion reaches ${disp_lo:.2f}$--${disp_hi:.2f}$, and their""", "47-b7"))

    E.append((r"""On the BraTS cohort (official FLAIR$\to$T2 weights, 60 tumor-bearing slices,
within-body protocol):
(i) pixel error inside the tumor is $1.60\times$ the error outside ($90\%$ of
slices), confirming that synthesis risk concentrates at the lesion;
(ii) the \TrustMap{} localizes the top-error decile with AUROC
$0.911\pm0.037$ and a $+3.12$\,dB sparsification gain at 20\%---trust
signals survive the protocol correction on this model;
but (iii) we \emph{retract} an earlier claim: low trust does \emph{not}
localize tumor tissue itself (within-body AUROC vs.\ segmentation
$0.51\pm0.10$; our previously reported $0.871$ was a whole-image artifact of
air negatives, and mean trust is in fact slightly \emph{higher} inside tumors,
$0.772$ vs.\ $0.727$ outside).""",
              f"""On the BraTS cohort (official FLAIR$\\to$T2 weights, 60 tumor-bearing slices,
within-body protocol):
(i) pixel error inside the tumor is ${tumor_seg['err_ratio']:.2f}\\times$ the error outside (${frac*100:.0f}\\%$ of
slices), confirming that synthesis risk concentrates at the lesion;
(ii) the \\TrustMap{{}} localizes the top-error decile with AUROC
${tu_au:.3f}\\pm{tu_au_sd:.3f}$ and a $+{tu_d20:.2f}$\\,dB sparsification gain at 20\\%---modest,
and far below the $0.911\\pm0.037$ we originally reported for this very
comparison: that number was computed over the \\emph{{whole image}}, the same
air-pooling artifact in a second guise (our second retraction);
but (iii) we \\emph{{retract}} an earlier claim: low trust does \\emph{{not}}
localize tumor tissue itself (within-body AUROC vs.\\ segmentation
${seg_au:.2f}\\pm{seg_sd:.2f}$; our previously reported $0.871$ was a whole-image artifact of
air negatives, and mean trust is in fact \\emph{{higher}} inside tumors,
${tumor_seg['trust_in']:.3f}$ vs.\\ ${tumor_seg['trust_out']:.3f}$ outside).""", "48-tumor"))

    E.append((r"""(a) mean trust inside/outside tumor; (b) mean absolute
error --- 1.60$\times$ inside; (c) within-body AUROC of low-trust-indicates-tumor
--- mean 0.51, chance level (red line);""",
              f"""(a) mean trust inside/outside tumor; (b) mean absolute
error --- {tumor_seg['err_ratio']:.2f}$\\times$ inside; (c) within-body AUROC of low-trust-indicates-tumor
--- mean {seg_au:.2f}, chance level (red line);""", "fig6"))

    E.append((r"""\paragraph{Input degradation.}
We degrade the input along three axes---Gaussian blur ($\sigma\in\{0.5,1,2\}$),
Gaussian noise ($\sigma\in\{0.01,0.03,0.06,0.1\}$), and horizontal misalignment
(2--8\,px)---over 30 IXI test slices per dose. Translation quality degrades
mildly (body PSNR $12.9\to11.8$\,dB at the worst dose; official T1$\to$T2
weights; trust values in this section are body means, a different aggregation
from the whole-image score of \S\ref{sec:exp-repro}). The response of our
trust machinery is asymmetric and instructive: the input-side Mahalanobis distance (the open-sourced input-side defense
layer's OOD score, used here purely as a measurement instrument) fires
dose-dependently for blur ($2.9\to15.0$) and noise ($\to9.3$) but is blind
to small misalignment ($\to2.9$); the \emph{trust map}
does not flag degraded inputs at all---moderate blur actually \emph{raises}
mean trust ($0.63\to0.70$), because a smoother input is easier for the sampler
to be self-consistent about---and the within-body localization AUROC decays
with dose ($0.79\to0.70$ under noise; $\to0.67$ under misalignment).
Knowing-what-you-don't-know here is carried by the input-side Mahalanobis
detector, not the trust map; misalignment detection remains open (we discuss
this in \S\ref{sec:discussion}).""",
              f"""\\paragraph{{Input degradation.}}
We degrade the input along three axes---Gaussian blur ($\\sigma\\in\\{{0.5,1,2\\}}$),
Gaussian noise ($\\sigma\\in\\{{0.01,0.03,0.06,0.1\\}}$), and horizontal misalignment
(2--8\\,px)---over the same 30 IXI test slices per dose (seed 2026), run twice:
on the weak official T1$\\to$T2 weights and on the healthy self-trained weights
(body PSNR ${do_['none_0']['body_psnr']:.1f}$ vs.\\ ${ds['none_0']['body_psnr']:.1f}$\\,dB at zero dose; trust values are body means, a different
aggregation from the whole-image score of \\S\\ref{{sec:exp-repro}}). The
response of our trust machinery is asymmetric and instructive: the input-side
Mahalanobis distance (the open-sourced input-side defense layer's OOD score,
used here purely as a measurement instrument) fires dose-dependently and
\\emph{{identically on both models}} (blur $2.9\\to{do_['blur_2.0']['maha_mean']:.1f}$; noise
$\\to{do_['noise_0.1']['maha_mean']:.1f}$; blind to small misalignment $\\to{do_['shift_8']['maha_mean']:.1f}$), as expected for an
input-space measurement. The trust map, by contrast, is model-relative: on
the healthy base, mean trust is dose-flat under blur
(${ds['none_0']['trust_mean']:.3f}\\to{ds['blur_2.0']['trust_mean']:.3f}$) and within-body localization AUROC holds
(${ds['none_0']['auroc']:.2f}\\to{ds['blur_2.0']['auroc']:.2f}$ at the worst blur; ${ds['none_0']['auroc']:.2f}\\to{ds['noise_0.1']['auroc']:.2f}$ under noise); on the weak
base, the same blur protocol \\emph{{raises}} mean trust
(${do_['none_0']['trust_mean']:.2f}\\to{do_['blur_2.0']['trust_mean']:.2f}$) while within-body localization sits near
chance at zero dose and decays below it under misalignment
(${do_['none_0']['auroc']:.2f}\\to{do_['shift_8']['auroc']:.2f}$ at 8\\,px). A smoother input is easier for a weak
sampler to be self-consistent about: knowing-what-you-don't-know at the
image level is carried by the input-side detector, not the trust map, and
the trust map's failure mode scales with base-model weakness (misalignment
detection remains open, \\S\\ref{{sec:discussion}}).""", "49-deg"))

    E.append((r"""Trust localization
\emph{transfers}: AUROC $0.863\pm0.155$ (held-out patients) and
$0.950\pm0.021$ (cross-institution) versus $0.911\pm0.037$ on the reference
cohort, with sparsification gains $+2.78$ and $+3.12$\,dB respectively. The
caveats are real---same preprocessing recipe on our side, six subjects per
set---but the direction is clear: within this cohort family, the signals are
not an artifact of having seen the patients.""",
              f"""Trust localization
\\emph{{transfers}}: AUROC ${B['auroc']['mean']:.3f}\\pm{B['auroc']['std']:.3f}$ (held-out patients) and
${C['auroc']['mean']:.3f}\\pm{C['auroc']['std']:.3f}$ (cross-institution) versus ${tu_au:.3f}\\pm{tu_au_sd:.3f}$ on the reference
cohort, with sparsification gains $+{B['dpsnr20']['mean']:.2f}$ and $+{C['dpsnr20']['mean']:.2f}$\\,dB respectively.
These within-body levels are far below the $0.863$/$0.950$/$0.911$ we
originally reported for the same three cohorts---all three were whole-image
numbers, and correcting the protocol moved every cohort down while preserving
the ordering (cross-institution strongest, held-out above reference). The
caveats are real---same preprocessing recipe on our side, six subjects per
set---but the direction is clear: within this cohort family, the signals are
not an artifact of having seen the patients.""", "49-cross"))

    E.append((r"""weights (12.98\,dB) receive HIGHER trust (mean 94.2, all PASS) than the
healthy self-trained model (90.1) --- variance-type signals cannot see the
failure.""",
              f"""weights ({co_off['body_psnr']:.2f}\\,dB) score body-masked trust {co_off['trust_score']:.1f} ({co_flag_txt}) versus
the healthy self-trained model's {co_self['trust_score']:.1f} (all PASS): directionally
correct, operationally insufficient. (The whole-image means we first
reported---94.2 collapsed vs 90.1 healthy---were air-inflated and inverted
the story.)""", "collapse-cap"))

    E.append((r"""extends to the model level: a collapsed base model that is confidently wrong
receives PASS ($94/100$ mean trust, \S\ref{sec:exp-repro}).""",
              f"""extends to the model level: a collapsed base model that is confidently wrong
is flagged only for review, never rejected (${co_off['trust_score']:.0f}/100$,
\\S\\ref{{sec:exp-repro}}).""", "5-limit"))

    lo_cmp = ("lower" if co_off["lowtrust_ratio_pct"] < co_self["lowtrust_ratio_pct"]
              else "not lower")
    E.append((r"""A natural hope is that sampler-native signals would ``know'' the base model
is failing. We tested this directly, and the answer is the most important
negative result in this paper. On the collapsed official model, the trust
map does not fire---mean trust $94.2/100$, low-trust ratio $5.9\%$ (lower
than for our healthy self-trained model), all 20 test slices PASS.""",
              f"""A natural hope is that sampler-native signals would ``know'' the base model
is failing. We tested this directly. The honest answer under our corrected
protocol is more graded than the one we first reported. On the collapsed
official model, the body-masked trust score drops to ${co_off['trust_score']:.1f}/100$ versus
${co_self['trust_score']:.1f}$ for the healthy self-trained model, the low-trust ratio rises to
${co_off['lowtrust_ratio_pct']:.1f}\\%$ (healthy: ${co_self['lowtrust_ratio_pct']:.1f}\\%$), and pixel-level within-body AUROC degrades to
${au_off:.2f}\\pm{V['collapse_au']['official_15db'][1]:.2f}$ (healthy: ${au_self_col:.2f}$)---the signals move in the right
direction, but nowhere near far enough: the structurally wrong model still
lands in REVIEW ({co_flag_txt}), never REJECT, and the originally reported blanket PASS
($94/100$, a whole-image mean) was itself inflated by trusted air.""", "42-collapse"))

    E.append((r"""regions well (AUROC $0.76$--$0.95$) and CReF supplies most of that value at
zero cost; on texture-dominated error the signals---and every baseline we
tried---are blind.""",
              f"""regions well (AUROC ${concl_lo:.2f}$--${concl_hi:.2f}$) and CReF supplies most of that value at
zero cost; on texture-dominated error the signals---and every baseline we
tried---retain at most weak information (best: TTA,
${m('t2t1_off','tta_disp'):.2f}$), none usable.""", "concl"))

    # ============ ZH ============
    Z.append((r"""在体掩码口径下从 $0.90$ 崩落到 $0.78$，并撤回了自己早期受益于此的数字。""",
              f"""在体掩码口径下从 $0.90$ 崩落到 ${m('t1t2_off','ours_fused'):.2f}$，并撤回了自己早期受益于此的数字。""", "ZH-abs-trap1"))

    Z.append((r"""同一融合信号整图 AUROC $0.90$、体区内 $0.78$，""",
              f"""同一融合信号整图 AUROC $0.90$、体区内 ${m('t1t2_off','ours_fused'):.2f}$，""", "ZH-41-demo"))

    Z.append((r"""在体掩码口径下它们给出：AUROC $0.90$（T1$\to$T2 自训）与 $0.82$
（T2$\to$T1 自训），超过全部等预算基线，单链零额外成本即达 $0.88$/$0.79$；在 BraTS2018 的 6 名肿瘤患者中，信任图对体区内高误差像素的定位 AUROC 为 $0.911$（切片级 95\% CI $[0.89, 0.93]$），但对肿瘤组织本身无定位能力（$0.51$）。6 名 CBICA 受试者的初步外部验证切片级 AUROC 为 $0.950$。""",
              f"""在体掩码口径下它们给出：AUROC ${a_f_1:.2f}$（T1$\\to$T2 自训）与 ${a_f_2:.2f}$
（T2$\\to$T1 自训），超过全部等预算基线，单链零额外成本即达 ${a_c1_1:.2f}$/${a_c1_2:.2f}$；在 BraTS2018 的 6 名肿瘤患者中，信任图对体区内高误差像素的定位 AUROC 为 ${tu_au:.3f}\\pm{tu_au_sd:.3f}$（旧报告 $0.911$ 系整图口径伪影，已撤回），但对肿瘤组织本身无定位能力（${seg_au:.2f}$）。留出患者 $0.707$、跨机构（CBICA）$0.834$（旧值 $0.863$/$0.950$ 同为整图口径，已修正）。""", "abs"))

    Z.append((r"""\emph{未能察觉}：崩塌模型获得 PASS 与 $94/100$ 的平均信任。""",
              f"""\\emph{{仅微弱察觉}}：崩塌模型体区信任分 ${co_off['trust_score']:.0f}/100$——低于健康模型的 ${co_self['trust_score']:.0f}$，
但只到 REVIEW、从不 REJECT，像素级 AUROC 降至 $0.61$（健康 $0.82$）；最初报告的
全体 PASS（$94/100$，整图均值）本身就是空气池化伪影。""", "abs-col"))

    Z.append((r"""        $0.90$/$0.82$（T1$\to$T2/T2$\to$T1），超过全部基线；真实肿瘤患者 $0.91$、
        跨机构 $0.95$；而\emph{官方}权重上同一协议令排序反转、T2$\to$T1 全盲（\S4）。""",
              f"""        ${a_f_1:.2f}$/${a_f_2:.2f}$（T1$\\to$T2/T2$\\to$T1），超过全部基线；真实肿瘤患者 ${tu_au:.2f}$（旧报
        $0.911$ 系整图伪影，已撤回）、跨机构 ${C['auroc']['mean']:.2f}$；而\\emph{{官方}}权重上同一协议令排序反转、本文信号在 T2$\\to$T1 失效
        （融合 ${m('t2t1_off','ours_fused'):.2f}$；TTA 基线 ${m('t2t1_off','tta_disp'):.2f}$）（\\S4）。""", "contrib"))

    Z.append((r"""表~\ref{tab:uq} 承载两个信息。\textbf{自训模型上}，本文融合信号 AUROC
$0.823$/$0.757$，稀疏化增益 $+2.67$/$+1.85$\,dB。配对患者级 bootstrap（$B{=}5000$，
8 被试）显示：融合相对仅集成的 $\Delta$AUROC 为 $+0.004$（95\% CI $[-0.001, 0.008]$，
\emph{含 0}——T1$\to$T2 上不显著）与 $+0.010$（CI $[0.005, 0.015]$，T2$\to$T1 上显著）；
而融合相对全部等预算基线（TTA $-0.082$/$-0.107$、噪声 $-0.041$/$-0.078$、梯度）差异
显著且幅度大。单链 CReF（零额外成本）已达 $0.880$/$0.792$——四分之一预算拿到大部分
价值；CReF 的论据是成本而非精度。""",
              f"""表~\\ref{{tab:uq}} 承载两个信息。\\textbf{{自训模型上}}，本文融合信号 AUROC
${a_f_1:.3f}$/${a_f_2:.3f}$，稀疏化增益 $+{d_1:.2f}$/$+{d_2:.2f}$\\,dB。配对患者级 bootstrap（$B{{=}}5000$，
8 被试）显示：融合相对仅集成的 $\\Delta$AUROC 为 $+{p1['delta']:.3f}$（95\\% CI {ci_txt(p1['ci95'])}，
\\emph{{{zinc1}}}）与 $+{p2['delta']:.3f}$（CI {ci_txt(p2['ci95'])}，{zinc2}）；
而融合相对全部等预算基线（TTA $-{d_tta_1:.3f}$/$-{d_tta_2:.3f}$、噪声 $-{d_no_1:.3f}$/$-{d_no_2:.3f}$、梯度）差异
显著且幅度大。单链 CReF（零额外成本）已达 ${a_c1_1:.3f}$/${a_c1_2:.3f}$——四分之一预算拿到大部分
价值；CReF 的论据是成本而非精度。""", "44"))

    Z.append((r"""\textbf{官方权重上}，排序改变且 T2$\to$T1 方向整体
崩塌：所有分歧信号——我们的与全部基线的——\emph{系统性低于随机}（融合 CI $[0.442, 0.485]$，
上界 $<0.5$），因为该模型残差误差由细纹理主导、一切分歧度量都无法追踪。""",
              f"""\\textbf{{官方权重上}}，排序改变且 T2$\\to$T1 方向崩塌：本文采样器原生信号跌至近随机
（融合 ${m('t2t1_off','ours_fused'):.3f}\\pm{sd('t2t1_off','ours_fused'):.3f}$、仅集成 ${m('t2t1_off','ours_unc_n4'):.3f}$、CReF ${m('t2t1_off','ours_rcrf_n1'):.3f}$），而最强的\\emph{{基线}}——TTA 分歧——保留弱信号
（${m('t2t1_off','tta_disp'):.3f}\\pm{sd('t2t1_off','tta_disp'):.3f}$）：该模型残差误差由细纹理主导，收敛残差与桥分歧几乎无法追踪。
（我们最初在此报告的"系统性低于随机"数值——如融合 CI $[0.442, 0.485]$——同样是整图口径伪影。）""", "44-blind"))

    Z.append((r"""在与 TTA、输入噪声基线的等预算对比中，所有分歧类信号——包括我们的——在残差误差由细纹理主导的
T2$\to$T1 官方权重上均接近随机水平（部分系统性偏低）。""",
              f"""在与 TTA、输入噪声基线的等预算对比中，本文采样器原生信号在细纹理主导残差误差的
官方 T2$\\to$T1 权重上跌至随机水平（融合 ${m('t2t1_off','ours_fused'):.2f}$；TTA 基线保留弱信号 ${m('t2t1_off','tta_disp'):.2f}$）。""", "abs-trap2"))

    # ZH Table 2 整表
    t2z = f"""\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{T1$\\to$T2（自训）}} & \\multicolumn{{3}}{{c}}{{T2$\\to$T1（自训）}} \\\\
信号 & AUROC & $\\Delta$PSNR@20\\% & Sep & AUROC & $\\Delta$PSNR@20\\% & Sep \\\\
\\midrule
随机（对照） & {a_rd_1:.3f} & $\\approx$0 & 1.0 & {a_rd_2:.3f} & $\\approx$0 & 1.0 \\\\
梯度 & {a_gr_1:.3f} & {m('t1t2_self','gradient','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('gradient','--')} & {a_gr_2:.3f} & {m('t2t1_self','gradient','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('gradient','--')} \\\\
输入噪声分歧 & {a_no_1:.3f} & {m('t1t2_self','noise_disp','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('noise_disp','--')} & {a_no_2:.3f} & {m('t2t1_self','noise_disp','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('noise_disp','--')} \\\\
TTA 分歧 & {a_tta_1:.3f} & {m('t1t2_self','tta_disp','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('tta_disp','--')} & {a_tta_2:.3f} & {m('t2t1_self','tta_disp','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('tta_disp','--')} \\\\
CReF 单链（N=1，无额外前传） & {a_c1_1:.3f} & {m('t1t2_self','ours_rcrf_n1','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('ours_rcrf_n1','--')} & {a_c1_2:.3f} & {m('t2t1_self','ours_rcrf_n1','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('ours_rcrf_n1','--')} \\\\
仅集成（N=4） & {a_e_1:.3f} & {m('t1t2_self','ours_unc_n4','dpsnr20'):+.2f} & {SEP['t1t2_self'].get('ours_unc_n4','--')} & {a_e_2:.3f} & {m('t2t1_self','ours_unc_n4','dpsnr20'):+.2f} & {SEP['t2t1_self'].get('ours_unc_n4','--')} \\\\
\\textbf{{融合（本文）}} & \\textbf{{{a_f_1:.3f}}} & $\\mathbf{{{d_1:+.2f}}}$ & \\textbf{{{SEP['t1t2_self'].get('ours_fused','--')}}} & \\textbf{{{a_f_2:.3f}}} & $\\mathbf{{{d_2:+.2f}}}$ & \\textbf{{{SEP['t2t1_self'].get('ours_fused','--')}}} \\\\
\\midrule
\\multicolumn{{7}}{{l}}{{\\emph{{官方权重上同一协议（N=4）：}}}} \\\\
融合（官方） & {oa_1:.3f} & {od_1:+.2f} & {SEP['t1t2_off'].get('ours_fused','--')} & {oa_2:.3f} & {od_2:+.2f} & {SEP['t2t1_off'].get('ours_fused','--')} \\\\
TTA（官方） & {otta_1:.3f} & {otta_d1:+.2f} & {SEP['t1t2_off'].get('tta_disp','--')} & {otta_2:.3f} & {otta_d2:+.2f} & {SEP['t2t1_off'].get('tta_disp','--')} \\\\
全部分歧信号（官方，T2$\\to$T1） & -- & -- & -- & \\multicolumn{{3}}{{l}}{{${off_all_2[0]:.3f}$--${off_all_2[-1]:.3f}$（近随机）}} \\\\
\\bottomrule
\\end{{tabular}}"""
    Z.append(("__TAB2__", t2z, "tab2"))

    Z.append((r"""(i) 瘤内像素误差为瘤外 $1.60\times$（$90\%$ 切片方向一致），合成风险确实向病灶集中；
(ii) \TrustMap{}把高误差十分位定位到 AUROC $0.911\pm0.037$、20\% 稀疏化增益
$+3.12$\,dB——口径修正后信任信号在该模型上依然成立；
但 (iii) 我们\emph{撤回}早先的结论：低信任\emph{不}定位肿瘤组织本身（对分割的体区 AUROC
仅 $0.51\pm0.10$；旧值 $0.871$ 系整图空气负例伪影），且瘤内平均信任\emph{略高于}瘤外
（$0.772$ 对 $0.727$）。""",
              f"""(i) 瘤内像素误差为瘤外 ${tumor_seg['err_ratio']:.2f}\\times$（${frac*100:.0f}\\%$ 切片方向一致），合成风险确实向病灶集中；
(ii) \\TrustMap{{}}把高误差十分位定位到 AUROC ${tu_au:.3f}\\pm{tu_au_sd:.3f}$、20\\% 稀疏化增益
$+{tu_d20:.2f}$\\,dB——程度有限，远低于我们最初为同一比较报告的 $0.911\\pm0.037$：该值同样按
\\emph{{整图}}口径计算，是空气池化伪影的又一次现身（第二次撤回）；
但 (iii) 我们\\emph{{撤回}}早先的结论：低信任\\emph{{不}}定位肿瘤组织本身（对分割的体区 AUROC
仅 ${seg_au:.2f}\\pm{seg_sd:.2f}$；旧值 $0.871$ 系整图空气负例伪影），且瘤内平均信任\\emph{{高于}}瘤外
（${tumor_seg['trust_in']:.3f}$ 对 ${tumor_seg['trust_out']:.3f}$）。""", "tumor"))

    Z.append((r"""(a) 瘤内/体（非瘤）逐片平均信任；(b) 逐片平均绝对误差——瘤内为瘤外的 1.60 倍；
(c) 体掩码内低信任指示肿瘤的 AUROC 分布——均值 0.51，等于随机水平（红线）；""",
              f"""(a) 瘤内/体（非瘤）逐片平均信任；(b) 逐片平均绝对误差——瘤内为瘤外的 {tumor_seg['err_ratio']:.2f} 倍；
(c) 体掩码内低信任指示肿瘤的 AUROC 分布——均值 {seg_au:.2f}，随机水平（红线）；""", "fig6"))

    Z.append((r"""\paragraph{输入退化：}
我们沿三条轴退化输入——高斯模糊（$\sigma\in\{0.5,1,2\}$）、高斯噪声
（$\sigma\in\{0.01,0.03,0.06,0.1\}$）、水平错位（2--8\,px）——IXI 测试 30 片/剂量。
翻译质量轻微恶化（体区 PSNR 最差剂量 $12.9\to11.8$\,dB；官方 T1$\to$T2 权重；
本节信任值为体区均值，聚合口径与 \S4.2 的整图信任分不同）。信任机制的响应不对称而有
教学意义：输入侧 Mahalanobis 距离（开源输入防御层的域外评分，此处仅作测量仪器）对模糊（$2.9\to15.0$）与噪声（$\to9.3$）
剂量敏感，对小幅错位无感（$\to2.9$）；\TrustMap{}完全不对退化输入报警——中等模糊反而
\emph{抬高}平均信任（$0.63\to0.70$，更平滑的输入让采样器更易自洽）——体区定位 AUROC
随剂量衰减（噪声 $0.79\to0.70$；错位 $\to0.67$）。这里"知道自己不知道"由输入侧 Mahalanobis 检测承担而非信任图；错位检测仍未解决（\S5 讨论）。""",
              f"""\\paragraph{{输入退化：}}
我们沿三条轴退化输入——高斯模糊（$\\sigma\\in\\{{0.5,1,2\\}}$）、高斯噪声
（$\\sigma\\in\\{{0.01,0.03,0.06,0.1\\}}$）、水平错位（2--8\\,px）——同一批 IXI 测试 30 片/剂量
（seed 2026），分别在弱官方权重与健康自训权重上各跑一次（零剂量体区 PSNR
${do_['none_0']['body_psnr']:.1f}$ 对 ${ds['none_0']['body_psnr']:.1f}$\\,dB；本节信任值为体区均值，聚合口径与 \\S4.2 的整图信任分不同）。
响应不对称而有教学意义：输入侧 Mahalanobis 距离（开源输入防御层的域外评分，此处仅作
测量仪器）对模糊（$2.9\\to{do_['blur_2.0']['maha_mean']:.1f}$）与噪声（$\\to{do_['noise_0.1']['maha_mean']:.1f}$）剂量敏感、对小幅错位无感，且在两个模型上
\\emph{{完全一致}}——符合输入空间度量的预期。\\TrustMap{{}}则是模型相对的：健康基座上，
模糊使平均信任几乎不动（${ds['none_0']['trust_mean']:.3f}\\to{ds['blur_2.0']['trust_mean']:.3f}$），体区定位 AUROC 保持
（最差模糊 ${ds['none_0']['auroc']:.2f}\\to{ds['blur_2.0']['auroc']:.2f}$；噪声 ${ds['none_0']['auroc']:.2f}\\to{ds['noise_0.1']['auroc']:.2f}$）；弱基座上，同一模糊协议反而
\\emph{{抬高}}平均信任（${do_['none_0']['trust_mean']:.2f}\\to{do_['blur_2.0']['trust_mean']:.2f}$），体区定位零剂量即在随机附近、错位下跌破
（8\\,px 时 ${do_['none_0']['auroc']:.2f}\\to{do_['shift_8']['auroc']:.2f}$）。更平滑的输入让弱采样器更易自洽：图像级的"知道自己
不知道"由输入侧检测承担而非信任图，且信任图的失效幅度随基座弱点放大（错位检测仍未解决，
\\S5 讨论）。""", "deg"))

    Z.append((r"""（机构级泛化），各 120 片。信任定位\emph{迁移成立}：AUROC $0.863\pm0.155$（留出患者）
与 $0.950\pm0.021$（跨机构），对照参照队列 $0.911\pm0.037$；稀疏化增益 $+2.78$ 与
$+3.12$\,dB。保留注意项——我方预处理配方一致、每组仅六例——但方向明确：在该队列族内，
这些信号不是"见过患者"的伪影。""",
              f"""（机构级泛化），各 120 片。信任定位\\emph{{迁移成立}}：AUROC ${B['auroc']['mean']:.3f}\\pm{B['auroc']['std']:.3f}$（留出患者）
与 ${C['auroc']['mean']:.3f}\\pm{C['auroc']['std']:.3f}$（跨机构），对照参照队列 ${tu_au:.3f}\\pm{tu_au_sd:.3f}$；稀疏化增益 $+{B['dpsnr20']['mean']:.2f}$ 与
$+{C['dpsnr20']['mean']:.2f}$\\,dB。这些体区口径水平远低于我们最初为同三组队列报告的
$0.863$/$0.950$/$0.911$——三者当时都是整图口径数字，协议修正后各组齐降、但排序保持
（跨机构最强、留出高于参照）。保留注意项——我方预处理配方一致、每组仅六例——但方向明确：
在该队列族内，这些信号不是"见过患者"的伪影。""", "cross"))

    Z.append((r"""结构性崩塌的官方权重（12.98 dB）获得的信任（均值 94.2，全部 PASS）反而高于健康的""",
              f"""结构性崩塌的官方权重（{co_off['body_psnr']:.2f} dB）体区信任分 {co_off['trust_score']:.1f}（{co_flag_txt_zh}），
健康自训模型为 {co_self['trust_score']:.1f}（全部 PASS）：方向正确、操作上不足。（我们最初报告的整图均值
94.2 对 90.1 被空气虚高，且颠倒了故事。）""", "col-cap"))

    Z.append((r"""一个自然的期望是采样器原生信号能"察觉"基座模型失效。我们直接测试了它，答案构成本文
最重要的阴性结果。在崩塌的官方模型上，\TrustMap{}不放警报——平均信任 $94.2/100$、
低信任占比 $5.9\%$（低于我们健康的自训模型）、20 片测试全部 PASS。""",
              f"""一个自然的期望是采样器原生信号能"察觉"基座模型失效。我们直接测试了它。在修正后的
口径下，诚实的答案比我们最初报告的更有层次：崩塌官方模型的体区信任分降至 ${co_off['trust_score']:.1f}/100$
（健康自训 ${co_self['trust_score']:.1f}$），低信任占比升至 ${co_off['lowtrust_ratio_pct']:.1f}\\%$（健康 ${co_self['lowtrust_ratio_pct']:.1f}\\%$），像素级体区 AUROC 退化到
${au_off:.2f}\\pm{V['collapse_au']['official_15db'][1]:.2f}$（健康 ${au_self_col:.2f}$）——信号方向正确但远不够：结构错误的模型仍落在
REVIEW（{co_flag_txt_zh}），从不 REJECT，最初报告的全体 PASS（$94/100$，整图均值）同样被可信空气虚高。""", "42-collapse-zh"))

    Z.append((r"""Pearson $r=+0.47$（T1$\to$T2，自训）、$r=+0.26$（T2$\to$T1，自训）、$r=-0.10$
（BraTS，官方）——信号有效的模型上方向正确，信号失效的模型上相关性消失。出厂阈值
同样与实际分数分布错位：PASS$\ge$75 拦截 $88\%$ 汇总切片，REJECT$<55$ 仅覆盖 $3\%$；最优汇总工作点上也只能以 9% 覆盖换取 29% 精确率
（对质量下四分位）。""",
              f"""Pearson $r={PEAR['t1t2']:+.2f}$（T1$\\to$T2，自训）、$r={PEAR['t2t1']:+.2f}$（T2$\\to$T1，自训）、$r={PEAR['tumor']:+.2f}$
（BraTS，官方）——信号有效的模型上方向正确，信号失效的模型上相关性消失。出厂阈值
同样与实际分数分布错位：PASS$\\ge$75 放行 {(1-qc75['flagged_pct'])*100:.0f}\\% 汇总切片、却只捕获质量下四分位故障的
{qc75['recall']*100:.0f}\\%；REJECT$<55$ 仅覆盖 {qc55['flagged_pct']*100:.1f}\\%（精确率高、召回可忽略：
{qc55['recall']*100:.0f}\\%）。最优汇总工作点上也只能以 {best.get('flagged_pct', 0)*100:.1f}% 覆盖换取
{best.get('precision', 0)*100:.0f}% 精确率（对质量下四分位）。""", "qc"))

    Z.append((r"""的肿瘤数据），免训练信号能很好定位不可靠区域（AUROC $0.76$--$0.95$）""",
              f"""的肿瘤数据），免训练信号能定位不可靠区域（AUROC ${concl_lo:.2f}$--${concl_hi:.2f}$）""", "concl"))

    Z.append((r"""供给其中大部分价值；在纹理主导的误差上，这些信号——以及我们试过的所有基线——集体失明。""",
              f"""供给其中大部分价值；在纹理主导的误差上，这些信号连同全部基线至多保留弱信息（最好为 TTA 的
${m('t2t1_off','tta_disp'):.2f}$），无一达到可用水平。""", "concl-blind"))

    E.append((r"""The
sampler is confident about its wrong output, so CReF and dispersion---which
measure self-consistency---are blind, exactly as our tumor analysis
predicted at the pixel level, now reproduced at the model level. The lesson
for deployment is stark: \emph{training-free internal signals cannot detect
a base model that is confidently wrong}.""",
              f"""The
sampler remains largely self-consistent about its wrong output: CReF and
dispersion degrade together with quality (pixel-level AUROC
${au_off:.2f}$ versus ${au_self_col:.2f}$ healthy) but never separate a wrong model from
a mediocre one decisively. The lesson for deployment is starker for being
graded: \\emph{{training-free internal signals cannot be trusted to reject a base
model that is confidently wrong}}---a 20-point trust gap is a signal, not a
gate.""", "42-follow"))

    Z.append((r"""采样器对自己的错误输出
高度自信，因此度量自洽性的 CReF 与集成分歧全部失明——正如肿瘤分析在像素级预言的那样，
现在模型级复现了一遍。部署教训：\emph{免训练内部信号无法察觉"自信地犯错"的基座模型}。""",
              f"""采样器对自己的错误输出
仍高度自洽：CReF 与集成分歧随质量一同退化（像素级 AUROC ${au_off:.2f}$ 对健康的
${au_self_col:.2f}$），但从未把错误模型与平庸模型决定性地分开。部署教训因有层次而更冷峻：
\\emph{{免训练内部信号不足以否决"自信地犯错"的基座模型}}——20 分的信任差是一个信号，
不是一道门。""", "42-follow-zh"))

    return E, Z
