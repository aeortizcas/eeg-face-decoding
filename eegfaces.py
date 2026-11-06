"""Face processing in EEG: N170 and time-resolved decoding on OpenNeuro ds002718.

Data: Wakeman & Henson multi-subject face processing dataset (EEG part, BIDS/EEGLAB,
250 Hz, 70 EEG channels), repackaged by Martinez, Truong, Makeig & Delorme. Participants
saw famous, unfamiliar and scrambled faces.

    python eegfaces.py --subjects 2 3 4 5 6 7 --out results
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

S3 = "https://s3.amazonaws.com/openneuro.org/ds002718"
TMIN, TMAX = -0.2, 0.8
N170 = (0.15, 0.20)                     # seconds
REJECT_UV = 150.0                       # peak-to-peak rejection threshold


def fetch(sub: int, root: str) -> dict:
    """Download the four files of one subject (about 235 MB) if they are not there yet."""
    s = f"sub-{sub:03d}"
    paths = {}
    for kind, ext in [("eeg", "set"), ("events", "tsv"), ("electrodes", "tsv"), ("channels", "tsv")]:
        name = f"{s}_task-FaceRecognition_{kind}.{ext}"
        dst = os.path.join(root, s, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            print(f"  downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{S3}/{s}/eeg/{name}", dst)
        paths[kind] = dst
    return paths


def load_epochs(paths: dict) -> mne.Epochs:
    raw = mne.io.read_raw_eeglab(paths["eeg"], preload=True, verbose="error")
    ch = pd.read_csv(paths["channels"], sep="\t")
    kind = {"HEOG": "eog", "VEOG": "eog", "EOG": "eog", "ECG": "ecg", "EKG": "ecg"}
    types = {str(n).strip(): kind.get(str(t).strip().upper(), "eeg") for n, t in zip(ch["name"], ch["type"])}
    raw.set_channel_types({k: v for k, v in types.items() if k in raw.ch_names and v != "eeg"},
                          on_unit_change="ignore")
    # Electrode positions from BIDS (the .set positions are rotated for some subjects). They are in cm
    # in the EEGLAB/BESA convention (x -> nose, y -> left ear, z -> up), which we checked with blink
    # artefacts (the most VEOG-correlated channels have the largest x). MNE's head frame is
    # x -> right ear, y -> nose, so (x, y, z)_mne = (-y, x, z)_eeglab.
    el = pd.read_csv(paths["electrodes"], sep="\t")
    eeg_names = set(mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))["ch_names"])
    pos = {str(n): np.array([-y, x, z]) / 100.0 for n, x, y, z in zip(el["name"], el["x"], el["y"], el["z"])
           if str(n) in eeg_names}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"), on_missing="ignore")

    raw.filter(0.1, 40.0, picks=["eeg", "eog"], verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")

    ev = pd.read_csv(paths["events"], sep="\t")
    ev = ev[ev.event_type == "faces"].copy()
    ev["cond"] = ev.face_type                       # famous / unfamiliar / scrambled
    codes = {"famous": 1, "unfamiliar": 2, "scrambled": 3}
    samples = np.round(ev.onset.to_numpy() * raw.info["sfreq"]).astype(int) + raw.first_samp
    events = np.c_[samples, np.zeros(len(ev), int), ev.cond.map(codes).to_numpy()]
    epochs = mne.Epochs(raw, events, event_id=codes, tmin=TMIN, tmax=TMAX, baseline=(TMIN, 0.0),
                        picks="eeg", reject=None, preload=True, verbose="error")
    # bad channels: median peak-to-peak amplitude more than 3x the median over channels -> interpolate,
    # so every subject keeps the same channel set; then reject noisy epochs.
    ptp = np.median(np.ptp(epochs.get_data(), axis=2), axis=0)
    epochs.info["bads"] = [c for c, v in zip(epochs.ch_names, ptp) if v > 3 * np.median(ptp)]
    if epochs.info["bads"]:
        print("  interpolating bad channels:", epochs.info["bads"], flush=True)
        epochs.interpolate_bads(reset_bads=True, verbose="error")
    epochs.drop_bad(reject=dict(eeg=REJECT_UV * 1e-6), verbose="error")
    return epochs


def decode(epochs: mne.Epochs, a: list[str], b: list[str], seed: int = 0) -> np.ndarray:
    """Time-resolved AUC of a logistic regression on all EEG channels (5-fold CV)."""
    ea, eb = epochs[a], epochs[b]
    X = np.concatenate([ea.get_data(), eb.get_data()])
    y = np.r_[np.ones(len(ea)), np.zeros(len(eb))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    est = SlidingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose="error")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_multiscore(est, X, y, cv=cv, n_jobs=1).mean(0)


def roi_channels(info, k: int = 6) -> list[str]:
    """Right posterior-temporal ROI chosen from electrode positions alone (no data peeking):
    the k channels closest to the right occipito-temporal site (x=+5 cm, y=-6 cm)."""
    pos = {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"] if np.isfinite(ch["loc"][:3]).all()}
    target = np.array([0.05, -0.06])
    d = {n: np.linalg.norm(p[:2] - target) for n, p in pos.items()}
    return sorted(d, key=d.get)[:k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    evk, auc_fs, auc_fu, counts = {"faces": [], "scrambled": []}, [], [], {}
    times, roi, info = None, None, None
    for s in a.subjects:
        print(f"sub-{s:03d}", flush=True)
        ep = load_epochs(fetch(s, a.data))
        counts[s] = {c: len(ep[c]) for c in ["famous", "unfamiliar", "scrambled"]}
        print("  epochs kept:", counts[s], flush=True)
        evk["faces"].append(ep[["famous", "unfamiliar"]].average())
        evk["scrambled"].append(ep["scrambled"].average())
        auc_fs.append(decode(ep, ["famous", "unfamiliar"], ["scrambled"]))
        auc_fu.append(decode(ep, ["famous"], ["unfamiliar"]))
        times, info = ep.times, ep.info
        roi = roi or roi_channels(info)

    # ---------------- grand averages
    ga = {k: mne.grand_average(v) for k, v in evk.items()}
    diff = mne.combine_evoked([ga["faces"], ga["scrambled"]], weights=[1, -1])
    roi_ts = {k: ga[k].copy().pick(roi).data.mean(0) * 1e6 for k in ga}
    w = (times >= N170[0]) & (times <= N170[1])
    sub_diff = np.array([(f.copy().pick(roi).data.mean(0) - c.copy().pick(roi).data.mean(0))[w].mean() * 1e6
                         for f, c in zip(evk["faces"], evk["scrambled"])])

    # ---------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axvspan(*[t * 1000 for t in N170], color="0.9")
    for k, c in [("faces", "#d62728"), ("scrambled", "#1f77b4")]:
        ax.plot(times * 1000, roi_ts[k], color=c, label=k)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis(); ax.set_xlabel("time (ms)"); ax.set_ylabel("µV (negative up)")
    ax.set_title(f"N170, right occipito-temporal ROI (n={len(a.subjects)})"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_n170.png"), dpi=200); plt.close(fig)

    fig = diff.plot_topomap(times=[0.17], ch_type="eeg", show=False, colorbar=True)
    fig.set_size_inches(4.2, 3.4)
    fig.axes[0].set_title("faces − scrambled, 170 ms")
    fig.savefig(os.path.join(a.out, "fig_topomap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    for arr, lab, c in [(np.array(auc_fs), "faces vs scrambled", "#d62728"),
                        (np.array(auc_fu), "famous vs unfamiliar", "#2ca02c")]:
        m, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * arr.mean(0)
        ax.plot(times * 1000, m, color=c, label=lab); ax.fill_between(times * 1000, m - se, m + se, color=c, alpha=0.2)
    ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("decoding AUC"); ax.legend(frameon=False)
    ax.set_title("time-resolved decoding (mean ± s.e.m. over subjects)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_decoding.png"), dpi=200); plt.close(fig)

    # ---------------- summary
    def peak(arr):
        m = np.array(arr).mean(0); i = int(np.argmax(m)); return float(m[i]), float(times[i] * 1000)
    summ = dict(subjects=a.subjects, epochs=counts, roi=roi,
                n170_diff_uV_mean=float(sub_diff.mean()), n170_diff_uV_per_subject=sub_diff.round(2).tolist(),
                n170_subjects_negative=int((sub_diff < 0).sum()),
                faces_vs_scrambled_peak_auc=peak(auc_fs), famous_vs_unfamiliar_peak_auc=peak(auc_fu))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "decoding.npz"), times=times, auc_fs=np.array(auc_fs), auc_fu=np.array(auc_fu))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
