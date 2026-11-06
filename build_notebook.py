"""Build eeg_face_decoding.ipynb (self-contained: eegfaces.py is embedded with %%writefile)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "eegfaces.py"), encoding="utf-8").read()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [
    md("""
# Face processing in EEG: N170 and time-resolved decoding

Open data: **OpenNeuro ds002718** (Wakeman & Henson face processing dataset, EEG part).
Participants viewed famous, unfamiliar and scrambled faces while EEG was recorded (70 channels, 250 Hz).

This notebook
1. downloads a few subjects directly from the public OpenNeuro bucket (~235 MB each),
2. preprocesses them with MNE-Python (0.1–40 Hz, average reference, epochs −200…800 ms, 150 µV rejection),
3. shows the face-sensitive **N170** over right occipito-temporal electrodes,
4. decodes **faces vs. scrambled** and **famous vs. unfamiliar** over time with logistic regression (5-fold CV, AUC).

Runs on the free Colab CPU runtime. Six subjects take roughly 15–25 minutes.
"""),
    code("""
!pip -q install mne
SUBJECTS = [2, 3, 4, 5, 6, 7]   # sub-002 ... sub-019 are available
"""),
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": ("%%writefile eegfaces.py\n" + SRC).splitlines(True)},
    code("""
import subprocess, sys
subprocess.run([sys.executable, "eegfaces.py", "--subjects", *map(str, SUBJECTS), "--out", "results"], check=True)
"""),
    code("""
from IPython.display import Image, display
import json
for f in ["fig_n170.png", "fig_topomap.png", "fig_decoding.png"]:
    display(Image(f"results/{f}", width=520))
print(json.dumps(json.load(open("results/summary.json")), indent=1))
"""),
    code("""
import shutil
shutil.make_archive("eeg_results", "zip", "results")
print("download eeg_results.zip from the file panel")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "eeg_face_decoding.ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote eeg_face_decoding.ipynb")
