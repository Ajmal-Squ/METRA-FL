# METRA-FL

Official reproducibility package for **METRA-FL: Multi-Evidence Trust Scoring and Robust Aggregation for Poisoning-Resilient Federated Edge Intrusion Detection**.

METRA-FL scores each submitted client update using directional agreement, validation contribution, historical reputation, and norm-outlier evidence. Updates below the trust threshold are rejected. Accepted updates are clipped coordinate-wise and combined using normalized trust and local sample count.

## Repository contents

- `metra_fl_binary_rerun.py`: resumable experiment runner used to generate the reported results.
- `analyze_binary_results.py`: validates the experiment matrix, calculates paired tests, and regenerates result summaries and plots.
- `figures/generate_architecture.py`: programmatic architecture-figure generator required by the analysis script.
- `reference_results/`: the raw and summarized outputs from the completed run.
- `requirements.txt`: Python dependencies.

The manuscript and LaTeX source are intentionally not included.

## Dataset

The experiments use the official UNSW-NB15 training and testing CSV files:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

The dataset is not redistributed in this repository. Download the two files from the official UNSW-NB15 provider and place them in a directory named `data`.

## Run in Google Colab

1. Open a new Colab notebook and select **Runtime > Change runtime type > T4 GPU**.
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Upload or clone this repository:

```python
!git clone https://github.com/Ajmal-Squ/METRA-FL.git
%cd METRA-FL
!pip install -r requirements.txt
```

4. Place the two UNSW-NB15 CSV files in:

```text
/content/drive/MyDrive/METRA_FL/data
```

5. Start the complete experiment:

```python
!python metra_fl_binary_rerun.py \
  --data-dir "/content/drive/MyDrive/METRA_FL/data" \
  --out "/content/drive/MyDrive/METRA_FL/binary_rerun_results" \
  --seeds 11,29,47 \
  --rounds 30 \
  --zero-rounds 20
```

Each completed task is checkpointed. If Colab disconnects, run the same command again; completed tasks are skipped automatically.

The expected experiment matrix contains 78 federated runs and 27 post-preprocessing held-out-family runs, for 105 runs in total.

## Validate the supplied results

The reference outputs can be checked without rerunning model training:

```bash
python analyze_binary_results.py --results reference_results --out reproduced_analysis
```

A successful check prints:

```text
Validated 78 federated and 27 held-out-family runs
```

## Experimental boundary

- Dataset: UNSW-NB15 official train/test split.
- Task: binary normal-versus-attack classification.
- Seeds: 11, 29, and 47.
- Communication rounds: 30; held-out-family rounds: 20.
- Client counts: 4, 20, and 50.
- Dirichlet concentration: 0.1, 0.3, and 1.0.
- Malicious-client fraction: 20% in poisoning experiments.
- Attacks: label flip, sign flip/scaling, and fixed-feature backdoor.

Pseudo-clients are statistical partitions of one dataset rather than independently operated edge sites. The family-transfer protocol removes the held-out family after fitting the common preprocessor and is therefore not presented as strict zero-day detection. The code measures SHA-256 hashing and compact JSON construction, not signatures or external-ledger consensus.

## Reproducibility note

The stored CSV files retain the internal key `trust_v2x` because this was the identifier used during the completed runs. The analysis code maps that internal key to the final method name, METRA-FL. Renaming the stored key would change the original experimental records without improving reproducibility.

## License

The source code is released under the MIT License. The UNSW-NB15 dataset remains subject to the terms specified by its provider.
