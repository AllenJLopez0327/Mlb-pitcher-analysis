# MLB Pitcher Cy Young Predictor (2015–2025)

> IEOR 142A Spring 2026 — Team Project

## Overview

A two-part machine learning pipeline that:
1. Classifies MLB starting pitchers into archetypes using arsenal characteristics (PCA + clustering)
2. Predicts Cy Young Award vote share and winners using performance metrics (regression + classification + SHAP)

**Core question:** Can how a pitcher throws predict how well they perform — and do voters actually reward the best performers?

## Project Structure
Mlb-pitcher-analysis/
├── Data/
│   ├── Raw/          # Downloaded CSVs (gitignored — see below)
│   └── Processed/    # Cleaned and merged master dataset
├── Notebooks/        # Jupyter notebooks (run in order)
├── Outputs/          # Saved figures and results
├── src/              # Reusable Python scripts
└── requirements.txt

## Data Sources

All raw CSVs are gitignored due to size. Download and place in `Data/Raw/`:

| File | Source | Instructions |
|------|--------|--------------|
| `fangraphs-Stats.csv` | FanGraphs Major League Leaders | Pitchers, as SP, 50 IP, 2015–2025, Dashboard preset, Split Seasons |
| `fangraphs-Pitch Type.csv` | FanGraphs Pitch-Level Data | Statcast → Pitch Type, same filters |
| `fangraphs-Velocity.csv` | FanGraphs Pitch-Level Data | Statcast → Velocity, same filters |
| `fangraphs-H-Movement.csv` | FanGraphs Pitch-Level Data | Statcast → H-Movement, same filters |
| `fangraphs-V-Movement.csv` | FanGraphs Pitch-Level Data | Statcast → V-Movement, same filters |
| `fangraphs-Spin.csv` | FanGraphs Pitch-Level Data | Statcast → Spin, same filters |
| `fangraphs-Arm Angel.csv` | FanGraphs Pitch-Level Data | Statcast → Arm Angle, same filters |
| `CY young data.csv` | Baseball Reference | AL + NL Cy Young voting, 2015–2025, compiled manually |
| `Edge Cases Relievers.csv` | FanGraphs | 6 relievers with 5%+ Cy Young vote share manually added |

## Notebooks (run in order)

- `01_Data_Collection.ipynb` — Load, clean, and merge all data sources into master dataset
- More coming soon...

## Setup

```bash
git clone https://github.com/AllenJLopez0327/Mlb-pitcher-analysis.git
cd Mlb-pitcher-analysis
pip install -r requirements.txt
jupyter notebook
```

## Team
- Allen Lopez
- Abdulrahman Al-Ghlai
- Aniketh Eswara
- Ian Alexis Mendoza Juarez
- Liam Cameron Yung
- William Phillip Deutchman