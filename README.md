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


## Dashboard (Bonus +5%)
We are building an interactive Plotly Dash web app with 4 tabs:
- **Tab 1: Pitcher Archetype Explorer** — PCA plot, select pitcher/year, see their archetype
- **Tab 2: Cy Young Predictor** — input stats, get predicted vote share and ranking
- **Tab 3: Historical Snubs** — model winner vs actual winner 2015–2025
- **Tab 4: 2025 Live Leaderboard** — current AL/NL Cy Young race

## Data
All raw CSVs are gitignored. Download instructions are in the README. 
Share CSVs with teammates via Google Drive or iMessage since they can't be pushed to GitHub.

## Getting Started for Teammates
1. Clone the repo
2. Get the raw CSVs from Allen and place in `Data/Raw/`
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order starting with `01_Data_Collection.ipynb`

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