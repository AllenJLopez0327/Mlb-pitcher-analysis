# Mlb-pitcher-analysis

# MLB Pitcher Analysis

A machine learning project analyzing MLB pitcher arsenal data (2020-2025) 
to predict pitcher performance and Cy Young Award winners.

## Project Structure

Mlb-pitcher-analysis/
├── Data/
│   ├── Raw/          # Original datasets
│   └── Processed/    # Cleaned and merged data
├── Notebooks/        # Jupyter notebooks
├── src/              # Reusable Python scripts
├── Outputs/
│   └── figures/      # Saved plots
└── requirements.txt


## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/AllenJLopez0327/Mlb-pitcher-analysis.git
cd Mlb-pitcher-analysis
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

### 4. Open notebooks in order
- `01_Data_Collection.ipynb` - Load and explore datasets
- More coming soon...

## Datasets
- **MLB Pitcher Arsenal 2020-2025** - Statcast pitch metrics (already in Data/Raw/)
- **FanGraphs Pitching Stats** - ERA, FIP, WAR (pulled via pybaseball)
- **Cy Young Voting Results** - Award voting history (pulled via pybaseball)

## Team
- Allen Lopez
- [Teammate 2]
- [Teammate 3]