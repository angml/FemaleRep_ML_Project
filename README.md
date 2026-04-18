# Predicting Film Genre from Female Representation Features

**DS 4400**  
**Team:** Mahika Sharma & Angelia Loo

Using over 20,000 films collected from the TMDb API (2005–2024), we explore whether female representation features (cast gender composition, director gender, and crew demographics) can predict a film's primary genre. We train and compare four ML models: Logistic Regression, MLP, Random Forest, and SVM.

## Dataset

- **Source:** TMDb API
- **Size:** 20,019 films
- **Label:** Primary genre (5 classes: Action, Comedy, Drama, Horror, Romance)
- **Features:** Lead gender, director gender, % female cast, % female crew (writers/producers), lead/director age, release year, vote average, popularity

## How to Run

1. Ensure `CSV_FOR_MODEL.csv` is in the same directory
2. Run `Models.ipynb` to train and evaluate all four models

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch requests tqdm
```
