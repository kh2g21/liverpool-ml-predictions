# premier-league-ml-predictions
A machine learning project predicting outcomes of Liverpool's matches during the 2021-2022 Premier League season.

## Project Overview
This project aims to predict the outcomes of Liverpool's 2021-2022 Premier League matches using machine learning models. The main goal is to predict the number of goals Liverpool scores and concedes using match statistics, team data, and various features like shooting, possession, and defensive actions.

Two different machine learning models are used:
- **Random Forest Regressor** for predicting Liverpool's goals.
- **XGBoost Regressor** for predicting the opponent's goals.

## Files in the Project

### 1. `2021-2022.csv`
This CSV file contains the match data for the entire Premier League 2021-2022 season. It includes information such as the teams playing, goals scored, and other basic match statistics (e.g., shots, fouls, yellow cards).

### 2. `england_premier_league_table_22.csv`
This file contains team-level statistics for the entire Premier League season, including goals for, goals against, points, expected goals (xG), and goal difference. This data is used to provide a context for team strength.

### 3. `england_premier_league_stats_22.csv`
This file includes detailed statistics about each team, such as possession percentage, assists, and disciplinary records (yellow/red cards). These stats are used to enrich the feature set for the models.

### 4. `england_premier_league_squad_shooting_22.csv`
This file contains squad-level shooting statistics, including total shots, shots on target, goals per shot, and expected goals (xG). This data is used as input features to improve the goal prediction accuracy.

### 5. `england_premier_league_squad_possession_22.csv`
This file provides information about each team’s average possession percentage over the season. Possession is an important feature, as it often correlates with offensive and defensive performance.

### 6. `england_premier_league_squad_defensive_actions_22.csv`
This file contains defensive stats such as total pressures, successful pressures, and clearances. It helps model the defensive capabilities of teams, contributing to the goal predictions.

### 7. `premierleague_ml.py`
The main script that processes the datasets, trains the machine learning models, and makes predictions for each match. The script performs the following tasks:
- Loads and merges all datasets.
- Creates additional features like goals scored and conceded by Liverpool.
- Splits the data into training and test sets.
- Trains Random Forest and XGBoost models for predicting goals.
- Calculates and prints the Mean Squared Error (MSE) for predicted Liverpool and opponent goals.
- Outputs match-by-match predictions alongside the actual results.

## How to Run the Project

### Prerequisites
- Python 3.x
- Required Python packages:
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `numpy`

### Steps to Run
1. Install the necessary packages:
   ```bash
   pip install pandas scikit-learn xgboost numpy
