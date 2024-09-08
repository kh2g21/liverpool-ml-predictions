import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('2021-2022.csv')
league_table = pd.read_csv('england_premier_league_table_22.csv', sep=';')
stats = pd.read_csv('england_premier_league_stats_22.csv', sep=';')
shooting = pd.read_csv('england_premier_league_squad_shooting_22.csv', sep=';')
possession = pd.read_csv('england_premier_league_squad_possession_22.csv', sep=';')
defensive = pd.read_csv('england_premier_league_squad_defensive_actions_22.csv', sep=';')

# Filter Liverpool matches
liverpool_data = df[(df['HomeTeam'] == 'Liverpool') | (df['AwayTeam'] == 'Liverpool')].copy()

# Create 'LiverpoolGoals' and 'OpponentGoals' columns
liverpool_data['LiverpoolGoals'] = liverpool_data.apply(
    lambda row: row['FTHG'] if row['HomeTeam'] == 'Liverpool' else row['FTAG'], axis=1)

liverpool_data['OpponentGoals'] = liverpool_data.apply(
    lambda row: row['FTAG'] if row['HomeTeam'] == 'Liverpool' else row['FTHG'], axis=1)

# Create mappings for each relevant feature from stats dataset
# Mapping for league table (GF: Goals For, GA: Goals Against, Pts, xG, xGA, GD: Goal Difference)
league_mapping = league_table.set_index('Squad')[['GF', 'GA', 'Pts', 'xG', 'xGA', 'GD']].to_dict(orient='index')

# Mapping for stats (Possession, Assists, Yellow and Red Cards)
stats_mapping = stats.set_index('Squad')[['Poss', 'AstTot', 'CrdYel', 'CrdRed']].to_dict(orient='index')

# Mapping for shooting (Total Shots, Shots on Target, Goals per Shot, xG)
shooting_mapping = shooting.set_index('Squad')[['TotSh', 'TotShTg', 'G;Sh', 'xG']].to_dict(orient='index')

# Mapping for defensive actions (Pressures, Successful Pressures, Clearances)
defensive_mapping = defensive.set_index('Squad')[['PressTot', 'SuccPress', 'Clear']].to_dict(orient='index')

# Function to extract features for a team from the mappings
def get_team_features(team, mapping, default_values):
    return mapping.get(team, default_values)

# Define default values for each set of features (in case the team is missing in the dataset)
default_league = {'GF': 0, 'GA': 0, 'Pts': 0, 'xG': 0, 'xGA': 0, 'GD': 0}
default_stats = {'Poss': 0, 'AstTot': 0, 'CrdYel': 0, 'CrdRed': 0}
default_shooting = {'TotSh': 0, 'TotShTg': 0, 'G;Sh': 0, 'xG': 0}
default_defensive = {'PressTot': 0, 'SuccPress': 0, 'Clear': 0}

# Apply features for each match
for idx, row in liverpool_data.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    # Home team features
    home_league = get_team_features(home_team, league_mapping, default_league)
    home_stats = get_team_features(home_team, stats_mapping, default_stats)
    home_shooting = get_team_features(home_team, shooting_mapping, default_shooting)
    home_defensive = get_team_features(home_team, defensive_mapping, default_defensive)
    
    # Away team features
    away_league = get_team_features(away_team, league_mapping, default_league)
    away_stats = get_team_features(away_team, stats_mapping, default_stats)
    away_shooting = get_team_features(away_team, shooting_mapping, default_shooting)
    away_defensive = get_team_features(away_team, defensive_mapping, default_defensive)
    
    # Add the features to the dataframe
    for key, value in home_league.items():
        liverpool_data.loc[idx, f'Home_{key}'] = value
    for key, value in home_stats.items():
        liverpool_data.loc[idx, f'Home_{key}'] = value
    for key, value in home_shooting.items():
        liverpool_data.loc[idx, f'Home_{key}'] = value
    for key, value in home_defensive.items():
        liverpool_data.loc[idx, f'Home_{key}'] = value
    
    for key, value in away_league.items():
        liverpool_data.loc[idx, f'Away_{key}'] = value
    for key, value in away_stats.items():
        liverpool_data.loc[idx, f'Away_{key}'] = value
    for key, value in away_shooting.items():
        liverpool_data.loc[idx, f'Away_{key}'] = value
    for key, value in away_defensive.items():
        liverpool_data.loc[idx, f'Away_{key}'] = value

# Select relevant features for modeling
features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',  # Match stats
            'Home_GF', 'Home_GA', 'Home_Pts', 'Home_xG', 'Home_xGA', 'Home_GD',  # Home team league features
            'Away_GF', 'Away_GA', 'Away_Pts', 'Away_xG', 'Away_xGA', 'Away_GD',  # Away team league features
            'Home_Poss', 'Home_AstTot', 'Home_CrdYel', 'Home_CrdRed',  # Home team stats
            'Away_Poss', 'Away_AstTot', 'Away_CrdYel', 'Away_CrdRed',  # Away team stats
            'Home_TotSh', 'Home_TotShTg', 'Home_G;Sh', 'Home_PressTot', 'Home_SuccPress', 'Home_Clear',  # Home shooting/defense
            'Away_TotSh', 'Away_TotShTg', 'Away_G;Sh', 'Away_PressTot', 'Away_SuccPress', 'Away_Clear'  # Away shooting/defense
           ]

X = liverpool_data[features]
y_liverpool = liverpool_data['LiverpoolGoals']
y_opponent = liverpool_data['OpponentGoals']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train_liverpool, y_test_liverpool = train_test_split(X, y_liverpool, test_size=0.2, random_state=42)
_, _, y_train_opponent, y_test_opponent = train_test_split(X, y_opponent, test_size=0.2, random_state=42)

# Train the models on the training set only
rf_liverpool = RandomForestRegressor(n_estimators=100, random_state=42)
rf_liverpool.fit(X_train, y_train_liverpool)

xgb_opponent = XGBRegressor(n_estimators=100, random_state=42)
xgb_opponent.fit(X_train, y_train_opponent)

# Make predictions for all matches, including the training and test set
pred_liverpool_goals = np.round(rf_liverpool.predict(X))  # Predict Liverpool goals
pred_opponent_goals = np.round(xgb_opponent.predict(X))    # Predict opponent goals

# Match-by-match predictions for all 38 games (Ensure correct indices)
for i, idx in enumerate(liverpool_data.index):  # Loop through entire dataset by original indices
    actual_liverpool_goals = y_liverpool.iloc[i]
    actual_opponent_goals = y_opponent.iloc[i]
    
    predicted_lfc_goals = int(pred_liverpool_goals[i])  # Predicted Liverpool goals
    predicted_opponent_goals = int(pred_opponent_goals[i])  # Predicted opponent goals

    home_team = liverpool_data.loc[idx, 'HomeTeam']
    away_team = liverpool_data.loc[idx, 'AwayTeam']

    # Print the results in a readable format
    if home_team == 'Liverpool':
        print(f"Match: Liverpool vs {away_team}")
        print(f"Actual Score: {actual_liverpool_goals} - {actual_opponent_goals}")
        print(f"Predicted Score: {predicted_lfc_goals} - {predicted_opponent_goals}\n")
    else:
        print(f"Match: {home_team} vs Liverpool")
        print(f"Actual Score: {actual_opponent_goals} - {actual_liverpool_goals}")
        print(f"Predicted Score: {predicted_opponent_goals} - {predicted_lfc_goals}\n")
