import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
from glob import glob

# ---- Elo Setup ----
def initialize_elo():
    return 1500

k_factor = 32
elo_ratings = defaultdict(initialize_elo)
elo_by_surface = defaultdict(initialize_elo)
surface_stats = defaultdict(lambda: {'wins':0, 'losses':0})

SURFACE_ELO_WEIGHT = 0.6
SURFACE_WIN_PCT_WEIGHT = 0.3
OVERALL_ELO_WEIGHT = 0.1

sackmann_path = '/Users/derektmathews/Downloads/tennis_atp-master'
custom_data_path = '/Users/derektmathews/Documents/TML-Database'

def expected_score(player_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

def surface_win_pct(player, surface):
    stats = surface_stats.get((player, surface), {'wins':0, 'losses':0})
    total = stats['wins'] + stats['losses']
    return (stats['wins'] / total) if total > 0 else 0

def refresh_data():
    global elo_ratings, elo_by_surface, surface_stats

    historical_files = glob(os.path.join(sackmann_path, 'atp_matches_*.csv'))
    historical_dfs = [pd.read_csv(f, low_memory=False) for f in historical_files]
    historical_data = pd.concat(historical_dfs, ignore_index=True)

    custom_2025 = pd.read_csv(os.path.join(custom_data_path, '2025.csv'))

    common_columns = custom_2025.columns.tolist()
    historical_data_aligned = historical_data[common_columns]
    combined_data = pd.concat([historical_data_aligned, custom_2025], ignore_index=True)

    combined_data['tourney_date'] = pd.to_numeric(combined_data['tourney_date'], errors='coerce')
    combined_data = combined_data.dropna(subset=['tourney_date'])
    combined_data['tourney_date'] = combined_data['tourney_date'].astype(int).astype(str)
    combined_data['tourney_date'] = pd.to_datetime(combined_data['tourney_date'], format='%Y%m%d', errors='coerce')
    combined_data = combined_data.dropna(subset=['tourney_date'])
    combined_data = combined_data.sort_values('tourney_date').reset_index(drop=True)

    # Reset Elo and stats
    elo_ratings = defaultdict(initialize_elo)
    elo_by_surface = defaultdict(initialize_elo)
    surface_stats = defaultdict(lambda: {'wins':0, 'losses':0})

    for idx, match in combined_data.iterrows():
        w_name, l_name = match['winner_name'], match['loser_name']
        surface = match['surface']

        w_elo = elo_ratings[w_name]
        l_elo = elo_ratings[l_name]
        w_surface_elo = elo_by_surface[(w_name, surface)]
        l_surface_elo = elo_by_surface[(l_name, surface)]

        w_exp = expected_score(w_elo, l_elo)
        l_exp = expected_score(l_elo, w_elo)

        elo_ratings[w_name] += k_factor * (1 - w_exp)
        elo_ratings[l_name] += k_factor * (0 - l_exp)

        w_surface_exp = expected_score(w_surface_elo, l_surface_elo)
        l_surface_exp = expected_score(l_surface_elo, w_surface_elo)

        elo_by_surface[(w_name, surface)] += k_factor * (1 - w_surface_exp)
        elo_by_surface[(l_name, surface)] += k_factor * (0 - l_surface_exp)

        surface_stats[(w_name, surface)]['wins'] += 1
        surface_stats[(l_name, surface)]['losses'] += 1


def project_match(player1, player2, surface):
    p1_surface_stats = surface_stats.get((player1, surface), {'wins':0, 'losses':0})
    p2_surface_stats = surface_stats.get((player2, surface), {'wins':0, 'losses':0})

    p1_surface_elo = elo_by_surface.get((player1, surface), 1500)
    p2_surface_elo = elo_by_surface.get((player2, surface), 1500)

    p1_surface_pct = surface_win_pct(player1, surface)
    p2_surface_pct = surface_win_pct(player2, surface)

    p1_elo = elo_ratings[player1]
    p2_elo = elo_ratings[player2]

    p1_combined = (SURFACE_ELO_WEIGHT * p1_surface_elo) + \
                  (OVERALL_ELO_WEIGHT * p1_elo) + \
                  (SURFACE_WIN_PCT_WEIGHT * (p1_surface_pct * 1000))

    p2_combined = (SURFACE_ELO_WEIGHT * p2_surface_elo) + \
                  (OVERALL_ELO_WEIGHT * p2_elo) + \
                  (SURFACE_WIN_PCT_WEIGHT * (p2_surface_pct * 1000))

    p1_win_prob = expected_score(p1_combined, p2_combined)
    p2_win_prob = 1 - p1_win_prob

    spread = round((p1_win_prob - p2_win_prob) * 6, 1)
    total_games = round(20 + (10 * abs(p1_win_prob - 0.5)), 1)

    return {
        'player1': player1,
        'player2': player2,
        'surface': surface,
        'p1_surface_wins': p1_surface_stats['wins'],
        'p1_surface_losses': p1_surface_stats['losses'],
        'p1_surface_elo': round(p1_surface_elo, 1),
        'p1_surface_win_pct': round(p1_surface_pct, 3),
        'p2_surface_wins': p2_surface_stats['wins'],
        'p2_surface_losses': p2_surface_stats['losses'],
        'p2_surface_elo': round(p2_surface_elo, 1),
        'p2_surface_win_pct': round(p2_surface_pct, 3),
        'p1_combined_rating': round(p1_combined, 2),
        'p2_combined_rating': round(p2_combined, 2),
        'p1_win_prob': round(p1_win_prob, 3),
        'p2_win_prob': round(p2_win_prob, 3),
        'projected_spread': spread,
        'projected_total_games': total_games
    }


def monte_carlo_match_sim(player1, player2, surface, best_of=3, n_simulations=10000):
    base_projection = project_match(player1, player2, surface)
    p1_prob, p2_prob = base_projection['p1_win_prob'], base_projection['p2_win_prob']

    set_probs = []
    for _ in range(n_simulations):
        p1_sets, p2_sets = 0, 0
        while p1_sets < (best_of // 2 + 1) and p2_sets < (best_of // 2 + 1):
            winner = np.random.choice([1,2], p=[p1_prob, p2_prob])
            if winner == 1:
                p1_sets +=1
            else:
                p2_sets +=1
        set_probs.append((p1_sets, p2_sets))

    score_counts = Counter(set_probs)
    set_score_probs = {f'{k[0]}-{k[1]}': round(v / n_simulations, 3) for k, v in score_counts.items()}

    return {
        'MC_set_score_probs': set_score_probs
    }


def format_projection_table(result):
    player1 = result['player1']
    player2 = result['player2']

    player_metrics = [
        ('Surface Wins', int(result['p1_surface_wins']), int(result['p2_surface_wins'])),
        ('Surface Losses', int(result['p1_surface_losses']), int(result['p2_surface_losses'])),
        ('Surface Elo', round(result['p1_surface_elo'], 1), round(result['p2_surface_elo'], 1)),
        ('Surface Win %', f"{round(result['p1_surface_win_pct']*100,1)}%", f"{round(result['p2_surface_win_pct']*100,1)}%"),
        ('Combined Rating', round(result['p1_combined_rating'], 2), round(result['p2_combined_rating'], 2)),
        ('Win Probability (Elo)', f"{round(result['p1_win_prob']*100,1)}%", f"{round(result['p2_win_prob']*100,1)}%")
    ]

    player_df = pd.DataFrame(player_metrics, columns=['Metric', player1, player2])

    shared_metrics = {
        'Surface': result['surface'],
        'Projected Spread': result['projected_spread'],
        'Projected Total Games': result['projected_total_games'],
        'MC Set Score Probabilities': result['MC_set_score_probs']
    }

    shared_df = pd.DataFrame(shared_metrics.items(), columns=['Metric', 'Value'])

    return player_df, shared_df
