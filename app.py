import streamlit as st
from PIL import Image
import pandas as pd
import datetime
from tennis_model import project_match, monte_carlo_match_sim, format_projection_table, refresh_data

DAILY_CSV_PATH = '/Users/derektmathews/Documents/daily_matchups.csv'

st.set_page_config(page_title="Deke's Match Projections 🎾", page_icon="🎾")
# Load and display logo and title
col1, col2 = st.columns([1, 4])
with col1:
    logo = Image.open("Topspin.png")
    st.image(logo, width=80)
with col2:
    st.markdown("<h1 style='padding-top: 20px;'>Topspin Analytics</h1>", unsafe_allow_html=True)



if 'last_refresh' not in st.session_state or st.session_state['last_refresh'].date() < datetime.datetime.now().date():
    with st.spinner('Refreshing Elo and surface data...'):
        refresh_data()
        st.session_state['last_refresh'] = datetime.datetime.now()

st.title("Deke's Match Projections 🎾")

@st.cache_data(show_spinner=False)
def load_daily_matchups():
    try:
        return pd.read_csv(DAILY_CSV_PATH)
    except Exception as e:
        st.error(f"Error loading daily matchups: {e}")
        return pd.DataFrame()

if st.sidebar.button("🔄 Refresh Daily Matchups"):
    load_daily_matchups.clear()
    st.success("Daily matchups refreshed!")

daily_matchups = load_daily_matchups()

st.sidebar.header("Select from Today's Matchups")
selected_match = None
if not daily_matchups.empty:
    matchup_labels = [f"{row['Player1']} vs {row['Player2']} ({row['Surface']})" for _, row in daily_matchups.iterrows()]
    selection = st.sidebar.selectbox("Choose a matchup:", options=[""] + matchup_labels)

    if selection:
        selected_match = daily_matchups.iloc[matchup_labels.index(selection)]
        player1 = selected_match['Player1']
        player2 = selected_match['Player2']
        surface = selected_match['Surface']
    else:
        player1 = st.sidebar.text_input("Player 1 Name")
        player2 = st.sidebar.text_input("Player 2 Name")
        surface = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass"])
else:
    player1 = st.sidebar.text_input("Player 1 Name")
    player2 = st.sidebar.text_input("Player 2 Name")
    surface = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass"])

best_of = st.sidebar.radio("Best Of", [3, 5])

if st.sidebar.button("Run Projection"):
    if player1 and player2:
        base_result = project_match(player1, player2, surface)
        mc_result = monte_carlo_match_sim(player1, player2, surface, best_of=best_of)
        result = {**base_result, **mc_result}

        player_df, shared_df = format_projection_table(result)

        for metric in ['Surface Elo', 'Combined Rating']:
            for col in [player1, player2]:
                player_df.loc[player_df['Metric'] == metric, col] = pd.to_numeric(
                    player_df.loc[player_df['Metric'] == metric, col], errors='coerce').round(1)

        spread = round(result['projected_spread'], 1)
        fav_player = player1 if result['p1_win_prob'] > result['p2_win_prob'] else player2
        shared_df.loc[shared_df['Metric'] == 'Projected Spread', 'Value'] = f"{fav_player} -{spread}"

        shared_df.loc[shared_df['Metric'] == 'Projected Total Games', 'Value'] = round(result['projected_total_games'], 1)

        mc_probs = result['MC_set_score_probs']
        sorted_mc_probs = sorted(mc_probs.items(), key=lambda x: x[1], reverse=True)
        mc_prob_lines = [f"{player1 if k in ['2-0', '2-1'] else player2} {k}: {v:.3f}" for k, v in sorted_mc_probs]
        mc_prob_formatted = '\n'.join(mc_prob_lines)
        shared_df.loc[shared_df['Metric'] == 'MC Set Score Probabilities', 'Value'] = mc_prob_formatted

        def highlight_elo_prob(row):
            return ['background-color: teal; color: white'] * len(row) if row['Metric'] == 'Win Probability (Elo)' else [''] * len(row)

        # Add Elo Confidence (Stars)
        p1_prob = result['p1_win_prob'] * 100
        p2_prob = result['p2_win_prob'] * 100
        max_prob = max(p1_prob, p2_prob)

        if max_prob >= 90:
            stars = '⭐⭐⭐⭐⭐'
        elif max_prob >= 80:
            stars = '⭐⭐⭐⭐'
        elif max_prob >= 70:
            stars = '⭐⭐⭐'
        elif max_prob >= 60:
            stars = '⭐⭐'
        elif max_prob >= 50:
            stars = '⭐'
        else:
            stars = ''

        confidence_row = {'Metric': 'Elo Confidence', player1: stars if p1_prob > p2_prob else '', player2: stars if p2_prob > p1_prob else ''}
        player_df = pd.concat([player_df, pd.DataFrame([confidence_row])], ignore_index=True)

        st.subheader("=== Player Comparison ===")
        st.table(player_df.style.apply(highlight_elo_prob, axis=1))

        st.subheader("=== Model Projections ===")
        st.table(shared_df.style.set_properties(**{'white-space': 'pre-wrap'}))

    else:
        st.warning("Please enter both player names to run projection.")
else:
    st.info("Enter match details and click 'Run Projection' to see results.")