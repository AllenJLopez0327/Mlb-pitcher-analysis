import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle

# ── Load data ──────────────────────────────────────────────
master = pd.read_csv('Data/Processed/master_dataset.csv')
master['LOB%'] = pd.to_numeric(master['LOB%'].astype(str).str.replace('%', ''), errors='coerce') / 100
snub_df = pd.read_csv('Notebooks/dashboard/snubs.csv')
archetypes_df = pd.read_csv('Notebooks/dashboard/archetypes.csv')

with open('Notebooks/dashboard/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

tree_features = ['xERA', 'SIERA', 'WAR', 'K%', 'HR/9', 'BB%',
                 'K-BB%', 'K/BB', 'WHIP', 'BABIP', 'LOB%',
                 'ERA-', 'FIP-', 'IP', 'W', 'L']

# ── App ────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MLB Cy Young Predictor"

COLORS = {'bg': '#0a0e1a', 'card': '#141824', 'accent': '#3b82f6',
          'gold': '#f59e0b', 'text': '#e2e8f0', 'muted': '#64748b'}

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh',
                              'fontFamily': 'Arial, sans-serif', 'color': COLORS['text']}, children=[

    html.H1("⚾ MLB Pitcher Cy Young Predictor (2015–2025)",
            style={'textAlign': 'center', 'padding': '30px 0 10px',
                   'color': COLORS['gold'], 'fontSize': '28px'}),

    dcc.Tabs(id='tabs', value='tab-1',
             colors={'border': COLORS['card'], 'primary': COLORS['accent'],
                     'background': COLORS['card']},
             children=[
                dcc.Tab(label='⚾ Pitcher Archetypes', value='tab-1',
                        style={'color': COLORS['muted']},
                        selected_style={'color': COLORS['text'], 'backgroundColor': COLORS['bg']}),
                dcc.Tab(label='🏆 Cy Young Predictor', value='tab-2',
                        style={'color': COLORS['muted']},
                        selected_style={'color': COLORS['text'], 'backgroundColor': COLORS['bg']}),
                dcc.Tab(label='📊 Historical Snubs', value='tab-3',
                        style={'color': COLORS['muted']},
                        selected_style={'color': COLORS['text'], 'backgroundColor': COLORS['bg']}),
                dcc.Tab(label='🔥 2025 Leaderboard', value='tab-4',
                        style={'color': COLORS['muted']},
                        selected_style={'color': COLORS['text'], 'backgroundColor': COLORS['bg']}),
             ]),

    html.Div(id='tab-content', style={'padding': '20px'})
])

# ── Tab 1: Archetypes ───────────────────────────────────────
def render_archetypes():
    color_map = {
        'Standard Power Starter': '#3b82f6',
        'Finesse/Soft Tosser': '#22c55e',
        'Knuckle Curve Specialist': '#f59e0b',
        'Power Splitter/Cutter Arm': '#a855f7'
    }

    fig = px.scatter(archetypes_df, x='PC1', y='PC2', color='Archetype',
                     color_discrete_map=color_map,
                     hover_data=['Name', 'Season', 'Team'],
                     opacity=0.5, title='MLB Pitcher Archetypes (2015–2025)')

    winners = archetypes_df[archetypes_df['CY_winner'] == 1]
    fig.add_trace(go.Scatter(
        x=winners['PC1'], y=winners['PC2'],
        mode='markers+text',
        marker=dict(symbol='star', size=16, color='gold',
                    line=dict(color='white', width=1)),
        text=winners['Name'] + " '" + winners['Season'].astype(str).str[-2:],
        textposition='top center', textfont=dict(size=8, color='white'),
        name='Cy Young Winner', showlegend=True
    ))

    fig.update_layout(
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font_color=COLORS['text'], height=600,
        xaxis_title='PC1 — Splitter/Changeup Reliance',
        yaxis_title='PC2 — Breaking Ball Reliance'
    )

    return html.Div([
        html.H3('Pitcher Archetype Explorer', style={'color': COLORS['gold']}),
        html.P('Pitchers clustered by arsenal characteristics using PCA + KMeans. Gold stars = Cy Young winners.',
               style={'color': COLORS['muted']}),
        dcc.Graph(figure=fig)
    ])

# ── Tab 2: Cy Young Predictor ───────────────────────────────
def render_predictor():
    seasons = sorted(master['Season'].unique(), reverse=True)
    
    return html.Div([
        html.H3('Cy Young Vote Share Predictor', style={'color': COLORS['gold']}),
        html.P('Select a season and pitcher to see predicted vote share and ranking.',
               style={'color': COLORS['muted']}),
        
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
            html.Div([
                html.Label('Season', style={'color': COLORS['muted']}),
                dcc.Dropdown(id='season-dropdown',
                             options=[{'label': s, 'value': s} for s in seasons],
                             value=2025, clearable=False,
                             style={'backgroundColor': COLORS['card'], 'color': '#000'})
            ], style={'width': '200px'}),
            
            html.Div([
                html.Label('League', style={'color': COLORS['muted']}),
                dcc.Dropdown(id='league-dropdown',
                             options=[{'label': 'AL', 'value': 'AL'},
                                      {'label': 'NL', 'value': 'NL'}],
                             value='AL', clearable=False,
                             style={'backgroundColor': COLORS['card'], 'color': '#000'})
            ], style={'width': '150px'}),
        ]),
        
        html.Div(id='predictor-output')
    ])

@app.callback(
    Output('predictor-output', 'children'),
    Input('season-dropdown', 'value'),
    Input('league-dropdown', 'value')
)
def update_predictor(season, league):
    df = master[(master['Season'] == season) & (master['League'] == league)].copy()
    df = df.dropna(subset=tree_features)
    
    if len(df) == 0:
        return html.P('No data available.', style={'color': COLORS['muted']})
    
    df['Predicted_Share'] = xgb_model.predict(df[tree_features])
    df['Predicted_Share'] = df['Predicted_Share'].clip(0, 1)
    df = df.sort_values('Predicted_Share', ascending=False).head(10)
    df['Rank'] = range(1, len(df) + 1)
    df['Predicted_Share'] = (df['Predicted_Share'] * 100).round(1).astype(str) + '%'
    df['Actual_Share'] = (df['Share'] * 100).round(1).astype(str) + '%'
    df['Winner'] = df['CY_winner'].apply(lambda x: '🏆' if x == 1 else '')
    
    return dash_table.DataTable(
        data=df[['Rank', 'Name', 'Team', 'Predicted_Share', 'Actual_Share', 'Winner']].to_dict('records'),
        columns=[{'name': c, 'id': c} for c in ['Rank', 'Name', 'Team', 'Predicted_Share', 'Actual_Share', 'Winner']],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': 'bold'},
        style_cell={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'padding': '10px'},
        style_data_conditional=[{
            'if': {'filter_query': '{Winner} = "🏆"'},
            'backgroundColor': '#1a2a1a', 'color': COLORS['gold']
        }]
    )

# ── Tab 3: Historical Snubs ─────────────────────────────────
def render_snubs():
    snub_df['Is_Snub'] = snub_df['Snub'].apply(lambda x: 1 if '❌' in str(x) else 0)
    
    fig = px.bar(snub_df, x='Season', y='Model Pred Share',
                 color='Snub', color_discrete_map={'✅ NO': '#22c55e', '❌ YES': '#ef4444'},
                 hover_data=['Actual Winner', 'Model Pick', 'League'],
                 barmode='group', title='Model Pick vs Actual Winner by Season')
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['card'],
        font_color=COLORS['text'], height=400
    )
    
    return html.Div([
        html.H3('Historical Snubs Analysis (2015–2025)', style={'color': COLORS['gold']}),
        html.P(f'Model agrees with voters in {(snub_df["Snub"] == "✅ NO").sum()}/22 races. Potential snubs highlighted in red.',
               style={'color': COLORS['muted']}),
        dcc.Graph(figure=fig),
        html.H4('Full Results Table', style={'color': COLORS['gold'], 'marginTop': '20px'}),
        dash_table.DataTable(
            data=snub_df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in snub_df.columns if c != 'Is_Snub'],
            style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'padding': '10px'},
            style_data_conditional=[{
                'if': {'filter_query': '{Snub} contains "❌"'},
                'backgroundColor': '#2a1a1a', 'color': '#ef4444'
            }]
        )
    ])

# ── Tab 4: 2025 Leaderboard ─────────────────────────────────
def render_leaderboard():
    df_2025 = master[master['Season'] == 2025].copy()
    df_2025 = df_2025.dropna(subset=tree_features)
    df_2025['Predicted_Share'] = xgb_model.predict(df_2025[tree_features])
    df_2025['Predicted_Share'] = df_2025['Predicted_Share'].clip(0, 1)
    
    al = df_2025[df_2025['League'] == 'AL'].sort_values('Predicted_Share', ascending=False).head(10).copy()
    nl = df_2025[df_2025['League'] == 'NL'].sort_values('Predicted_Share', ascending=False).head(10).copy()
    
    al['Rank'] = range(1, len(al) + 1)
    nl['Rank'] = range(1, len(nl) + 1)
    al['Predicted_Share'] = (al['Predicted_Share'] * 100).round(1).astype(str) + '%'
    nl['Predicted_Share'] = (nl['Predicted_Share'] * 100).round(1).astype(str) + '%'
    
    table_style = {
        'style_header': {'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': 'bold'},
        'style_cell': {'backgroundColor': COLORS['card'], 'color': COLORS['text'], 'padding': '10px'}
    }
    
    return html.Div([
        html.H3('2025 Cy Young Leaderboard', style={'color': COLORS['gold']}),
        html.P('Model predictions for the 2025 season based on current stats.',
               style={'color': COLORS['muted']}),
        
        html.Div(style={'display': 'flex', 'gap': '40px'}, children=[
            html.Div(style={'flex': 1}, children=[
                html.H4('🏆 AL Cy Young Race', style={'color': COLORS['gold']}),
                dash_table.DataTable(
                    data=al[['Rank', 'Name', 'Team', 'Predicted_Share']].to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in ['Rank', 'Name', 'Team', 'Predicted_Share']],
                    **table_style
                )
            ]),
            html.Div(style={'flex': 1}, children=[
                html.H4('🏆 NL Cy Young Race', style={'color': COLORS['gold']}),
                dash_table.DataTable(
                    data=nl[['Rank', 'Name', 'Team', 'Predicted_Share']].to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in ['Rank', 'Name', 'Team', 'Predicted_Share']],
                    **table_style
                )
            ])
        ])
    ])

# ── Main callback ───────────────────────────────────────────
@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-1':
        return render_archetypes()
    elif tab == 'tab-2':
        return render_predictor()
    elif tab == 'tab-3':
        return render_snubs()
    elif tab == 'tab-4':
        return render_leaderboard()

if __name__ == '__main__':
    app.run(debug=True)