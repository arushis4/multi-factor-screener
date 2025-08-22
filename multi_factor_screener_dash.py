# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go

# -----------------------------
# Functions
# -----------------------------
def fetch_stock_data(tickers, period="5y"):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            data[ticker] = {"history": hist, "info": info}
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return data

def calculate_ratios(stock_info):
    ratios = {}
    for ticker, info in stock_info.items():
        ratios[ticker] = {
            "PE": info.get("trailingPE", np.nan),
            "PB": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "RevenueGrowth": info.get("revenueGrowth", np.nan)
        }
    return pd.DataFrame(ratios).T

def score_stocks(df, weights=None):
    df = df.copy()

    # z-score standardization
    df['PE_z'] = (df['PE'] - df['PE'].mean()) / df['PE'].std() * -1
    df['PB_z'] = (df['PB'] - df['PB'].mean()) / df['PB'].std() * -1
    df['ROE_z'] = (df['ROE'] - df['ROE'].mean()) / df['ROE'].std()
    df['Rev_z'] = (df['RevenueGrowth'] - df['RevenueGrowth'].mean()) / df['RevenueGrowth'].std()

    # default weights
    if weights is None:
        weights = {"PE_z":0.15, "PB_z":0.15, "ROE_z":0.35, "Rev_z":0.35}

    # normalize if sum != 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k,v in weights.items()}

    # calculate total score
    df['Total Score'] = sum(df[factor] * weight for factor, weight in weights.items())

    return df.sort_values('Total Score', ascending=False)

def plot_stock_price(history_df, ticker):
    return go.Scatter(x=history_df.index, y=history_df['Close'], mode='lines', name=ticker)

# -----------------------------
# Dash App Layout
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Multi-Factor Stock Screener", style={"textAlign":"center", "color":"#000000"}),

    html.Div([
        html.Label("Enter tickers separated by commas:"),
        dcc.Input(id="tickers-input", type="text", value="", style={"width":"50%", "marginTop": "10px", "marginLeft": "10px", "marginBottom": "10px"})
    ], style={"display":"flex", "alignItems":"center", "marginTop":"30px", "marginBottom": "10px"}),

    html.Div([
        html.Label("Allocate metric weights (sum ≤ 100%)"),
        html.Div([
            html.Label("P/E:"),
            dcc.Input(id="weight-pe", type="number", value=25, min=0, max=100, step=1, style={"width":"60px"}),
            html.Label("P/B:"),
            dcc.Input(id="weight-pb", type="number", value=25, min=0, max=100, step=1, style={"width":"60px"}),
            html.Label("ROE:"),
            dcc.Input(id="weight-roe", type="number", value=25, min=0, max=100, step=1, style={"width":"60px"}),
            html.Label("Revenue Growth:"),
            dcc.Input(id="weight-rev", type="number", value=25, min=0, max=100, step=1, style={"width":"60px"})
        ], style={"display":"flex", "gap":"15px", "alignItems":"center"}),

        html.Div(id="weights-sum-output", style={"marginTop":"10px", "color":"#FF3333", "fontWeight":"bold"})
    ]),

    html.Div([
        html.Label("Select price chart timeframe:"),
        dcc.Dropdown(
            id="chart-timeframe",
            options=[
                {"label": "1D", "value": "1d"},
                {"label": "5D", "value": "5d"},
                {"label": "1M", "value": "1mo"},
                {"label": "6M", "value": "6mo"},
                {"label": "YTD", "value": "ytd"},
                {"label": "1Y", "value": "1y"},
                {"label": "5Y", "value": "5y"},
                {"label": "Max", "value": "max"}
            ],
            value="5y",
            style={"width":"200px"}
        )
    ], style={"marginTop":"20px"}),

    html.Button("Run", id="run-button", n_clicks=0, style={"marginTop":"20px"}),

    dcc.Graph(id="stock-price-graph"),

    html.H2("Top-Ranked Stocks"),
    html.Div(id="scored-table")
])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    [Output("stock-price-graph", "figure"),
     Output("scored-table", "children"),
     Output("weights-sum-output", "children")],
    [Input("run-button", "n_clicks"),
     Input("tickers-input", "value"),
     Input("weight-pe", "value"),
     Input("weight-pb", "value"),
     Input("weight-roe", "value"),
     Input("weight-rev", "value"),
     Input("chart-timeframe", "value")]
)
def update_screener(n_clicks, tickers_input, w_pe, w_pb, w_roe, w_rev, chart_timeframe):
    # Handle missing tickers
    if not tickers_input:
        return {}, "No tickers entered.", ""

    # Clean tickers list
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) == 0:
        return {}, "No valid tickers entered.", ""

    if len(tickers) > 30:
        return {}, "Please enter 30 or fewer tickers at a time for performance reasons.", ""

    # Validate and sanitize weights
    w_pe = float(w_pe or 0)
    w_pb = float(w_pb or 0)
    w_roe = float(w_roe or 0)
    w_rev = float(w_rev or 0)

    total_weight = w_pe + w_pb + w_roe + w_rev
    if total_weight > 100:
        return {}, "⚠️ Sum of weights exceeds 100%! Please adjust.", f"⚠️ Sum of weights is {total_weight}%, which exceeds 100%!"

    # Normalize weights to sum to 1
    if total_weight == 0:
        weights = {"PE_z":0.15, "PB_z":0.15, "ROE_z":0.35, "Rev_z":0.35}
        weights_sum_message = "Using default weights (sum = 100%)."
    else:
        weights = {
            "PE_z": w_pe / total_weight,
            "PB_z": w_pb / total_weight,
            "ROE_z": w_roe / total_weight,
            "Rev_z": w_rev / total_weight
        }
        weights_sum_message = f"Using weights: PE={w_pe}%, PB={w_pb}%, ROE={w_roe}%, Revenue Growth={w_rev}%."

    # Fetch data
    data = fetch_stock_data(tickers, period=chart_timeframe)
    ratios_df = calculate_ratios({t:data[t]["info"] for t in tickers if t in data})
    if ratios_df.empty:
        return {}, "No data available for entered tickers.", weights_sum_message

    # Score stocks
    scored_df = score_stocks(ratios_df, weights=weights)

    # Prepare display table
    display_df = scored_df.drop(columns=['PE_z','PB_z','ROE_z','Rev_z']).copy()
    display_df['Total Score'] = display_df['Total Score'].round(2)
    display_df['Rank'] = display_df['Total Score'].rank(ascending=False, method='min').astype(int)
    display_df[['PE', 'PB', 'ROE', 'RevenueGrowth']] = display_df[['PE', 'PB', 'ROE', 'RevenueGrowth']].round(3)
    display_df = display_df.rename(columns={
        "PE": "P/E",
        "PB": "P/B",
        "ROE": "Return on Equity",
        "RevenueGrowth": "Revenue Growth"
    })
    cols = ['Rank'] + [c for c in display_df.columns if c != 'Rank']
    display_df = display_df[cols]

    # Create stock price figure
    traces = [plot_stock_price(data[t]["history"], t) for t in tickers if t in data]
    figure = go.Figure(traces)
    figure.update_layout(title="Stock Price History", xaxis_title="Date", yaxis_title="Price")

    # DataTable
    table_html = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in display_df.reset_index().rename(columns={"index":"Ticker"}).columns],
        data=display_df.reset_index().rename(columns={"index":"Ticker"}).to_dict('records'),
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1e1e1e", "color": "#39FF14"},
        style_cell={"backgroundColor": "#000000", "color": "#FFFFFF", "textAlign": "center"},
    )

    # Explanatory note
    note = html.Div(
        f"Metrics (P/E, P/B, ROE, Revenue Growth) were standardized using z-scores. {weights_sum_message} Higher Total Score indicates better stock.",
        style={"color":"#555555", "fontStyle":"italic", "marginTop":"10px", "textAlign":"center"}
    )

    return figure, html.Div([table_html, note]), f"Sum of weights: {total_weight}%"

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8050)

# For deployment
server = app.server

"""
# Additions
*   A way to add more advanced analysis
*   Aesthetics & app like experience
"""
