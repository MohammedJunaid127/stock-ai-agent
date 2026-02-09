#
!pip install --quiet "urllib3==1.26.16" "requests==2.31.0" "six==1.16.0"
!pip install --quiet yfinance pandas numpy openai vaderSentiment plotly gradio

import os
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import gradio as gr

try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

CONFIG = {
    "NEWS_API": "NEWSAPI",
    "NEWS_API_KEY": os.getenv("NEWS_API_KEY", ""),
    "LLM": "openai",
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "MAX_RISK_PER_TRADE_PCT": 0.02,
}
if client and CONFIG.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]

analyzer = SentimentIntensityAnalyzer()
paper_portfolio = {"cash": 10000.0, "positions": {}, "history": []}

def fetch_price_history(symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            df.columns = ["_".join(map(str, c)) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        raise ValueError(f"No valid Close prices for {symbol}")
    return df

def fetch_news_newsapi(query: str, page_size: int = 10):
    key = CONFIG.get("NEWS_API_KEY", "")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": page_size, "language": "en", "sortBy": "publishedAt"}
    headers = {"X-Api-Key": key}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []

def fetch_news(symbol: str, n: int = 10):
    return fetch_news_newsapi(symbol, n)

def analyze_news_sentiment_vader(articles):
    scores = []
    for a in (articles or []):
        title = a.get("title") or ""
        desc = a.get("description") or ""
        text = f"{title} {desc}"
        scores.append(analyzer.polarity_scores(text)["compound"])
    return float(np.mean(scores)) if scores else 0.0

def analyze_news_with_llm(articles, symbol: str = ""):
    if not client:
        return analyze_news_sentiment_vader(articles)
    combined = "\n\n".join([f"{(a.get('title') or '').strip()} - {(a.get('description') or '').strip()}" for a in (articles or [])[:6]])
    prompt = (
        "You are a financial sentiment evaluator. Rate overall sentiment from -1 (bearish) to 1 (bullish). "
        "Return JSON with keys: score (number), reasoning (string).\n\n" + combined
    )
    try:
        resp = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200)
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        score = data.get("score")
        try:
            return float(score)
        except Exception:
            return analyze_news_sentiment_vader(articles)
    except Exception:
        return analyze_news_sentiment_vader(articles)

def position_size_cash(amount_usd: float, price: float, risk_pct: float) -> int:
    try:
        price = float(price)
    except Exception:
        return 0
    if price <= 0:
        return 0
    alloc = amount_usd * risk_pct
    return max(0, int(alloc // price))


def agent_decision(symbol: str, user_amount_usd: float):
    df = fetch_price_history(symbol, period="30d")
    last_close = float(df["Close"].iloc[-1])
    short_ma = float(df["Close"].tail(5).mean())
    long_ma = float(df["Close"].tail(20).mean()) if len(df) >= 20 else short_ma
    tech_signal = "bullish" if short_ma > long_ma else "bearish"
    articles = fetch_news(symbol, 6)
    sentiment = analyze_news_with_llm(articles, symbol) if CONFIG.get("LLM") == "openai" and CONFIG.get("OPENAI_API_KEY") else analyze_news_sentiment_vader(articles)
    tech_score = 1.0 if tech_signal == "bullish" else -1.0
    combined = 0.6 * tech_score + 0.4 * sentiment
    combined = float(max(-1.0, min(1.0, combined)))
    if combined > 0.35:
        action = "buy"
    elif combined < -0.35:
        action = "sell"
    else:
        action = "hold"
    shares = position_size_cash(user_amount_usd, last_close, CONFIG.get("MAX_RISK_PER_TRADE_PCT", 0.02)) if action == "buy" else 0
    reasoning = (
        f"AI Trading Analysis for {symbol}\n"
        f"Last Price: ${last_close:.2f}\n"
        f"Short MA (5): {short_ma:.2f}\n"
        f"Long MA (20): {long_ma:.2f}\n"
        f"Technical Bias: {tech_signal}\n"
        f"News Sentiment Score: {sentiment:.3f}\n"
        f"Combined Score: {combined:.3f}\n"
        f"Decision: {action.upper()}\n"
        f"Shares Suggested: {shares}\n"
    )
    return {"symbol": symbol, "action": action, "shares": shares, "price": last_close, "sentiment": sentiment, "combined_score": combined, "tech_signal": tech_signal, "reasoning": reasoning, "articles": articles[:3]}


def make_chart(symbol: str, period: str = "60d"):
    try:
        df = fetch_price_history(symbol, period=period)
    except Exception:
        return go.Figure()
    if df is None or df.empty or "Close" not in df.columns:
        return go.Figure()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index.tolist(), y=df["Close"].astype(float).tolist(), mode="lines", name="Close"))
    if len(df) >= 5:
        ma5 = df["Close"].rolling(5).mean().dropna()
        fig.add_trace(go.Scatter(x=ma5.index.tolist(), y=ma5.astype(float).tolist(), mode="lines", name="MA5"))
    if len(df) >= 20:
        ma20 = df["Close"].rolling(20).mean().dropna()
        fig.add_trace(go.Scatter(x=ma20.index.tolist(), y=ma20.astype(float).tolist(), mode="lines", name="MA20"))
    fig.update_layout(title=f"{symbol} Price", xaxis_title="Date", yaxis_title="Price", height=450)
    return fig

def news_markdown(symbol: str, n: int = 5):
    arts = fetch_news(symbol, n) or []
    blocks = []
    for a in arts:
        title = str(a.get("title") or "No title").strip()
        desc = str(a.get("description") or "").strip()
        url = str(a.get("url") or "").strip()
        if url:
            blocks.append(f"### [{title}]({url})\n\n{desc}")
        else:
            blocks.append(f"### {title}\n\n{desc}")
    return "\n\n---\n\n".join(blocks) if blocks else "No recent articles found."

def portfolio_summary_text():
    total = 0.0
    rows = []
    for sym, qty in paper_portfolio.get("positions", {}).items():
        price = None
        try:
            df = fetch_price_history(sym, period="5d")
            price = float(df["Close"].iloc[-1])
        except Exception:
            price = None
        value = price * qty if price is not None else None
        if value is not None:
            total += value
        rows.append({"symbol": sym, "shares": qty, "last_price": price, "value": value})
    equity = total + paper_portfolio.get("cash", 0.0)
    return {"equity": round(equity, 2), "cash": round(paper_portfolio.get("cash", 0.0), 2), "positions": rows}

def execute_trade(symbol: str, side: str, shares: int):
    symbol = symbol.upper().strip()
    try:
        shares = int(shares)
    except Exception:
        return {"status": "error", "message": "Invalid shares"}
    if shares <= 0:
        return {"status": "error", "message": "Shares must be > 0"}
    try:
        df = fetch_price_history(symbol, period="5d")
        price = float(df["Close"].iloc[-1])
    except Exception as e:
        return {"status": "error", "message": f"Price fetch failed: {e}"}
    if side == "buy":
        cost = shares * price
        if cost > paper_portfolio.get("cash", 0):
            return {"status": "error", "message": "Insufficient cash"}
        paper_portfolio["cash"] -= cost
        paper_portfolio["positions"][symbol] = paper_portfolio["positions"].get(symbol, 0) + shares
        paper_portfolio["history"].append({"time": str(pd.Timestamp.now()), "symbol": symbol, "side": "buy", "price": price, "shares": shares})
        return {"status": "filled", "symbol": symbol, "shares": shares, "price": price}
    elif side == "sell":
        held = paper_portfolio["positions"].get(symbol, 0)
        if shares > held:
            return {"status": "error", "message": "Not enough shares to sell"}
        paper_portfolio["positions"][symbol] = held - shares
        proceeds = shares * price
        paper_portfolio["cash"] += proceeds
        paper_portfolio["history"].append({"time": str(pd.Timestamp.now()), "symbol": symbol, "side": "sell", "price": price, "shares": shares})
        return {"status": "filled", "symbol": symbol, "shares": shares, "price": price}
    else:
        return {"status": "error", "message": "Unknown side"}

def run_agent_ui(symbol: str, cash: float):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"error": "Symbol is required."}, go.Figure(), "No news.", "", portfolio_summary_text()
    try:
        decision = agent_decision(symbol, float(cash))
    except Exception as e:
        return {"error": f"Decision failed: {e}"}, go.Figure(), "No news.", "", portfolio_summary_text()
    explanation = decision.get("reasoning")
    chart = make_chart(symbol)
    news_md = news_markdown(symbol)
    port = portfolio_summary_text()
    return decision, chart, news_md, explanation, port

app = gr.Blocks()

with app:
    gr.Markdown("# AI Stock Agent – Enhanced Dashboard")
    with gr.Row():
        inp_symbol = gr.Textbox(label="Stock Symbol", value="AAPL")
        inp_cash = gr.Number(label="Available Cash", value=10000.0)
        inp_shares = gr.Number(label="Manual Shares", value=1)
    btn_run = gr.Button("Run Agent")
    btn_buy = gr.Button("Buy")
    btn_sell = gr.Button("Sell")
    trade_output = gr.JSON(label="Trade Result")
    with gr.Tabs():
        with gr.TabItem("Decision"):
            out_decision = gr.JSON(label="Agent Decision")
            out_explain = gr.Markdown(label="LLM Explanation")
        with gr.TabItem("Chart"):
            out_chart = gr.Plot(label="Price Chart")
        with gr.TabItem("News"):
            out_news = gr.Markdown(label="Latest News")
        with gr.TabItem("Portfolio"):
            out_port = gr.JSON(label="Portfolio Summary")
    btn_run.click(run_agent_ui, inputs=[inp_symbol, inp_cash], outputs=[out_decision, out_chart, out_news, out_explain, out_port])
    def buy_click(symbol, shares):
        return execute_trade(symbol, "buy", int(shares)), portfolio_summary_text()
    def sell_click(symbol, shares):
        return execute_trade(symbol, "sell", int(shares)), portfolio_summary_text()
    btn_buy.click(fn=buy_click, inputs=[inp_symbol, inp_shares], outputs=[trade_output, out_port])
    btn_sell.click(fn=sell_click, inputs=[inp_symbol, inp_shares], outputs=[trade_output, out_port])

app.launch(share=True)
