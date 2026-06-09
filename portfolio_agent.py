import yfinance as yf
import pandas as pd
from typing import TypedDict, Optional, Dict, Any, List, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import vertexai
from vertexai.generative_models import GenerativeModel

from backtest import run_backtest_simulation

class WorkerResult(TypedDict):
    ticker: str
    result: Dict[str, Any]

class SupervisorState(TypedDict):
    tickers: List[str]
    start_date: date
    end_date: date
    sma_months: int
    timing: str
    initial_investment: float
    worker_results: Annotated[List[WorkerResult], operator.add]
    analysis: Optional[str]
    allocations: Optional[Dict[str, int]]

class WorkerState(TypedDict):
    ticker: str
    start_date: date
    end_date: date
    sma_months: int
    timing: str
    initial_investment: float

def supervisor_assign(state: SupervisorState):
    """Assigns the backtest task to a worker for each ticker in the portfolio."""
    return [
        Send("worker_backtest", {
            "ticker": t,
            "start_date": state["start_date"],
            "end_date": state["end_date"],
            "sma_months": state["sma_months"],
            "timing": state["timing"],
            "initial_investment": state["initial_investment"],
        }) for t in state["tickers"]
    ]

def worker_backtest(state: WorkerState):
    """Worker node: Fetches data and runs the simulation for a single ticker."""
    ticker = state["ticker"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    
    extra_lookback = relativedelta(months=24)
    fetch_start_date = start_date - extra_lookback
    
    # Fetch data safely without concurrent yf.download bugs
    stock_data_full = yf.Ticker(ticker).history(start=fetch_start_date, end=end_date)
    irx_data_full = yf.Ticker("^IRX").history(start=fetch_start_date, end=end_date)
    
    # Strip timezone to match original yf.download behavior
    if not stock_data_full.empty:
        stock_data_full.index = stock_data_full.index.tz_localize(None)
    if not irx_data_full.empty:
        irx_data_full.index = irx_data_full.index.tz_localize(None)
    
    if stock_data_full.empty:
        # Return empty result if no data
        return {"worker_results": [{"ticker": ticker, "result": {"error": "No data found"}}]}
        
    result = run_backtest_simulation(
        stock_data_full,
        irx_data_full,
        start_date,
        end_date,
        state["sma_months"],
        state["timing"],
        state["initial_investment"]
    )
    
    return {"worker_results": [{"ticker": ticker, "result": result}]}

def supervisor_analyze(state: SupervisorState):
    """Supervisor node: Collects results and queries Gemini for comparative portfolio analysis."""
    import streamlit as st
    
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account
            # Convert AttrDict to standard dict
            creds_info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(creds_info)
    except Exception:
        pass
        
    if creds:
        vertexai.init(project="sma-agent-v2", location="us-central1", credentials=creds)
    else:
        vertexai.init(project="sma-agent-v2", location="us-central1")
        
    model = GenerativeModel("gemini-2.5-flash")
    
    results_summary = ""
    for wr in state.get("worker_results", []):
        ticker = wr["ticker"]
        res = wr["result"]
        if "error" in res:
            results_summary += f"\\nTicker: {ticker} - ERROR: {res['error']}"
            continue
            
        results_summary += f"\\nTicker: {ticker}\\n"
        results_summary += f"  Final Value: ${res.get('final_value', 0):.2f}\\n"
        results_summary += f"  Total Return: {res.get('total_return', 0) * 100:.2f}%\\n"
        results_summary += f"  CAGR: {res.get('cagr', 0) * 100:.2f}%\\n"
        results_summary += f"  Sharpe Ratio: {res.get('sharpe', 0):.2f}\\n"
        results_summary += f"  Sortino Ratio: {res.get('sortino', 0):.2f}\\n"
        
    prompt = f"""
You are an expert quantitative portfolio manager. I have run systematic backtests on a portfolio of assets.
Here are the backtest results for the same time period and strategy parameters:
{results_summary}

Based on these results, please provide a comprehensive professional analysis.
You MUST format your entire response as a strictly valid JSON object with EXACTLY two keys:
1. "allocations": A dictionary mapping each ticker symbol to its suggested integer allocation percentage (e.g. {{"AAPL": 40, "MSFT": 40, "VTI": 20}}). They must sum to 100.
2. "analysis_text": A string containing your comprehensive professional analysis formatted in markdown. This text MUST include:
    - Which ticker has the best risk-adjusted return and why.
    - The rationale for your chosen portfolio allocation percentages.
    - Brief correlation insights.
    - A plain English executive summary.

Your output must be strictly parseable by Python's json.loads(). Do not include markdown code block backticks around the JSON.
"""
    response = model.generate_content(prompt)
    import json
    import re
    try:
        # Strip markdown backticks if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        allocations = data.get("allocations", {})
        analysis_text = data.get("analysis_text", response.text)
    except Exception as e:
        allocations = {}
        analysis_text = response.text
        
    return {"analysis": analysis_text, "allocations": allocations}

def build_portfolio_graph():
    """Builds and compiles the multi-agent LangGraph StateGraph."""
    workflow = StateGraph(SupervisorState)
    
    workflow.add_node("worker_backtest", worker_backtest)
    workflow.add_node("supervisor_analyze", supervisor_analyze)
    
    # Send from START to dynamically spawn workers
    workflow.add_conditional_edges(START, supervisor_assign, ["worker_backtest"])
    
    # After all workers complete, route to analysis
    workflow.add_edge("worker_backtest", "supervisor_analyze")
    workflow.add_edge("supervisor_analyze", END)
    
    return workflow.compile()
