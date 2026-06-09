import yfinance as yf
import pandas as pd
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import vertexai
from vertexai.generative_models import GenerativeModel

# Import the existing simulation logic
from backtest import run_backtest_simulation

class AgentState(TypedDict):
    ticker: str
    start_date: date
    end_date: date
    sma_months: int
    timing: str
    initial_investment: float
    stock_data_full: Optional[pd.DataFrame]
    irx_data_full: Optional[pd.DataFrame]
    backtest_result: Optional[Dict[str, Any]]
    recommendation: Optional[str]

def fetch_data(state: AgentState):
    """Fetches historical stock and ^IRX data from yfinance."""
    ticker = state["ticker"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    
    # Provide enough lookback buffer for SMA calculations up to 12 months,
    # adding an extra year for safe window boundaries as in ui.py.
    extra_lookback = relativedelta(months=24)
    fetch_start_date = start_date - extra_lookback
    
    stock_data_full = yf.download(ticker, start=fetch_start_date, end=end_date)
    irx_data_full = yf.download("^IRX", start=fetch_start_date, end=end_date)
    
    return {"stock_data_full": stock_data_full, "irx_data_full": irx_data_full}

def run_backtest(state: AgentState):
    """Runs the initial backtest using the provided parameters."""
    result = run_backtest_simulation(
        state["stock_data_full"],
        state["irx_data_full"],
        state["start_date"],
        state["end_date"],
        state["sma_months"],
        state["timing"],
        state["initial_investment"]
    )
    return {"backtest_result": result}

def check_signal(state: AgentState):
    """Conditional edge: Route based on Sharpe ratio."""
    sharpe = state.get("backtest_result", {}).get("sharpe", 0)
    if sharpe < 1:
        return "try_alternatives"
    return "generate_recommendation"

def try_alternatives(state: AgentState):
    """Tests alternative SMA parameters and keeps the best one."""
    alternatives = [7, 8, 9, 10, 11, 12]
    best_result = state["backtest_result"]
    best_sharpe = best_result.get("sharpe", -float('inf'))
    
    for test_sma in alternatives:
        if test_sma == state["sma_months"]:
            continue
            
        test_result = run_backtest_simulation(
            state["stock_data_full"],
            state["irx_data_full"],
            state["start_date"],
            state["end_date"],
            test_sma,
            state["timing"],
            state["initial_investment"]
        )
        
        test_sharpe = test_result.get("sharpe", 0)
        if test_sharpe > best_sharpe:
            best_sharpe = test_sharpe
            best_result = test_result
            
    return {"backtest_result": best_result}

def generate_recommendation(state: AgentState):
    """Calls Gemini to interpret backtest results."""
    vertexai.init(project="sma-agent-v2", location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")
    
    ticker = state["ticker"]
    result = state["backtest_result"]
    
    prompt = f"""
You are an expert financial analyst. I have run a stock backtest simulation on {ticker}.
Here are the final backtest results:
- SMA Months Used: {result.get('sma_months')}
- Trading Frequency: {result.get('timing')}
- Final Portfolio Value: ${result.get('final_value', 0):.2f}
- Total Return: {result.get('total_return', 0) * 100:.2f}%
- CAGR: {result.get('cagr', 0) * 100:.2f}%
- Sharpe Ratio: {result.get('sharpe', 0):.2f}
- Sortino Ratio: {result.get('sortino', 0):.2f}

Please interpret these backtest results in plain English and provide a clear, actionable recommendation.
"""
    response = model.generate_content(prompt)
    return {"recommendation": response.text}

def build_agent_graph():
    """Builds and compiles the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("run_backtest", run_backtest)
    workflow.add_node("try_alternatives", try_alternatives)
    workflow.add_node("generate_recommendation", generate_recommendation)
    
    # Define edges
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "run_backtest")
    
    # Conditional edge after backtest
    workflow.add_conditional_edges(
        "run_backtest",
        check_signal,
        {
            "try_alternatives": "try_alternatives",
            "generate_recommendation": "generate_recommendation"
        }
    )
    
    workflow.add_edge("try_alternatives", "generate_recommendation")
    workflow.add_edge("generate_recommendation", END)
    
    return workflow.compile()

def run_agent(ticker: str, start_date: date, end_date: date, sma_months: int, timing: str, initial_investment: float) -> str:
    """Executes the complete agent workflow and returns the natural language recommendation."""
    graph = build_agent_graph()
    
    initial_state = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "sma_months": sma_months,
        "timing": timing,
        "initial_investment": initial_investment,
        "stock_data_full": None,
        "irx_data_full": None,
        "backtest_result": None,
        "recommendation": None
    }
    
    final_state = graph.invoke(initial_state)
    return final_state["recommendation"]

if __name__ == "__main__":
    # Example usage:
    # res = run_agent("AAPL", date(2020, 1, 1), date.today(), 10, "Monthly", 10000.0)
    # print(res)
    pass
