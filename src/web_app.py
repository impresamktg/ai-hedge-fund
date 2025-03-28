from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
from main import run_hedge_fund
from utils.analysts import ANALYST_ORDER
from llm.models import LLM_ORDER, get_model_info
from backtester import Backtester

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    analysts = [{"display": display, "value": value} for display, value in ANALYST_ORDER]
    models = [{"display": display, "value": value} for display, value, _ in LLM_ORDER]
    return render_template('index.html', analysts=analysts, models=models)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        tickers = [t.strip() for t in data.get('tickers', '').split(',')]
        if not tickers or tickers[0] == '':
            return jsonify({"error": "No tickers provided"}), 400

        selected_analysts = data.get('analysts', [])
        if not selected_analysts:
            return jsonify({"error": "No analysts selected"}), 400

        model_choice = data.get('model')
        if not model_choice:
            return jsonify({"error": "No model selected"}), 400

        # Initialize portfolio with user-specified investment amount
        initial_capital = float(data.get('initial_capital', 100000))
        portfolio = {
            "cash": initial_capital,
            "margin_requirement": float(data.get('margin_requirement', 0.0)),
            "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for ticker in tickers},
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
        }

        result = run_hedge_fund(
            tickers=tickers,
            start_date=data['start_date'],
            end_date=data['end_date'],
            portfolio=portfolio,
            show_reasoning=True,
            selected_analysts=selected_analysts,
            model_name=model_choice,
            model_provider=get_model_info(model_choice).provider.value if get_model_info(model_choice) else "OpenAI"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        tickers = [t.strip() for t in data['tickers'].split(',')]

        backtester = Backtester(
            agent=run_hedge_fund,
            tickers=tickers,
            start_date=data['start_date'],
            end_date=data['end_date'],
            initial_capital=float(data.get('initial_capital', 100000)),
            model_name=model_choice,
            model_provider=get_model_info(model_choice).provider.value if get_model_info(model_choice) else "OpenAI",
            selected_analysts=data['analysts'],
            initial_margin_requirement=float(data.get('margin_requirement', 0.0))
        )

        metrics = backtester.run_backtest()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)