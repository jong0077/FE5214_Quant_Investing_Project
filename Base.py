
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from RetrieveFactors import *

class BaseStrategy:
    def __init__(self, PriceData, HighData, LowData, VolumeData, IndustryMap, MomentumDaysList=[5, 21, 63], VolDays=21,
                 ModelStartPeriod='2004-01-01', ModelEndPeriod='2020-12-31', cost_fn=None, benchmark=None):
        self.price = PriceData
        self.volume = VolumeData
        self.price_volume = PriceData * VolumeData
        self.high = HighData
        self.low = LowData
        self.IndustryMap = pd.DataFrame([{'Industry': key, 'Symbol': symbol}
                                         for key, symbols in IndustryMap.items() for symbol in symbols])
        self.MomentumDaysList = MomentumDaysList
        self.VolDays = VolDays
        self.ModelStartDate = ModelStartPeriod
        self.ModelEndDate = ModelEndPeriod
        self.cost_fn = cost_fn or (lambda w_prev, w_new: 0.002 * np.sum(np.abs(w_new - w_prev)))
        self.benchmark = np.log(benchmark['Close'] / benchmark['Close'].shift(1)).fillna(0) if benchmark is not None else None

        self.log_returns = np.log(self.price / self.price.shift(1)).fillna(0)
        self.volatility = self.log_returns.rolling(window=self.VolDays).std().clip(lower=0.005).fillna(0.005)
        self.sector_returns = self.IndustryReturns()
        self.volatility_factor = compute_volatility_factors(
            log_returns=self.log_returns, benchmark=self.benchmark, horizons=[5, 21, 63, 126, 252], gamma=0.5, sigma=0.03
        )
        
        self.volume_factor = compute_volume_factors(
            log_returns=self.log_returns, price_volume=self.price_volume, Industry_Map=self.IndustryMap, tau=5
        )

        self.normalized_returns = {
            m: NormalisedSectorNeutralReturns( price=self.price, log_return=self.log_returns, industry_map_df=self.IndustryMap, horizon=m, volatility_days=self.VolDays)
            for m in MomentumDaysList
        }

    def IndustryReturns(self):
        sector_returns = {}
        for industry, group in self.IndustryMap.groupby('Industry'):
            symbols = group['Symbol'].tolist()
            sector_returns[industry] = self.log_returns[symbols].mean(axis=1).fillna(0)
        return sector_returns

    def get_all_factors(self):
        if not hasattr(self, '_cached_tech_factors'):
            self._cached_tech_factors = technical_factors(self.high, self.low, self.price)

        # Existing factors
        factors = {f"v_{m}": self.normalized_returns[m] for m in self.MomentumDaysList}
        factors['vol'] = self.volatility_factor
        factors['volume'] = self.volume_factor

        # Add cached technicals
        factors['tech_BB_Pct'] = self._cached_tech_factors['BB_Pct']
        factors['ATR'] = self._cached_tech_factors['ATR']

        return factors

    
    def multivariate_regression_beta(self, regression_type='linear'):
        start, end = self.ModelStartDate, self.ModelEndDate
        y = self.log_returns[start:end].copy()

        for _, row in self.IndustryMap.iterrows():
            symbol, industry = row['Symbol'], row['Industry']
            y[symbol] = y[symbol] - self.sector_returns[industry].loc[start:end]

        beta_list, pval_list, r2_list, dates_list = [], [], [], []
        dates = y.index

        for t in range(len(dates) - 1):
            date_t = dates[t]
            date_tp1 = dates[t + 1]

            factors = self.get_all_factors()
            X_parts = [df.loc[date_t].rename(name) for name, df in factors.items()]
            X_df = pd.concat(X_parts, axis=1)

            y_vec = y.loc[date_tp1]
            data = pd.concat([X_df, y_vec], axis=1).dropna()
            if data.shape[0] < 10:
                beta_list.append([np.nan] * len(factors))
                pval_list.append([np.nan] * len(factors))
                r2_list.append(np.nan)
                dates_list.append(date_t)
                continue

            X_mat = data.iloc[:, :-1].values
            y_mat = data.iloc[:, -1].values

            if regression_type == 'linear':
                model = LinearRegression().fit(X_mat, y_mat)
                coefs = model.coef_
                pvals = [np.nan] * len(coefs)
                r2 = model.score(X_mat, y_mat)

            elif regression_type == 'lasso':
                model = Lasso(alpha=0.0001, max_iter=10000).fit(X_mat, y_mat)
                coefs = model.coef_
                pvals = [np.nan] * len(coefs)
                r2 = model.score(X_mat, y_mat)

            elif regression_type == 'xgb':
                model = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3).fit(X_mat, y_mat)
                coefs = model.feature_importances_
                pvals = [np.nan] * len(coefs)
                r2 = model.score(X_mat, y_mat)

            else:
                raise ValueError(f"Unknown regression type: {regression_type}")

            beta_list.append(coefs)
            pval_list.append(pvals)
            r2_list.append(r2)
            dates_list.append(date_t)

        cols = [f"beta_{name}" for name in factors.keys()]
        self.beta_df = pd.DataFrame(beta_list, index=dates_list, columns=cols)
        self.pval_history = pd.DataFrame(pval_list, index=dates_list, columns=cols)
        return self.beta_df, pd.Series(r2_list, index=dates_list)

    def generate_signals(self, current_date, beta_prev):
        scores = pd.Series(0.0, index=self.price.columns)
        factors = self.get_all_factors()

        for name, df in factors.items():
            key = f'beta_{name}'
            if key in beta_prev:
                scores += beta_prev[key] * df.loc[current_date]

        return scores  

    def summarize_factor_significance(self):
        if not hasattr(self, 'pval_history') or not hasattr(self, 'beta_df'):
            raise ValueError("Run multivariate_regression_beta() first.")

        mask_significant = self.pval_history < 0.05
        significance_freq = mask_significant.mean()
        avg_beta_when_significant = self.beta_df.where(mask_significant).mean()

        summary = pd.DataFrame({
            'Significance Frequency (p<0.05)': significance_freq,
            'Avg Beta (when p<0.05)': avg_beta_when_significant
        })
        return summary.sort_values(by='Significance Frequency (p<0.05)', ascending=False)

    def run_strategy(self, start_year=2010, end_year=2020, mode='longshort', rebalance_freq='Q'):
        if mode not in ['longshort', 'longonly']:
            raise ValueError("mode must be either 'longshort' or 'longonly'")

        portfolio_returns = pd.Series(dtype=float)
        raw_returns_series = pd.Series(dtype=float) 
        turnover_series = pd.Series(dtype=float)
        previous_weights = pd.Series(dtype=float)

        beta_df, _ = self.multivariate_regression_beta()
        rebal_betas = beta_df.resample(rebalance_freq).mean()
        shifted_betas = rebal_betas.shift(1)  # keep NaNs; handle them in the loop

        all_dates = self.log_returns.index
        target_date = pd.to_datetime(f'{start_year}-01-01')
        backtest_start = all_dates[all_dates < target_date][-1]
        backtest_end = f'{end_year}-12-31'
        actual_returns = self.log_returns.loc[backtest_start:backtest_end]
        grouped = actual_returns.groupby(pd.Grouper(freq=rebalance_freq))

        for period_end, _ in grouped:
            if period_end not in shifted_betas.index:
                continue

            beta_prev = shifted_betas.loc[period_end]
            if beta_prev.isnull().all():
                continue

            period_start = period_end - pd.tseries.frequencies.to_offset(rebalance_freq)
            period_dates = actual_returns.loc[period_start:period_end].index

            current_positions = pd.Series(0.0, index=self.price.columns)

            for current_date in period_dates:
                signals = self.generate_signals(current_date, beta_prev)
                if signals.isnull().all():
                    continue

                ranked = signals.sort_values(ascending=False)
                N = len(ranked)
                n_select = int(np.floor(0.2 * N))
                if n_select < 1:
                    continue

                new_long = ranked.index[:n_select]
                new_short = ranked.index[-n_select:] if mode == 'longshort' else []

                updated_positions = pd.Series(0.0, index=self.price.columns)
                updated_positions[new_long] = 1 / n_select
                if mode == 'longshort':
                    updated_positions[new_short] = -1 / n_select

                try:
                    # Find the next available trading day after current_date
                    next_date = actual_returns.index[actual_returns.index > current_date][0]
                except IndexError:
                    # No next trading day available (e.g., last date in data)
                    continue

                long_returns = actual_returns.loc[next_date, updated_positions > 0]
                short_returns = actual_returns.loc[next_date, updated_positions < 0] if mode == 'longshort' else 0

                raw_return = long_returns.mean() - short_returns.mean() if mode == 'longshort' else long_returns.mean()
                cost = self.cost_fn(previous_weights, updated_positions)

                raw_returns_series.at[next_date] = raw_return
                portfolio_returns.at[next_date] = raw_return - cost
                turnover_series.at[current_date] = np.sum(np.abs(updated_positions - previous_weights))

                previous_weights = updated_positions.copy()
                current_positions = updated_positions.copy()

        self.turnover_series = turnover_series
        return portfolio_returns, raw_returns_series


    def evaluate_performance(self, returns):
        returns = returns.dropna()

        if self.benchmark is None:
            raise ValueError("Benchmark data not provided.")

        aligned = pd.concat([returns, self.benchmark], axis=1).dropna()
        aligned.columns = ['Strategy', 'Benchmark']

        # === Cumulative metrics ===
        strat_cum = np.exp(aligned['Strategy'].cumsum())
        bench_cum = np.exp(aligned['Benchmark'].cumsum())
        total_days = len(aligned)

        # === Annualized return from log returns ===
        strat_return = np.exp(aligned['Strategy'].mean() * 252) - 1
        bench_return = np.exp(aligned['Benchmark'].mean() * 252) - 1

        # === Annualized volatility  ===
        strat_vol = aligned['Strategy'].std() * np.sqrt(252)
        bench_vol = aligned['Benchmark'].std() * np.sqrt(252)

        strat_sharpe = strat_return / strat_vol if strat_vol > 0 else np.nan
        bench_sharpe = bench_return / bench_vol if bench_vol > 0 else np.nan

        strat_dd = ((strat_cum.cummax() - strat_cum) / strat_cum.cummax()).max()
        bench_dd = ((bench_cum.cummax() - bench_cum) / bench_cum.cummax()).max()

        strat_calmar = strat_return / strat_dd if strat_dd > 0 else np.nan
        bench_calmar = bench_return / bench_dd if bench_dd > 0 else np.nan

        avg_turnover = self.turnover_series.mean() if hasattr(self, 'turnover_series') else np.nan

        print("==== Cumulative Performance Comparison ====")
        print(f"{'Metric':<25} {'Strategy':>12} {'Benchmark':>12}")
        print(f"{'Annualized Return':<25} {strat_return:>12.2%} {bench_return:>12.2%}")
        print(f"{'Annualized Volatility':<25} {strat_vol:>12.2%} {bench_vol:>12.2%}")
        print(f"{'Sharpe Ratio':<25} {strat_sharpe:>12.2f} {bench_sharpe:>12.2f}")
        print(f"{'Max Drawdown':<25} {strat_dd:>12.2%} {bench_dd:>12.2%}")
        print(f"{'Calmar Ratio':<25} {strat_calmar:>12.2f} {bench_calmar:>12.2f}")
        print(f"{'Average Daily Turnover':<25} {avg_turnover:>12.2%}")

        # === Year-by-Year Comparison ===
        yearly = aligned.groupby(aligned.index.year)
        year_stats = []

        for year, group in yearly:
            strat_ret = np.exp(group['Strategy'].sum()) - 1
            bench_ret = np.exp(group['Benchmark'].sum()) - 1
            alpha = strat_ret - bench_ret
            year_stats.append([year, strat_ret, bench_ret, alpha])

        year_df = pd.DataFrame(year_stats, columns=['Year', 'Strategy Return', 'Benchmark Return', 'Alpha'])
        year_df.set_index('Year', inplace=True)

        print("\n==== Year-by-Year Returns ====")
        print(year_df.to_string(formatters={
            'Strategy Return': '{:.2%}'.format,
            'Benchmark Return': '{:.2%}'.format,
            'Alpha': '{:.2%}'.format
        }))

        # === Cumulative return plot ===
        plt.figure(figsize=(12, 6))
        plt.plot(strat_cum, label="Strategy", linewidth=2)
        plt.xlim(returns.index.min(), returns.index.max())
        plt.plot(bench_cum, label="Benchmark", linewidth=2)
        plt.title("Cumulative Return Comparison")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # === Difference plot ===
        cum_diff = strat_cum - bench_cum
        plt.figure(figsize=(12, 3))
        plt.plot(cum_diff, label="Strategy - Benchmark", color='purple')
        plt.axhline(0, linestyle='--', color='gray')
        plt.title("Difference in Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Return Difference")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return {
            'strategy': {
                'annual_return': strat_return,
                'volatility': strat_vol,
                'sharpe_ratio': strat_sharpe,
                'max_drawdown': strat_dd,
                'calmar_ratio': strat_calmar,
            },
            'benchmark': {
                'annual_return': bench_return,
                'volatility': bench_vol,
                'sharpe_ratio': bench_sharpe,
                'max_drawdown': bench_dd,
                'calmar_ratio': bench_calmar,
            },
            'avg_turnover': avg_turnover,
            'yearly': year_df
        }

    def beta_stability_summary(self, rebalance_freq='Y', threshold=2.0):
        """
        Compute t-stats of factor betas over time to assess their stability.
        A t-stat ≥ threshold indicates a stable and statistically significant factor.
        """
        if not hasattr(self, 'beta_df'):
            raise ValueError("Run multivariate_regression_beta() first.")

        beta_df = self.beta_df.copy()
        grouped = beta_df.groupby(pd.Grouper(freq=rebalance_freq))

        summary_list = []

        for period_end, group in grouped:
            if group.empty:
                continue

            T = group.notna().sum()
            mean_beta = group.mean()
            std_beta = group.std()
            stderr = std_beta / np.sqrt(T)
            t_stat = mean_beta / stderr

            summary = pd.DataFrame({
                'Period': period_end,
                'Mean Beta': mean_beta,
                'Std Dev': std_beta,
                'N': T,
                'StdErr': stderr,
                'T-stat': t_stat,
                'Stable (|t| >= 2)': (t_stat.abs() >= threshold)
            })

            summary_list.append(summary)

        result = pd.concat(summary_list).reset_index().rename(columns={'index': 'Factor'})
        return result.sort_values(['Period', 'T-stat'], ascending=[True, False])