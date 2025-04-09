from Base import BaseStrategy
import numpy as np
import pandas as pd
import statsmodels.api as sm

class MonthlyRebalanceHoldWithMonthlyBetas(BaseStrategy):
    def multivariate_regression_beta(self):
        start, end = self.ModelStartDate, self.ModelEndDate
        y = self.log_returns[start:end].copy()

        # Sector-neutralize returns
        for _, row in self.IndustryMap.iterrows():
            symbol, industry = row['Symbol'], row['Industry']
            y[symbol] = y[symbol] - self.sector_returns[industry].loc[start:end]

        # Use 21-day forward returns for monthly horizon
        y_forward = y.rolling(window=21).sum().shift(-21)
        beta_list, pval_list, r2_list, dates_list = [], [], [], []
        dates = y.index[:-21]  # forward shift truncates last 21 days

        for date_t in dates:
            factors = self.get_all_factors()
            X_parts = [df.loc[date_t].rename(name) for name, df in factors.items()]
            X_df = pd.concat(X_parts, axis=1)

            y_vec = y_forward.loc[date_t]
            data = pd.concat([X_df, y_vec], axis=1).dropna()

            if data.shape[0] < 10:
                beta_list.append([np.nan] * len(factors))
                pval_list.append([np.nan] * len(factors))
                r2_list.append(np.nan)
                dates_list.append(date_t)
                continue

            X_mat = sm.add_constant(data.iloc[:, :-1].values)
            y_mat = data.iloc[:, -1].values
            model = sm.OLS(y_mat, X_mat).fit()

            beta_list.append(model.params[1:])
            pval_list.append(model.pvalues[1:])
            r2_list.append(model.rsquared)
            dates_list.append(date_t)

        cols = [f"beta_{name}" for name in factors.keys()]
        self.beta_df = pd.DataFrame(beta_list, index=dates_list, columns=cols)
        self.pval_history = pd.DataFrame(pval_list, index=dates_list, columns=cols)
        return self.beta_df, pd.Series(r2_list, index=dates_list)

    def run_strategy(self, start_year=2010, end_year=2020, mode='longshort'):
        if mode not in ['longshort', 'longonly']:
            raise ValueError("mode must be either 'longshort' or 'longonly'")

        portfolio_returns = pd.Series(dtype=float)
        raw_returns_series = pd.Series(dtype=float)
        turnover_series = pd.Series(dtype=float)
        previous_weights = pd.Series(dtype=float)

        beta_df, _ = self.multivariate_regression_beta()
        actual_returns = self.log_returns.loc[f'{start_year}-01-01':f'{end_year}-12-31']
        monthly_rebalance_dates = self.price.index.to_series().asfreq('MS').dropna().index

        for rebal_date in monthly_rebalance_dates:
            # Rolling quarter: 3 months before the rebalance date
            quarter_start = rebal_date - pd.DateOffset(months=3)
            quarter_end = rebal_date - pd.Timedelta(days=1)
            quarter_betas = beta_df.loc[quarter_start:quarter_end].dropna()

            if quarter_betas.empty:
                continue

            avg_beta = quarter_betas.mean()

            # Signal generation at rebalance date
            signals = self.generate_signals(rebal_date, avg_beta)
            if signals.isnull().all():
                print(f"⚠️ No valid signals at rebalance on {rebal_date.date()} — skipping.")
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

            # Hold for the full calendar month
            month_end = rebal_date + pd.offsets.MonthEnd(0)
            holding_dates = actual_returns.loc[rebal_date:month_end].index

            for date in holding_dates:
                # Use daily returns and keep weights fixed
                if date not in actual_returns.index:
                    continue

                daily_ret = actual_returns.loc[date]
                long_ret = daily_ret[updated_positions > 0].mean()
                short_ret = daily_ret[updated_positions < 0].mean() if mode == 'longshort' else 0.0
                raw_return = long_ret - short_ret if mode == 'longshort' else long_ret

                cost = self.cost_fn(previous_weights, updated_positions) if date == rebal_date else 0
                raw_returns_series.at[date] = raw_return
                portfolio_returns.at[date] = raw_return - cost
                turnover_series.at[date] = np.sum(np.abs(updated_positions - previous_weights)) if date == rebal_date else 0

            previous_weights = updated_positions.copy()

        self.turnover_series = turnover_series
        return portfolio_returns, raw_returns_series

    def evaluate_performance(self, returns):
        returns = returns.dropna()

        # Align for current evaluation only (do NOT modify self.benchmark)
        if self.benchmark is not None:
            benchmark_aligned = self.benchmark.loc[self.benchmark.index.intersection(returns.index)]
        else:
            raise ValueError("Benchmark data not provided.")

        aligned = pd.concat([returns, benchmark_aligned], axis=1).dropna()
        aligned.columns = ['Strategy', 'Benchmark']

        # Create a local benchmark copy aligned to strategy returns
        benchmark_aligned = self.benchmark.loc[self.benchmark.index.intersection(returns.index)]
        aligned = pd.concat([returns, benchmark_aligned], axis=1).dropna()
        aligned.columns = ['Strategy', 'Benchmark']

        print("Benchmark head:\n", self.benchmark.head())
        print("Benchmark used for evaluation:\n", aligned['Benchmark'].head())
        print("Dates:", aligned.index[:5])

        return super().evaluate_performance(aligned['Strategy'])
