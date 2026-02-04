"""
ECM-IV Estimation Module

Error Correction Model with Instrumental Variables for price pass-through analysis.
Supports asymmetric effects, HAC standard errors, and flexible controls.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.iv import IV2SLS
from typing import Optional, List, Dict, Tuple, Any


def estimate_ecm_iv(
    df: pd.DataFrame,
    price: str = 'ln_rt',
    cost: str = 'ln_wh',
    K_cost: int = 8,
    K_price: int = 2,
    L: int = 0,
    instruments_name: Optional[List[str]] = None,
    instruments_lags: Optional[List[int]] = None,
    instruments_lags_pos: Optional[List[int]] = None,
    instruments_lags_neg: Optional[List[int]] = None,
    asym: bool = False,
    hac_lags: int = 4,
    crop: bool = False,
    ec_controls: Optional[Dict[str, str]] = None,
    controls: Optional[Dict[str, str]] = None,
    date_col: str = 'date',
    entity_col: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Any, Any, Dict[str, float], pd.DataFrame]:
    """
    Estimate ECM pass-through model using 2SLS.
    
    Model specification:
        Δln(P_t^price) = α + Σ_{k=0}^{K_cost} β_k Δln(P_{t-k}^cost) 
                           + Σ_{j=1}^{K_price} γ_j Δln(P_{t-j}^price)
                           + θ · EC_{t-1} + controls + ε_t
    
    where EC_{t-1} = ln(P_{t-1}^price) - φ_0 - φ_1 ln(P_{t-1}^cost) - ec_controls
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data with date, entity, price, cost, and instrument columns.
        Price and cost columns should already be in logs.
    price : str
        Column name for log downstream price (default: 'ln_rt')
    cost : str
        Column name for log upstream cost (default: 'ln_wh')
    K_cost : int
        Number of cost lags including contemporaneous (default: 8)
    K_price : int
        Number of price lags for AR dynamics (default: 2)
    L : int
        Number of cost leads (default: 0)
    instruments_name : list of str, optional
        Column names to use as instruments (e.g., ['flu'])
    instruments_lags : list of int, optional
        Lags to create for differenced instruments in symmetric case (default: [2, 20])
    instruments_lags_pos : list of int, optional
        Lags for positive instrument shocks in asymmetric case.
        Use shorter lags for fast-transmitting shocks (e.g., [2, 4, 6]).
        If None and asym=True, falls back to instruments_lags.
    instruments_lags_neg : list of int, optional
        Lags for negative instrument shocks in asymmetric case.
        Use longer lags for slow-transmitting shocks (e.g., [15, 18, 20]).
        If None and asym=True, falls back to instruments_lags.
    asym : bool
        If True, split cost/price diffs into positive and negative components
    hac_lags : int
        Number of lags for HAC standard errors (default: 4)
    crop : bool
        If True, winsorize top and bottom 1% of observations
    ec_controls : dict, optional
        Controls for EC term regression. Keys are column names, values are:
        - 'fe': include as fixed effects
        - 'reg': include as regression control
    controls : dict, optional
        Controls for main ECM regression. Keys are column names, values are:
        - 'fe': include as fixed effects
        - 'reg': include as regression control
    date_col : str
        Name of date column (default: 'date')
    entity_col : str, optional
        Name of entity/panel column (default: None for pure time series)
    verbose : bool
        If True, print estimation results (default: True)
    
    Returns
    -------
    model : IV2SLS result
        Fitted 2SLS model
    fs_model : OLS result or dict
        First-stage regression(s). Dict if asym=True with multiple endogenous vars.
    coint_params : dict
        Cointegrating relationship parameters (phi_0, phi_1, and any ec_controls)
    df_est : pd.DataFrame
        Estimation sample with all constructed variables
    
    Examples
    --------
    Symmetric estimation:
    >>> model, fs, coint, df_est = estimate_ecm_iv(
    ...     df=df_weekly,
    ...     price='ln_rt',
    ...     cost='ln_wh',
    ...     instruments_name=['flu'],
    ...     instruments_lags=[2, 10, 20],
    ...     asym=False
    ... )
    
    Asymmetric estimation with different lag structures:
    >>> model, fs, coint, df_est = estimate_ecm_iv(
    ...     df=df_weekly,
    ...     price='ln_rt',
    ...     cost='ln_wh',
    ...     instruments_name=['flu'],
    ...     instruments_lags_pos=[2, 4, 6],     # Fast: outbreak → price spike
    ...     instruments_lags_neg=[15, 18, 20],  # Slow: recovery → flock rebuild
    ...     asym=True
    ... )
    """
    
    # Set defaults
    if instruments_name is None:
        instruments_name = []
    if instruments_lags is None:
        instruments_lags = []
    if ec_controls is None:
        ec_controls = {}
    if controls is None:
        controls = {}
    
    # Copy and prepare data
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if entity_col is not None:
        df = df.sort_values([entity_col, date_col]).reset_index(drop=True)
    else:
        df = df.sort_values(date_col).reset_index(drop=True)
    
    # Variables are assumed to be already in logs
    ln_price = price
    ln_cost = cost
    
    
    # This only affects the ECM estimation, not the long-run relationship
    if crop:
        df = _winsorize_panel(df, [price, cost], limits=(0.01, 0.01))
        if verbose:
            print(f"Winsorized {price} and {cost} at 1%/99% for ECM estimation")
    
    # Build cointegrating regression on FULL sample (levels, before creating lags)
    coint_params = _estimate_cointegrating_relationship(
        df, ln_price, ln_cost, ec_controls, verbose
    )
    
    # Compute error correction term
    df["ec"] = df[ln_price] - coint_params["phi_0"] - coint_params["phi_1"] * df[ln_cost]
    for col, spec in ec_controls.items():
        if spec == 'reg':
            df["ec"] -= coint_params[f"ec_{col}"] * df[col]
        elif spec == 'fe':
            # Subtract group means (absorbed by FE)
            for val in df[col].unique():
                if f"ec_{col}_{val}" in coint_params:
                    mask = df[col] == val
                    df.loc[mask, "ec"] -= coint_params[f"ec_{col}_{val}"]
    
    # First differences
    if entity_col is not None:
        df["d_price"] = df.groupby(entity_col)[ln_price].diff()
        df["d_cost"] = df.groupby(entity_col)[ln_cost].diff()
        df["ec_lag1"] = df.groupby(entity_col)["ec"].shift(1)
    else:
        df["d_price"] = df[ln_price].diff()
        df["d_cost"] = df[ln_cost].diff()
        df["ec_lag1"] = df["ec"].shift(1)
    
    # Create cost lags and leads
    cost_vars = _create_lag_lead_vars(
        df, "d_cost", entity_col, K_cost, L, asym, "cost"
    )
    
    # Create price lags
    price_vars = _create_price_lags(
        df, "d_price", entity_col, K_price, asym, "price"
    )
    
    # Create instruments (now uses differenced instruments)
    instrument_vars = _create_instruments(
        df, instruments_name, instruments_lags, entity_col, asym,
        instruments_lags_pos, instruments_lags_neg
    )
    
    # Create control variables (FE dummies and regression controls)
    control_vars, fe_info = _create_controls(df, controls, date_col)
    
    # Drop missing values
    all_vars = (
        ["d_price", "ec_lag1"] + 
        cost_vars + price_vars + instrument_vars + control_vars
    )
    df_est = df.dropna(subset=[v for v in all_vars if v in df.columns]).copy()
    
    if verbose:
        print(f"\nEstimation sample: {len(df_est)} observations")
        if asym:
            print(f"Instruments (positive shocks): {[v for v in instrument_vars if '_pos_' in v]}")
            print(f"Instruments (negative shocks): {[v for v in instrument_vars if '_neg_' in v]}")
        else:
            print(f"Instruments: {instrument_vars}")
    
    # Identify endogenous variable (contemporaneous cost change)
    endog_var = "d_cost_L0" if not asym else ["d_cost_pos_L0", "d_cost_neg_L0"]
    
    # Build variable lists for estimation
    exog_cost_vars = [v for v in cost_vars if v not in (
        [endog_var] if isinstance(endog_var, str) else endog_var
    )]
    
    # 2SLS Estimation
    y = df_est["d_price"]
    
    if isinstance(endog_var, str):
        endog = df_est[[endog_var]]
    else:
        endog = df_est[endog_var]
    
    exog_vars = exog_cost_vars + price_vars + ["ec_lag1"] + control_vars
    exog = sm.add_constant(df_est[exog_vars])
    
    instruments = df_est[instrument_vars] if instrument_vars else None
    
    if instruments is not None and len(instrument_vars) > 0:
        model = IV2SLS(
            dependent=y,
            exog=exog,
            endog=endog,
            instruments=instruments,
        ).fit(cov_type="kernel", kernel="bartlett", bandwidth=hac_lags) #just means Newey–West HAC for IV 
    else:
        # No instruments - run OLS
        X_ols = pd.concat([exog, endog], axis=1)
        model = sm.OLS(y, X_ols).fit(
            cov_type='HAC', 
            cov_kwds={'maxlags': hac_lags}
        )
    
    # First-stage diagnostics
    fs_model = None
    if instruments is not None and len(instrument_vars) > 0:
        fs_model = _first_stage_diagnostics(
            df_est, endog_var, exog_vars, instrument_vars, verbose
        )
    
    # Print results
    if verbose:
        _print_results(
            model, cost_vars, price_vars, control_vars, 
            fe_info, coint_params, asym
        )
    
    return model, fs_model, coint_params, df_est


def _winsorize_panel(
    df: pd.DataFrame, 
    cols: List[str], 
    limits: Tuple[float, float] = (0.01, 0.01)
) -> pd.DataFrame:
    """Winsorize specified columns at given percentile limits."""
    from scipy.stats import mstats
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col], limits=limits)
    return df


def _estimate_cointegrating_relationship(
    df: pd.DataFrame,
    ln_price: str,
    ln_cost: str,
    ec_controls: Dict[str, str],
    verbose: bool
) -> Dict[str, float]:
    """
    Estimate cointegrating relationship via OLS.
    
    ln(P^price) = φ_0 + φ_1 ln(P^cost) + ec_controls + u
    
    Uses all available observations (drops NAs only for variables in this regression).
    """
    params = {}
    
    # Build cointegrating regression
    X_vars = [ln_cost]
    
    # Identify FE columns
    fe_cols = []
    for col, spec in ec_controls.items():
        if spec == 'reg':
            X_vars.append(col)
        elif spec == 'fe':
            fe_cols.append(col)
    
    # Determine columns needed for cointegrating regression
    coint_needed_cols = [ln_price, ln_cost] + [c for c in X_vars if c != ln_cost] + fe_cols
    
    # Drop NAs only for columns used in cointegrating regression
    df_coint = df[coint_needed_cols].dropna().copy()
    
    # Reset index for clean alignment
    df_coint = df_coint.reset_index(drop=True)
    
    # Build X matrix
    X = sm.add_constant(df_coint[X_vars])
    
    # Add fixed effects dummies
    if fe_cols:
        for col in fe_cols:
            dummies = pd.get_dummies(df_coint[col], prefix=f"ec_{col}", drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    
    y = df_coint[ln_price]
    
    coint_model = sm.OLS(y, X).fit()
    
    params["phi_0"] = coint_model.params["const"]
    params["phi_1"] = coint_model.params[ln_cost]
    params["_n_obs_coint"] = int(coint_model.nobs)  # Store for diagnostics
    
    # Store control coefficients
    for col, spec in ec_controls.items():
        if spec == 'reg':
            params[f"ec_{col}"] = coint_model.params[col]
        elif spec == 'fe':
            for dummy_col in coint_model.params.index:
                if dummy_col.startswith(f"ec_{col}_"):
                    params[dummy_col] = coint_model.params[dummy_col]
    
    if verbose:
        print("=" * 60)
        print("COINTEGRATING RELATIONSHIP")
        print("=" * 60)
        print(f"Observations: {params['_n_obs_coint']}")
        print(f"{ln_price} = {params['phi_0']:.4f} + {params['phi_1']:.4f} × {ln_cost}")
        print(f"Long-run elasticity (φ₁): {params['phi_1']:.4f}")
        if ec_controls:
            print(f"EC controls: {list(ec_controls.keys())}")
    
    return params


def _create_lag_lead_vars(
    df: pd.DataFrame,
    var: str,
    entity_col: Optional[str],
    K: int,
    L: int,
    asym: bool,
    prefix: str
) -> List[str]:
    """Create lagged and lead variables, optionally split by sign."""
    created_vars = []
    
    # Leads (negative lags)
    for lead in range(1, L + 1):
        col_name = f"d_{prefix}_F{lead}"
        if entity_col is not None:
            df[col_name] = df.groupby(entity_col)[var].shift(-lead)
        else:
            df[col_name] = df[var].shift(-lead)
        if asym:
            df[f"d_{prefix}_pos_F{lead}"] = df[col_name].clip(lower=0)
            df[f"d_{prefix}_neg_F{lead}"] = df[col_name].clip(upper=0).abs()
            created_vars.extend([f"d_{prefix}_pos_F{lead}", f"d_{prefix}_neg_F{lead}"])
        else:
            created_vars.append(col_name)
    
    # Contemporaneous and lags
    for lag in range(K + 1):
        col_name = f"d_{prefix}_L{lag}"
        if entity_col is not None:
            df[col_name] = df.groupby(entity_col)[var].shift(lag)
        else:
            df[col_name] = df[var].shift(lag)
        if asym:
            df[f"d_{prefix}_pos_L{lag}"] = df[col_name].clip(lower=0)
            df[f"d_{prefix}_neg_L{lag}"] = df[col_name].clip(upper=0).abs()
            created_vars.extend([f"d_{prefix}_pos_L{lag}", f"d_{prefix}_neg_L{lag}"])
        else:
            created_vars.append(col_name)
    
    return created_vars


def _create_price_lags(
    df: pd.DataFrame,
    var: str,
    entity_col: Optional[str],
    K: int,
    asym: bool,
    prefix: str
) -> List[str]:
    """Create lagged price variables for AR dynamics."""
    created_vars = []
    
    for lag in range(1, K + 1):
        col_name = f"d_{prefix}_L{lag}"
        if entity_col is not None:
            df[col_name] = df.groupby(entity_col)[var].shift(lag)
        else:
            df[col_name] = df[var].shift(lag)
        if asym:
            df[f"d_{prefix}_pos_L{lag}"] = df[col_name].clip(lower=0)
            df[f"d_{prefix}_neg_L{lag}"] = df[col_name].clip(upper=0).abs()
            created_vars.extend([f"d_{prefix}_pos_L{lag}", f"d_{prefix}_neg_L{lag}"])
        else:
            created_vars.append(col_name)
    
    return created_vars


def _create_instruments(
    df: pd.DataFrame,
    instruments_name: List[str],
    instruments_lags: List[int],
    entity_col: Optional[str],
    asym: bool = False,
    instruments_lags_pos: Optional[List[int]] = None,
    instruments_lags_neg: Optional[List[int]] = None,
) -> List[str]:
    """
    Create lagged instruments in differences.
    
    For asymmetric estimation, allows different lag structures for
    positive vs negative instrument shocks to reflect different
    transmission mechanisms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame to add instrument columns to
    instruments_name : list of str
        Base instrument column names (in levels)
    instruments_lags : list of int
        Lags for symmetric case (ignored if asym=True and pos/neg lags provided)
    entity_col : str, optional
        Panel entity column for grouped operations
    asym : bool
        If True, split differenced instruments into pos/neg components
    instruments_lags_pos : list of int, optional
        Lags for positive instrument shocks (e.g., [2, 4, 6] for fast transmission)
        If None and asym=True, falls back to instruments_lags.
    instruments_lags_neg : list of int, optional
        Lags for negative instrument shocks (e.g., [15, 18, 20] for slow recovery)
        If None and asym=True, falls back to instruments_lags.
    
    Returns
    -------
    list of str
        Names of created instrument columns
    """
    instrument_vars = []
    
    # Set defaults for asymmetric lags
    if asym:
        if instruments_lags_pos is None:
            instruments_lags_pos = instruments_lags
        if instruments_lags_neg is None:
            instruments_lags_neg = instruments_lags
    
    for inst in instruments_name:
        # First, create the differenced instrument
        d_inst = f"d_{inst}"
        if entity_col is not None:
            df[d_inst] = df.groupby(entity_col)[inst].diff()
        else:
            df[d_inst] = df[inst].diff()
        
        if asym:
            # Positive shocks: use instruments_lags_pos
            for lag in instruments_lags_pos:
                col_name = f"d_{inst}_L{lag}"
                if col_name not in df.columns:
                    if entity_col is not None:
                        df[col_name] = df.groupby(entity_col)[d_inst].shift(lag)
                    else:
                        df[col_name] = df[d_inst].shift(lag)
                
                pos_col = f"d_{inst}_pos_L{lag}"
                df[pos_col] = df[col_name].clip(lower=0)
                instrument_vars.append(pos_col)
            
            # Negative shocks: use instruments_lags_neg
            for lag in instruments_lags_neg:
                col_name = f"d_{inst}_L{lag}"
                if col_name not in df.columns:
                    if entity_col is not None:
                        df[col_name] = df.groupby(entity_col)[d_inst].shift(lag)
                    else:
                        df[col_name] = df[d_inst].shift(lag)
                
                neg_col = f"d_{inst}_neg_L{lag}"
                df[neg_col] = df[col_name].clip(upper=0).abs()
                instrument_vars.append(neg_col)
        else:
            # Symmetric case: same lags for all
            for lag in instruments_lags:
                col_name = f"d_{inst}_L{lag}"
                if entity_col is not None:
                    df[col_name] = df.groupby(entity_col)[d_inst].shift(lag)
                else:
                    df[col_name] = df[d_inst].shift(lag)
                instrument_vars.append(col_name)
    
    return instrument_vars


def _create_controls(
    df: pd.DataFrame,
    controls: Dict[str, str],
    date_col: str
) -> Tuple[List[str], Dict[str, int]]:
    """Create control variables (FE dummies and regression controls)."""
    control_vars = []
    fe_info = {}
    
    for col, spec in controls.items():
        if spec == 'fe':
            # Handle special cases
            if col == 'month':
                df['_month'] = df[date_col].dt.month
                source_col = '_month'
            elif col == 'week':
                df['_week'] = df[date_col].dt.isocalendar().week
                source_col = '_week'
            elif col == 'year':
                df['_year'] = df[date_col].dt.year
                source_col = '_year'
            else:
                source_col = col
            
            dummies = pd.get_dummies(
                df[source_col], prefix=col, drop_first=True, dtype=float
            )
            dummy_cols = list(dummies.columns)
            df[dummy_cols] = dummies
            control_vars.extend(dummy_cols)
            fe_info[col] = len(dummy_cols)
            
        elif spec == 'reg':
            control_vars.append(col)
    
    return control_vars, fe_info


def _first_stage_diagnostics(
    df_est: pd.DataFrame,
    endog_var,  # Can be str or List[str]
    exog_vars: List[str],
    instrument_vars: List[str],
    verbose: bool
) -> Any:
    """Run first-stage regression and compute diagnostics."""
    
    endog_list = [endog_var] if isinstance(endog_var, str) else endog_var
    fs_models = {}
    
    for ev in endog_list:
        X_fs = sm.add_constant(df_est[exog_vars + instrument_vars])
        y_fs = df_est[ev]
        fs_model = sm.OLS(y_fs, X_fs).fit()
        
        # Compute partial F-statistic
        X_restricted = sm.add_constant(df_est[exog_vars])
        fs_restricted = sm.OLS(y_fs, X_restricted).fit()
        
        n = len(y_fs)
        k_full = X_fs.shape[1]
        k_iv = len(instrument_vars)
        f_stat = ((fs_restricted.ssr - fs_model.ssr) / k_iv) / (fs_model.ssr / (n - k_full))
        f_pval = 1 - stats.f.cdf(f_stat, k_iv, n - k_full)
        
        fs_models[ev] = {
            'model': fs_model,
            'f_stat': f_stat,
            'f_pval': f_pval
        }
        
        if verbose:
            print(f"\n--- First Stage for {ev} ---")
            print(f"First stage R²: {fs_model.rsquared:.4f}")
            print(f"Partial F-stat: {f_stat:.2f} (p={f_pval:.4f})")
            if f_stat < 10:
                print("  ⚠️  WARNING: F < 10 suggests weak instruments")
            for v in instrument_vars:
                sig = "***" if fs_model.pvalues[v] < 0.01 else (
                    "**" if fs_model.pvalues[v] < 0.05 else (
                    "*" if fs_model.pvalues[v] < 0.1 else ""
                ))
                print(f"  {v}: {fs_model.params[v]:.6f} (p={fs_model.pvalues[v]:.4f}) {sig}")
    
    return fs_models if len(fs_models) > 1 else list(fs_models.values())[0]


def _print_results(
    model: Any,
    cost_vars: List[str],
    price_vars: List[str],
    control_vars: List[str],
    fe_info: Dict[str, int],
    coint_params: Dict[str, float],
    asym: bool
) -> None:
    """Print formatted estimation results."""
    
    print("\n" + "=" * 60)
    print("2SLS RESULTS")
    print("=" * 60)
    
    # Key variables to display
    key_vars = ["const"] + cost_vars + price_vars + ["ec_lag1"]
    
    print(f"{'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'P-value':>10}")
    print("-" * 55)
    
    for var in key_vars:
        if var in model.params.index:
            coef = model.params[var]
            se = model.std_errors[var] if hasattr(model, 'std_errors') else model.bse[var]
            pval = model.pvalues[var]
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            print(f"{var:<25} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")
    
    # Fixed effects summary
    if fe_info:
        print("\nFixed Effects:")
        for fe_name, n_dummies in fe_info.items():
            print(f"  {fe_name}: {n_dummies} dummies")
    
    # Model fit
    r2 = model.rsquared if hasattr(model, 'rsquared') else model.rsquared_adj
    print(f"\nR-squared: {r2:.4f}")
    
    # Key parameters interpretation
    print("\n" + "=" * 60)
    print("KEY PARAMETERS")
    print("=" * 60)
    
    theta = model.params["ec_lag1"]
    theta_pval = model.pvalues["ec_lag1"]
    half_life = np.log(0.5) / np.log(1 + theta) if theta < 0 else np.nan
    
    print(f"EC coefficient (θ): {theta:.4f} (p={theta_pval:.4f})")
    if not np.isnan(half_life):
        print(f"Half-life of disequilibrium: {half_life:.1f} periods")
    else:
        print("Half-life: undefined (θ ≥ 0, no error correction)")
    
    # Contemporaneous pass-through
    if asym:
        if "d_cost_pos_L0" in model.params.index:
            print(f"Contemporaneous pass-through (positive): {model.params['d_cost_pos_L0']:.4f}")
        if "d_cost_neg_L0" in model.params.index:
            print(f"Contemporaneous pass-through (negative): {model.params['d_cost_neg_L0']:.4f}")
    else:
        if "d_cost_L0" in model.params.index:
            print(f"Contemporaneous pass-through (β₀): {model.params['d_cost_L0']:.4f}")
    
    print(f"Long-run elasticity (φ₁): {coint_params['phi_1']:.4f}")




def compute_cumulative_passthrough(
    model: Any,
    phi_1: float,
    K_cost: int = 8,
    K_price: int = 2,
    periods: int = 20,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    asym: bool = False
) -> pd.DataFrame:
    """
    Compute cumulative pass-through with bootstrapped confidence intervals.
    
    Implements Appendix methodology (Equations A1 & A2) from Borenstein, Cameron, Gilbert (1997).
    
    Parameters
    ----------
    model : Result object
        Fitted ECM model (statsmodels result)
    phi_1 : float
        Long-run elasticity from cointegrating relationship
    K_cost : int
        Number of cost lags used in estimation
    K_price : int
        Number of price lags used in estimation
    periods : int
        Horizon for impulse response
    n_boot : int
        Number of bootstrap draws
    ci_level : float
        Confidence level (e.g., 0.95)
    seed : int
        Random seed
    asym : bool
        Whether asymmetric specification was used
        
    Returns
    -------
    pd.DataFrame
        Cumulative pass-through paths and confidence intervals.
    """
    params = model.params
    cov = model.cov if hasattr(model, 'cov') else model.cov_params()
    
    def _compute_passthrough_path(
        params_dict: Dict[str, float], 
        phi_1_val: float,
        shock_type: str = 'symmetric'
    ) -> np.ndarray:
        """
        Compute pass-through path for given parameters.
        
        Logic:
        - Beta (Cost): Fixed based on shock_type (Exogenous input)
        - Gamma (Price AR): Dynamic based on direction of price path (Endogenous output)
        - Theta (EC): Standard error correction
        """
        
        # 1. Load Coefficients
        theta = params_dict.get("ec_lag1", 0.0)
        
        # Load BOTH sets of gamma coefficients for dynamic switching
        # (Used in Eq A2: gamma+ * MAX(dB,0) + gamma- * MIN(dB,0))
        if asym:
            gamma_pos = {j: params_dict.get(f"d_price_pos_L{j}", 0.0) for j in range(1, K_price + 1)}
            gamma_neg = {j: params_dict.get(f"d_price_neg_L{j}", 0.0) for j in range(1, K_price + 1)}
        else:
            # For symmetric models, pos and neg coefficients are identical
            gamma_common = {j: params_dict.get(f"d_price_L{j}", 0.0) for j in range(1, K_price + 1)}
            gamma_pos = gamma_common
            gamma_neg = gamma_common

        # Select Beta based on the SCENARIO (The shock is fixed exogenous input)
        # Reference: "For an initial crude price increase... all beta_i are replaced by beta_i+" [cite: 749]
        if shock_type == 'positive':
            beta = {i: params_dict.get(f"d_cost_pos_L{i}", 0.0) for i in range(K_cost + 1)}
        elif shock_type == 'negative':
            beta = {i: params_dict.get(f"d_cost_neg_L{i}", 0.0) for i in range(K_cost + 1)}
        else:
            beta = {i: params_dict.get(f"d_cost_L{i}", 0.0) for i in range(K_cost + 1)}

        # 2. Compute Path (Recursive)
        # B[k] = Cumulative response at period k
        
        # Period 0: Instantaneous effect (Beta_0)
        B = [beta.get(0, 0.0)]
        
        for k in range(1, periods):
            # A. Direct Cost Effect (if within lag window)
            direct = beta.get(k, 0.0)
            
            # B. Error Correction Adjustment
            # Pulls variable back toward long-run equilibrium based on deviation in (k-1)
            ec_adjustment = theta * (B[k - 1] - phi_1_val)
            
            # C. Autoregressive / Price Inertia Feedback
            # Implements Eq A2: Sum of gamma * change_in_B over lags
            ar_feedback = 0.0
            
            # Look back j periods
            for j in range(1, min(K_price + 1, k + 1)):
                # Calculate the specific change in B that happened j periods ago
                # dB = B[k-j] - B[k-j-1]
                prev_val_lag = B[k - j - 1] if (k - j - 1) >= 0 else 0.0
                dB = B[k - j] - prev_val_lag
                
                # Apply Dynamic Gamma Selection [cite: 752]
                # If that past price change was positive, use gamma_pos.
                # If it was negative, use gamma_neg.
                term_pos = gamma_pos.get(j, 0.0) * max(dB, 0.0)
                term_neg = gamma_neg.get(j, 0.0) * min(dB, 0.0)
                
                ar_feedback += term_pos + term_neg
            
            # Sum components
            # Note: B[k-1] is added because B is cumulative
            current_B = B[k - 1] + direct + ec_adjustment + ar_feedback
            B.append(current_B)
            
        return np.array(B)

    # --- Execution & Bootstrapping ---
    
    # 1. Point Estimates
    params_dict = params.to_dict()
    
    if asym:
        point_est_pos = _compute_passthrough_path(params_dict, phi_1, 'positive')
        point_est_neg = _compute_passthrough_path(params_dict, phi_1, 'negative')
    else:
        point_est = _compute_passthrough_path(params_dict, phi_1, 'symmetric')
    
    # 2. Bootstrap
    rng = np.random.default_rng(seed)
    cov_values = cov.values if hasattr(cov, 'values') else cov
    
    # Draw parameters from multivariate normal distribution
    param_draws = rng.multivariate_normal(params.values, cov_values, size=n_boot)
    
    if asym:
        boot_paths_pos = np.zeros((n_boot, periods))
        boot_paths_neg = np.zeros((n_boot, periods))
        
        for i, draw in enumerate(param_draws):
            draw_dict = dict(zip(params.index, draw))
            boot_paths_pos[i, :] = _compute_passthrough_path(draw_dict, phi_1, 'positive')
            boot_paths_neg[i, :] = _compute_passthrough_path(draw_dict, phi_1, 'negative')
        
        # Calculate Percentiles
        alpha = 1 - ci_level
        return pd.DataFrame({
            "period": range(periods),
            "passthrough_positive": point_est_pos,
            "ci_lower_positive": np.percentile(boot_paths_pos, 100 * alpha / 2, axis=0),
            "ci_upper_positive": np.percentile(boot_paths_pos, 100 * (1 - alpha / 2), axis=0),
            "passthrough_negative": point_est_neg,
            "ci_lower_negative": np.percentile(boot_paths_neg, 100 * alpha / 2, axis=0),
            "ci_upper_negative": np.percentile(boot_paths_neg, 100 * (1 - alpha / 2), axis=0),
            "asymmetry": point_est_pos - point_est_neg,
            # Note: The paper calculates asymmetry cost as integral (sum of differences)
            # This column is just the difference in levels per period
        })
        
    else:
        boot_paths = np.zeros((n_boot, periods))
        
        for i, draw in enumerate(param_draws):
            draw_dict = dict(zip(params.index, draw))
            boot_paths[i, :] = _compute_passthrough_path(draw_dict, phi_1, 'symmetric')
            
        alpha = 1 - ci_level
        return pd.DataFrame({
            "period": range(periods),
            "passthrough": point_est,
            "ci_lower": np.percentile(boot_paths, 100 * alpha / 2, axis=0),
            "ci_upper": np.percentile(boot_paths, 100 * (1 - alpha / 2), axis=0),
        })


if __name__ == "__main__":
    # Example usage
    print("ECM-IV Estimation Module")
    print("=" * 40)
    print("\nExample usage:")
    print("""
    from ecm_iv import estimate_ecm_iv, compute_cumulative_passthrough
    
    # Symmetric estimation with differenced instruments
    model, fs_model, coint_params, df_est = estimate_ecm_iv(
        df=df_weekly,
        price='ln_rt',
        cost='ln_wh',
        K_cost=8,
        K_price=2,
        L=0,
        instruments_name=['flu'],
        instruments_lags=[2, 10, 20],  # Lags of differenced instrument
        asym=False,
        hac_lags=4,
        crop=False,
        ec_controls={'region': 'fe'},
        controls={'region': 'fe', 'month': 'fe'}
    )
    
    # Asymmetric estimation with different lag structures for pos/neg shocks
    model_asym, fs_asym, coint_asym, df_asym = estimate_ecm_iv(
        df=df_weekly,
        price='ln_rt',
        cost='ln_wh',
        K_cost=8,
        K_price=2,
        L=0,
        instruments_name=['flu'],
        instruments_lags_pos=[2, 4, 6],     # Fast: outbreak → price spike
        instruments_lags_neg=[15, 18, 20],  # Slow: recovery → flock rebuild
        asym=True,
        hac_lags=4,
        crop=False,
        ec_controls={'region': 'fe'},
        controls={'region': 'fe', 'month': 'fe'}
    )
    
    # Compute cumulative pass-through with bootstrap CIs
    pt_df = compute_cumulative_passthrough(
        model,ß
        phi_1=coint_params['phi_1'],
        K_cost=8,
        K_price=2,
        periods=30,
        n_boot=2000,
        ci_level=0.95
    )
    
    # Print results at selected horizons
    print("Cumulative pass-through at selected horizons:")
    for h in [1, 4, 8, 12, 20, 29]:
        row = pt_df.loc[h]
        print(f"  h={h:2d}: {row['passthrough']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    # For asymmetric model
    pt_asym = compute_cumulative_passthrough(
        model_asym,
        phi_1=coint_asym['phi_1'],
        K_cost=8,
        K_price=2,
        periods=30,
        n_boot=2000,
        ci_level=0.95,
        asym=True
    )
    
    print("\\nAsymmetric pass-through at selected horizons:")
    for h in [1, 4, 8, 12, 20, 29]:
        row = pt_asym.loc[h]
        print(f"  h={h:2d}: pos={row['passthrough_positive']:.3f}, neg={row['passthrough_negative']:.3f}, diff={row['asymmetry']:.3f}")
    """)