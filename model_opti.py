# %% MERGED REGIME PIPELINE + OPTIMIZER
#!/usr/bin/env python3
"""
Merged script:
    - Regime-aware risk decomposition pipeline
    - Randomised optimizer

Important change vs standalone model.py:
    - The decomposition pipeline is now an internal engine.
    - It no longer runs as a standalone script.
    - The optimizer is the only main entry point and is responsible
      for all outputs / exports / plots.

Author: Amin Tarbouch
"""

from __future__ import annotations

import sys
import re
import io
import time
import traceback
import warnings
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception as exc:
    HMM_AVAILABLE = False
    HMM_IMPORT_ERROR = exc

# ==========================================================
# Path resolution
# ==========================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from path_config import (
    MASTER_clean,
    OUTPUT_DIR,
    DEFAULT_SHEET,
)

MASTER_CLEAN_PATH: Path = OUTPUT_DIR / "MASTER_CLEAN.xlsx"
LATEX_OUTPUT_DIR: Path = OUTPUT_DIR / "data_cleaned"
SELECTED_FEATURES_PATH: Path = LATEX_OUTPUT_DIR / "selected_features.csv"
MODEL_RESULTS_DIR: Path = OUTPUT_DIR / "model_results"
OPTIMIZER_OUTPUT_DIR: Path = OUTPUT_DIR / "optimizer_results"
OPTIMIZER_DATA_DIR: Path = OPTIMIZER_OUTPUT_DIR / "data"
OPTIMIZER_PLOTS_DIR: Path = OPTIMIZER_OUTPUT_DIR / "plots"

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from IPython.display import display
except Exception:
    display = print


# ==========================================================
# Excel helper
# ==========================================================

def read_excel_sheet_safe(
    path: str | Path,
    sheet_name: int | str = 0,
    context: str = "Excel file",
) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as exc:
        msg = str(exc)
        if "Worksheet named" not in msg and "Worksheet index" not in msg:
            raise
        xls = pd.ExcelFile(path)
        print(f"Sheet {sheet_name!r} not found in {context}: {path}")
        print(f"Available sheet(s): {xls.sheet_names}")
        print("Falling back to the first sheet.")
        return pd.read_excel(path, sheet_name=0)


# ==========================================================
# Raw-column whitelist
# ==========================================================

ALLOWED_RAW_COLUMNS: tuple[str, ...] = (
    "DATE", "ISIN", "TICKER", "BBG_SECURITY", "MID_PRICE", "DIRTY_PRICE",
    "BID_PRICE", "ASK_PRICE", "Z_SPREAD_MID", "PAR_VALUE",
    "RATING_SP_IS_FALLBACK", "RATING_MOODY_IS_FALLBACK", "RATING_FITCH_IS_FALLBACK",
    "DESCRIPTION", "CCY", "PRICE", "MV", "ISSUE_DT", "MATURITY",
    "INDUSTRY_SECTOR", "COUNTRY", "COUNTRY_FULL_NAME", "ISSUER", "CNTRY_OF_RISK",
    "CURRENCY", "CPN_TYP", "CPN_FREQ", "COUPON", "PAYMENT_RANK", "CALLABLE",
    "PUTABLE", "SINKABLE", "EFF_START", "EFF_END", "LATAM_CDS_CREDIT_RISK",
    "LATAM_JPM_SPREAD_CREDIT_RISK", "MA3_VIX_INDEX", "MA3_UKX_INDEX",
    "MA3_MOVE_INDEX", "MA3_CDX_IG_CDSI_GEN_5Y_CORP",
    "MA3_ITRX_EUR_CDSI_GEN_5Y_CORP", "MA3_GDP_CQOQ_INDEX",
    "MA3_EUGNEMUQ_INDEX", "MA3_CPI_YOY_INDEX", "MA3_ECCPEMUY_INDEX",
    "MA3_BFCIUS_INDEX", "MA3_BFCIEU_INDEX", "MA3_STOXX600", "MA3_SP500",
    "MA3_MSCI_EM", "MA3_JPM_EMBI", "MA3_EPUCEMEC_INDEX", "MA3_CAFZUUEM_INDEX",
    "MA3_AONILS_INDEX", "MA3_GLCRPCR_INDEX", "MA3_GLCRCBV_INDEX",
    "MA3_GLCRNCB_INDEX", "MA3_GCILMANM_INDEX", "MA3_GCILNATL_INDEX",
    "MA3_CRPTBRAZ_INDEX", "MA3_CRPTCHIN_INDEX", "MA3_CRPTHOKO_INDEX",
    "MA3_CRPTMEXI_INDEX", "MA3_CRPTKORE_INDEX", "MA3_CRPTINDI_INDEX",
    "MA3_CRPTUNAE_INDEX", "MA3_LATAM_FX_DIRECTION", "MA3_LATAM_FX_VOL",
    "MA3_GPR_LEVEL", "MA3_GPR_MOMENTUM", "MA3_GPR_VOL", "MA3_GPR_ACTS_SHARE",
    "MA3_GPR_LATAM_LEVEL", "MA3_GPR_LATAM_MOMENTUM", "MA3_GPR_LATAM_VOL",
    "MA3_GPR_BRA_LEVEL", "MA3_GPR_BRA_MOMENTUM", "MA3_GPR_BRA_VOL",
    "MA3_GPR_CHL_LEVEL", "MA3_GPR_CHL_MOMENTUM", "MA3_GPR_CHL_VOL",
    "MA3_GPR_COL_LEVEL", "MA3_GPR_COL_MOMENTUM", "MA3_GPR_COL_VOL",
    "MA3_GPR_MEX_LEVEL", "MA3_GPR_MEX_MOMENTUM", "MA3_GPR_MEX_VOL",
    "MA3_GPR_PER_LEVEL", "MA3_GPR_PER_MOMENTUM", "MA3_GPR_PER_VOL",
    "MA3_GPR_CRI_LEVEL", "MA3_GPR_CRI_MOMENTUM", "MA3_GPR_CRI_VOL",
    "AGE_YEARS", "ISSUE_YEAR", "OBS_YEAR", "OBS_MONTH", "OBS_WEEK",
    "MID_PRICE_CALC", "BA_SPREAD", "HARD_CCY", "OPTION_CALLABLE", "OPTION_PUTABLE",
    "OPTION_SINKABLE", "SENIORITY_SR_UNSECURED", "SENIORITY_SUBORDINATED",
    "SENIORITY_1ST_LIEN", "SENIORITY_2ND_LIEN", "SENIORITY_SECURED",
    "SENIORITY_UNSECURED", "Z_SPREAD_BPS",
)

ALLOWED_RAW_COLUMN_SET: set[str] = set(ALLOWED_RAW_COLUMNS)

ALLOWED_INTERNAL_COLUMNS: set[str] = {
    "MODEL_TARGET", "MODEL_TARGET_LEVEL", "TARGET_LAG",
    "YEAR_WEEK", "YEAR_MONTH", "CNTRY_OF_RISK_FULL_NAME",
}

ALLOWED_DUMMY_PREFIXES: tuple[str, ...] = (
    "CNTRY_OF_RISK_", "CURRENCY_", "CCY_",
    "INDUSTRY_SECTOR_", "PAYMENT_RANK_",
)

DISALLOWED_DERIVED_MODEL_FEATURES: set[str] = {"TTM", "SIZE", "LOG_SIZE", "TIME_FE"}


def is_allowed_source_or_internal_column(column: str, keep_internal: bool = True) -> bool:
    name = str(column).upper().strip()
    if name in ALLOWED_RAW_COLUMN_SET:
        return True
    if keep_internal and name in ALLOWED_INTERNAL_COLUMNS:
        return True
    if name.startswith(ALLOWED_DUMMY_PREFIXES):
        return True
    return False


def is_allowed_model_input_feature(feature: str) -> bool:
    name = str(feature).upper().strip()
    if name in DISALLOWED_DERIVED_MODEL_FEATURES:
        return False
    return is_allowed_source_or_internal_column(name, keep_internal=False)


def restrict_dataframe_to_allowed_columns(
    df: pd.DataFrame,
    *,
    keep_internal: bool = True,
    context: str = "dataset",
) -> pd.DataFrame:
    keep_cols = [
        col for col in df.columns
        if is_allowed_source_or_internal_column(col, keep_internal=keep_internal)
    ]
    dropped = [col for col in df.columns if col not in keep_cols]
    if dropped:
        print(
            f"Column whitelist [{context}]: kept {len(keep_cols):,}/{len(df.columns):,} columns; "
            f"dropped {len(dropped):,} extra column(s)."
        )
        preview = dropped[:20]
        print("Dropped preview:", preview)
        if len(dropped) > len(preview):
            print(f"... plus {len(dropped) - len(preview):,} more.")
    return df.loc[:, keep_cols].copy()


# ==========================================================
# Feature taxonomy
# ==========================================================

class FeatureTaxonomy:
    GLOBAL: str = "Global"
    EMERGING: str = "Emerging"
    BONDS: str = "Bonds"
    UNKNOWN: str = "Unknown"

    GLOBAL_FEATURES: set[str] = {
        "MA3_UKX_INDEX", "MA3_STOXX600", "MA3_SP500",
        "MA3_CDX_IG_CDSI_GEN_5Y_CORP", "MA3_ITRX_EUR_CDSI_GEN_5Y_CORP",
        "MA3_GDP_CQOQ_INDEX", "MA3_ECCPEMUY_INDEX", "MA3_EUGNEMUQ_INDEX",
        "MA3_CPI_YOY_INDEX", "MA3_BFCIUS_INDEX", "MA3_BFCIEU_INDEX",
        "MA3_VIX_INDEX", "MA3_MOVE_INDEX", "MA3_AONILS_INDEX",
        "MA3_GLCRPCR_INDEX", "MA3_GLCRCBV_INDEX", "MA3_GLCRNCB_INDEX",
        "MA3_GCILMANM_INDEX", "MA3_GCILNATL_INDEX", "MA3_GPR_LEVEL",
        "OBS_YEAR", "OBS_MONTH", "OBS_WEEK", "YEAR_WEEK", "YEAR_MONTH",
        "TIME_FE", "TIME_FE_2021", "TIME_FE_2022", "TIME_FE_2023",
        "TIME_FE_2024", "TIME_FE_2025", "TIME_FE_2026",
    }

    EMERGING_FEATURES: set[str] = {
        "CNTRY_OF_RISK",
        "CNTRY_OF_RISK_CL", "CNTRY_OF_RISK_CO", "CNTRY_OF_RISK_CR",
        "CNTRY_OF_RISK_GT", "CNTRY_OF_RISK_MX", "CNTRY_OF_RISK_PE",
        "MA3_JPM_EMBI", "MA3_MSCI_EM", "MA3_EPUCEMEC_INDEX", "MA3_CAFZUUEM_INDEX",
        "MA3_CRPTBRAZ_INDEX", "MA3_CRPTCHIN_INDEX", "MA3_CRPTHOKO_INDEX",
        "MA3_CRPTMEXI_INDEX", "MA3_CRPTKORE_INDEX", "MA3_CRPTINDI_INDEX",
        "MA3_CRPTUNAE_INDEX", "MA3_GPR_LATAM_MOMENTUM",
        "LATAM_CDS_CREDIT_RISK", "LATAM_JPM_SPREAD_CREDIT_RISK",
    }

    BOND_FEATURES: set[str] = {
        "CURRENCY", "CCY", "CPN_TYP", "CPN_FREQ", "PAYMENT_RANK",
        "CALLABLE", "PUTABLE", "SINKABLE", "EFF_START", "EFF_END",
        "ISSUE_YEAR", "MID_PRICE_CALC", "SIZE", "LOG_SIZE", "HARD_CCY",
        "MID_PRICE", "DIRTY_PRICE", "BID_PRICE", "ASK_PRICE", "BA_SPREAD",
        "OPTION_CALLABLE", "OPTION_PUTABLE", "OPTION_SINKABLE",
        "SENIORITY_SUBORDINATED", "SENIORITY_1ST_LIEN", "SENIORITY_2ND_LIEN",
        "SENIORITY_SECURED", "SENIORITY_UNSECURED", "SENIORITY_SR_UNSECURED",
        "Z_SPREAD_BPS", "Z_SPREAD_MID", "TARGET_LAG", "MODEL_TARGET",
        "COUPON", "PAR_VAL", "PAR_VALUE", "TTM", "AGE_YEARS",
        "PAYMENT_RANK_SUBORDINATED",
        "INDUSTRY_SECTOR_CONSUMER, CYCLICAL", "INDUSTRY_SECTOR_COMMUNICATIONS",
        "INDUSTRY_SECTOR_CONSUMER, NON_CYCLICAL", "INDUSTRY_SECTOR_ENERGY",
        "INDUSTRY_SECTOR_FINANCIAL", "INDUSTRY_SECTOR_INDUSTRIAL",
        "INDUSTRY_SECTOR_UTILITIES",
    }

    EMERGING_PREFIXES: tuple[str, ...] = (
        "CNTRY_OF_RISK_", "COUNTRY_", "COUNTRY_FULL_NAME_",
        "MA3_GPR_LATAM", "MA3_GPR_BRA", "MA3_GPR_BR",
        "MA3_GPR_CHL", "MA3_GPR_CL", "MA3_GPR_COL", "MA3_GPR_CO",
        "MA3_GPR_MEX", "MA3_GPR_MX", "MA3_GPR_PER", "MA3_GPR_PE",
        "MA3_GPR_CRI", "MA3_GPR_CR", "MA3_LATAM", "LATAM_",
        "LATAM_CDS_", "LATAM_JPM_", "LATAM_CREDIT_",
    )

    EMERGING_TOKENS: tuple[str, ...] = (
        "EMBI", "MSCI_EM", "EPUCEMEC", "CAFZUUEM", "LATAM",
        "CRPTBRAZ", "CRPTCHIN", "CRPTHOKO", "CRPTMEXI",
        "CRPTKORE", "CRPTINDI", "CRPTUNAE",
    )

    BOND_PREFIXES: tuple[str, ...] = (
        "CURRENCY_", "CCY_", "PAYMENT_RANK_", "INDUSTRY_SECTOR_",
        "SECTOR_LEVEL_2_", "MERRILL_SECTOR_3_", "SENIORITY_", "OPTION_",
    )

    GLOBAL_PREFIXES: tuple[str, ...] = ("TIME_FE_",)

    @classmethod
    def classify(cls, feature: str) -> str:
        name = str(feature).upper().strip()
        base = name[4:] if name.startswith("MA3_") else name
        if name in cls.EMERGING_FEATURES:
            return cls.EMERGING
        if name in cls.GLOBAL_FEATURES:
            return cls.GLOBAL
        if name in cls.BOND_FEATURES:
            return cls.BONDS
        if name.startswith(cls.EMERGING_PREFIXES):
            return cls.EMERGING
        if any(t in name for t in cls.EMERGING_TOKENS):
            return cls.EMERGING
        if name.startswith("MA3_GPR_") or base.startswith("GPR_"):
            return cls.GLOBAL
        if name.startswith(cls.BOND_PREFIXES):
            return cls.BONDS
        if name.startswith(cls.GLOBAL_PREFIXES):
            return cls.GLOBAL
        if name.startswith("MA3_"):
            return cls.GLOBAL
        return cls.BONDS

    @classmethod
    def is_global(cls, f: str) -> bool:
        return cls.classify(f) == cls.GLOBAL

    @classmethod
    def is_emerging(cls, f: str) -> bool:
        return cls.classify(f) == cls.EMERGING

    @classmethod
    def is_bond(cls, f: str) -> bool:
        return cls.classify(f) == cls.BONDS


# ==========================================================
# Feature block helpers
# ==========================================================

MACRO_PREFIXES: tuple[str, ...] = ("MA3_",)
MACRO_EXACT: set[str] = {"OBS_YEAR", "OBS_MONTH", "OBS_WEEK"}

EM_PREFIXES: tuple[str, ...] = (
    "CNTRY_OF_RISK_", "COUNTRY_", "COUNTRY_FULL_NAME_", "LATAM_",
)
EM_TOKENS: tuple[str, ...] = (
    "EMBI", "MSCI_EM", "CRPT", "GPR_LATAM", "EPUCEMEC", "CAFZUUEM",
    "CDS_CREDIT_RISK", "JPM_SPREAD_CREDIT_RISK",
)
BOND_PREFIXES_HELPER: tuple[str, ...] = (
    "CCY_", "CURRENCY_", "INDUSTRY_SECTOR_", "SECTOR_LEVEL_2_",
    "MERRILL_SECTOR_3_", "PAYMENT_RANK_", "TIME_FE_",
)
BOND_EXACT: set[str] = {
    "COUPON", "CPN_FREQ", "PAR_VAL", "PAR_VALUE", "MV",
    "TTM", "AGE_YEARS", "ISSUE_YEAR",
    "BA_SPREAD", "SIZE", "LOG_SIZE", "HARD_CCY",
    "MID_PRICE", "DIRTY_PRICE", "MID_PRICE_CALC",
    "SENIORITY_SR_UNSECURED", "SENIORITY_SUBORDINATED",
    "SENIORITY_1ST_LIEN", "SENIORITY_2ND_LIEN",
    "SENIORITY_SECURED", "SENIORITY_UNSECURED",
}


def fallback_classify_feature(feature: str) -> str:
    name = str(feature).upper().strip()
    if name.startswith("PC") or name.startswith("REGIME_"):
        return "Latent"
    if name in MACRO_EXACT or name.startswith(MACRO_PREFIXES):
        return "Emerging" if any(t in name for t in EM_TOKENS) else "Global"
    if name.startswith(EM_PREFIXES) or any(t in name for t in EM_TOKENS):
        return "Emerging"
    if name.startswith(BOND_PREFIXES_HELPER) or name in BOND_EXACT:
        return "Bonds"
    return "Bonds"


def normalize_feature_type(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"global", "macro"}:
        return "Global"
    if v in {"emerging", "em", "country", "country-risk"}:
        return "Emerging"
    if v in {"bonds", "bond", "issuer", "instrument"}:
        return "Bonds"
    if v in {"latent", "pca", "hmm", "regime"}:
        return "Latent"
    return "Unknown"


# ==========================================================
# Universe-specific feature filters
# ==========================================================

LATAM_SPECIFIC_TOKENS: tuple[str, ...] = (
    "LATAM", "GPR_LATAM", "LATAM_CDS", "LATAM_JPM", "LATAM_CREDIT",
    "GPR_BRA", "GPR_BR", "GPR_CHL", "GPR_CL", "GPR_COL", "GPR_CO",
    "GPR_MEX", "GPR_MX", "GPR_PER", "GPR_PE", "GPR_CRI", "GPR_CR",
    "GPR_ARG", "GPR_AR",
    "CRPTBRAZ", "CRPTMEXI", "CRPTCHIL", "CRPTCOLO",
    "CRPTPERU", "CRPTARG", "CRPTCOST",
)

LATAM_COUNTRY_DUMMY_NAMES: set[str] = {
    "CNTRY_OF_RISK_AR", "CNTRY_OF_RISK_BO", "CNTRY_OF_RISK_BR", "CNTRY_OF_RISK_CL",
    "CNTRY_OF_RISK_CO", "CNTRY_OF_RISK_CR", "CNTRY_OF_RISK_EC", "CNTRY_OF_RISK_SV",
    "CNTRY_OF_RISK_GT", "CNTRY_OF_RISK_HN", "CNTRY_OF_RISK_MX", "CNTRY_OF_RISK_NI",
    "CNTRY_OF_RISK_PA", "CNTRY_OF_RISK_PY", "CNTRY_OF_RISK_PE", "CNTRY_OF_RISK_UY",
    "CNTRY_OF_RISK_VE",
}


def is_latam_specific_feature(feature: str) -> bool:
    name = str(feature).upper().strip()
    return name.startswith("LATAM_") or any(t in name for t in LATAM_SPECIFIC_TOKENS)


def is_country_dummy_feature(feature: str) -> bool:
    name = str(feature).upper().strip()
    return (
        name.startswith("CNTRY_OF_RISK_")
        or name.startswith("COUNTRY_")
        or name.startswith("COUNTRY_FULL_NAME_")
    )


def sanitize_selected_features_file(
    features_path: str | Path,
    use_latam_specific_inputs: bool,
    overwrite: bool = True,
) -> pd.DataFrame:
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Selected-features file not found: {path}")

    features = pd.read_csv(path)
    features.columns = [str(c).upper().strip() for c in features.columns]
    if "FEATURE" not in features.columns:
        raise KeyError("selected_features.csv must contain a FEATURE column.")
    features["FEATURE"] = features["FEATURE"].astype(str).str.upper().str.strip()

    before = len(features)
    allowed_mask = features["FEATURE"].map(is_allowed_model_input_feature)
    disallowed = features.loc[~allowed_mask, "FEATURE"].tolist()
    features = features.loc[allowed_mask].copy()
    if disallowed:
        print(f"Column whitelist: removed {before - len(features):,} feature(s) outside allowed universe.")
        print("Preview:", disallowed[:20])

    if use_latam_specific_inputs:
        print("LATAM-specific inputs are allowed.")
        if overwrite:
            features.to_csv(path, index=False)
        return features

    before = len(features)
    keep_mask = ~features["FEATURE"].map(is_latam_specific_feature)
    filtered = features.loc[keep_mask].copy()
    removed = features.loc[~keep_mask, "FEATURE"].tolist()
    if overwrite:
        filtered.to_csv(path, index=False)
    print(f"ALL_EM mode: removed {before - len(filtered):,} LATAM-specific features.")
    for f in removed:
        print(f"  - {f}")
    return filtered


# ==========================================================
# Metric helpers
# ==========================================================

def _valid_metric_pair(y_true, y_pred):
    yt = pd.Series(y_true).astype(float)
    yp = pd.Series(y_pred).astype(float)
    if len(yp) != len(yt):
        yp.index = yt.index[:len(yp)]
    pair = pd.concat([yt.rename("y_true"), yp.rename("y_pred")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        raise ValueError("No valid finite observations for metric computation.")
    return pair["y_true"], pair["y_pred"]


def compute_rmse(y_true, y_pred) -> float:
    yt, yp = _valid_metric_pair(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def compute_mae(y_true, y_pred) -> float:
    yt, yp = _valid_metric_pair(y_true, y_pred)
    return float(mean_absolute_error(yt, yp))


# ==========================================================
# RiskDecompositionConfig
# ==========================================================

@dataclass
class RiskDecompositionConfig:
    data_path: str
    features_path: str
    sheet_name: int | str = 0

    target_col: str = "MODEL_TARGET"
    regime_target_col: Optional[str] = None
    source_target_col: str = "Z_SPREAD_MID"
    target_mode_label: str = "weekly_pct_change"

    id_col: str = "ISIN"
    date_col: str = "DATE"
    period_col: Optional[str] = None

    universe_mode: str = "LATAM"
    use_latam_specific_inputs: bool = True

    winsorize_target: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99

    ols_cov_type: str = "HC3"
    standardize_features: bool = True

    n_pca_components: int = 3
    min_isins_per_period: int = 5

    n_regimes: int = 3
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 500
    hmm_tol: float = 1e-4
    hmm_n_starts: int = 20
    hmm_random_state: int = 42
    hmm_include_pc1: bool = True
    hmm_features: list[str] = field(default_factory=list)
    hmm_auto_select_features: bool = True
    hmm_max_features: int = 8

    smooth_regime_states: bool = True
    min_regime_duration: int = 3

    output_dir: str = str(MODEL_RESULTS_DIR)

    def __post_init__(self) -> None:
        self.data_path = str(self.data_path)
        self.features_path = str(self.features_path)
        self.universe_mode = str(self.universe_mode).upper().strip()
        self.target_col = str(self.target_col).upper().strip()
        self.source_target_col = str(self.source_target_col).upper().strip()
        self.target_mode_label = str(self.target_mode_label).lower().strip()
        self.id_col = str(self.id_col).upper().strip()
        self.date_col = str(self.date_col).upper().strip()

        if self.regime_target_col is not None:
            self.regime_target_col = str(self.regime_target_col).upper().strip()
        if self.period_col is not None:
            self.period_col = str(self.period_col).upper().strip()

        if self.universe_mode not in {"LATAM", "ALL_EM"}:
            raise ValueError("universe_mode must be 'LATAM' or 'ALL_EM'.")
        if self.universe_mode == "ALL_EM" and self.use_latam_specific_inputs:
            raise ValueError("ALL_EM mode requires use_latam_specific_inputs=False.")
        if self.source_target_col not in {"Z_SPREAD_MID", "Z_SPREAD_BPS"}:
            raise ValueError(
                f"source_target_col must be 'Z_SPREAD_MID' or 'Z_SPREAD_BPS'. Got: {self.source_target_col!r}"
            )
        if not (0 <= self.winsorize_lower < self.winsorize_upper <= 1):
            raise ValueError("Winsorization bounds must satisfy 0 <= lower < upper <= 1.")
        if self.n_regimes != 3:
            raise ValueError("Exactly 3 regimes required: Calm, Transitional, Stress.")
        if self.hmm_covariance_type not in {"diag", "full", "tied", "spherical"}:
            raise ValueError(f"Invalid HMM covariance type: {self.hmm_covariance_type!r}")
        if self.min_regime_duration < 1:
            raise ValueError("min_regime_duration must be >= 1.")


# ==========================================================
# RegimeRiskDecompositionPipeline
# ==========================================================

class RegimeRiskDecompositionPipeline:
    """
    Interpretable regime-aware risk decomposition engine.

    Important:
        - This class is now intended to be called by the optimizer.
        - It does not act as a standalone output-producing script anymore.
        - Exports / plots are only triggered by the optimizer.
    """

    REGIME_ORDER = ["Calm", "Transitional", "Stress"]

    def __init__(self, config: RiskDecompositionConfig) -> None:
        self.config = config

        self.df: Optional[pd.DataFrame] = None
        self.df_model: Optional[pd.DataFrame] = None
        self.selected_features: list[str] = []
        self.feature_types: dict[str, str] = {}
        self.feature_blocks: pd.Series | None = None

        self.period_col: str = ""
        self.period_index: Optional[pd.Index] = None
        self.X_raw: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.y_regime: Optional[pd.Series] = None
        self.regime_residuals: Optional[pd.Series] = None
        self.scaler: Optional[StandardScaler] = None

        self.baseline_model = None
        self.baseline_summary_table: Optional[pd.DataFrame] = None
        self.baseline_contributions: Optional[pd.DataFrame] = None
        self.baseline_block_shares: Optional[pd.DataFrame] = None
        self.baseline_top_coefficients: Optional[pd.DataFrame] = None

        self.residuals: Optional[pd.Series] = None
        self.resid_wide: Optional[pd.DataFrame] = None
        self.pca_model: Optional[PCA] = None
        self.pc_factors: Optional[pd.DataFrame] = None
        self.pca_variance_table: Optional[pd.DataFrame] = None

        self.hmm_model: Optional[GaussianHMM] = None
        self.hmm_features_used: list[str] = []
        self.regime_probs: Optional[pd.DataFrame] = None
        self.regime_states_raw: Optional[pd.Series] = None
        self.regime_states: Optional[pd.Series] = None
        self.regime_durations: Optional[pd.DataFrame] = None
        self.regime_transition_matrix: Optional[pd.DataFrame] = None
        self.regime_adjustment_count: int = 0

        self.hmm_design: Optional[pd.DataFrame] = None
        self.hmm_decomp_model = None
        self.hmm_components: Optional[pd.DataFrame] = None
        self.hmm_component_shares: Optional[pd.DataFrame] = None
        self.regime_component_profile: Optional[pd.DataFrame] = None
        self.model_comparison: Optional[pd.DataFrame] = None
        self.regime_interaction_table: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run_all(self, show_plots: bool = True) -> None:
        print("\n" + "=" * 78)
        print("REGIME-AWARE RISK DECOMPOSITION PIPELINE")
        print("=" * 78)
        self.load_data()
        self.prepare_model_matrix()
        self.fit_baseline_regression()
        self.run_residual_pca()
        self.fit_compact_hmm()
        self.fit_hmm_conditioned_decomposition()
        self.build_model_comparison()
        self.print_key_tables()
        if show_plots:
            self.plot_all(show=True)
        print("\nPipeline complete.")

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    def load_data(self) -> None:
        data_path = Path(self.config.data_path)
        features_path = Path(self.config.features_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"MASTER_CLEAN.xlsx not found:\n  {data_path}\n"
                "Run data_cleaning.py first."
            )
        if not features_path.exists():
            raise FileNotFoundError(
                f"selected_features.csv not found:\n  {features_path}\n"
                "Run data_cleaning.py first."
            )

        if data_path.suffix.lower() in {".xlsx", ".xls"}:
            df = read_excel_sheet_safe(
                data_path,
                sheet_name=self.config.sheet_name,
                context="MASTER_CLEAN",
            )
        elif data_path.suffix.lower() == ".csv":
            df = pd.read_csv(data_path)
        else:
            df = pd.read_parquet(data_path)

        df.columns = [str(c).upper().strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = restrict_dataframe_to_allowed_columns(df, keep_internal=True, context="MASTER_CLEAN")

        features = pd.read_csv(features_path)
        features.columns = [str(c).upper().strip() for c in features.columns]
        if "FEATURE" not in features.columns:
            raise KeyError("selected_features.csv must contain a FEATURE column.")
        features["FEATURE"] = features["FEATURE"].astype(str).str.upper().str.strip()

        feature_types = (
            {row["FEATURE"]: normalize_feature_type(row["FEATURE_TYPE"]) for _, row in features.iterrows()}
            if "FEATURE_TYPE" in features.columns else {}
        )

        selected = [
            f for f in features["FEATURE"].tolist()
            if f in df.columns and is_allowed_model_input_feature(f)
        ]

        if not self.config.use_latam_specific_inputs:
            before = len(selected)
            selected = [f for f in selected if not is_latam_specific_feature(f)]
            print(f"ALL_EM mode: excluded {before - len(selected):,} LATAM-specific variables.")

        if not selected:
            raise ValueError("No selected features found in MASTER_CLEAN after filtering.")

        feature_types = {f: t for f, t in feature_types.items() if f in selected}
        for f in selected:
            if feature_types.get(f, "Unknown") == "Unknown":
                feature_types[f] = fallback_classify_feature(f)

        self.df = df
        self.selected_features = selected
        self.feature_types = feature_types

        print(f"Loaded MASTER_CLEAN : {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"Selected features   : {len(selected):,}")
        print("Feature blocks:")
        print(pd.Series({f: feature_types[f] for f in selected}).value_counts().to_string())

    # ----------------------------------------------------------
    # Prepare model matrix
    # ----------------------------------------------------------

    def _infer_period_col(self, df: pd.DataFrame) -> str:
        if self.config.period_col and self.config.period_col in df.columns:
            return self.config.period_col
        if "YEAR_WEEK" in df.columns:
            return "YEAR_WEEK"
        if "YEAR_MONTH" in df.columns:
            return "YEAR_MONTH"
        if self.config.date_col in df.columns:
            df[self.config.date_col] = pd.to_datetime(df[self.config.date_col], errors="coerce")
            iso = df[self.config.date_col].dt.isocalendar()
            df["YEAR_WEEK"] = (
                iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
            )
            return "YEAR_WEEK"
        raise KeyError("Could not infer period column. Expected YEAR_WEEK, YEAR_MONTH, or DATE.")

    def prepare_model_matrix(self) -> None:
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

        df = self.df.copy()
        period_col = self._infer_period_col(df)
        self.period_col = period_col

        regime_target_col = self.config.regime_target_col or self.config.target_col

        required = [self.config.target_col, regime_target_col, period_col]
        if self.config.id_col in df.columns:
            required.append(self.config.id_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        y_raw = pd.to_numeric(df[self.config.target_col], errors="coerce")
        y_regime_raw = pd.to_numeric(df[regime_target_col], errors="coerce")

        df["RAW_MODEL_TARGET"] = y_raw
        df["RAW_REGIME_TARGET"] = y_regime_raw
        valid = y_raw.notna() & y_regime_raw.notna()

        X_raw = df.loc[valid, self.selected_features].apply(pd.to_numeric, errors="coerce")
        y = y_raw.loc[valid].astype(float)
        y_regime = y_regime_raw.loc[valid].astype(float)
        df_model = df.loc[valid].copy()

        missing_share = X_raw.isna().mean()
        keep = missing_share[missing_share < 0.80].index.tolist()
        X_raw = X_raw[keep].fillna(X_raw[keep].median(numeric_only=True))
        nunique = X_raw.nunique(dropna=True)
        X_raw = X_raw[nunique[nunique > 1].index.tolist()].copy()

        if self.config.winsorize_target:
            lo, hi = y.quantile(self.config.winsorize_lower), y.quantile(self.config.winsorize_upper)
            y_w = y.clip(lo, hi)
            print(
                f"Final model target winsorized "
                f"{self.config.winsorize_lower:.1%}/{self.config.winsorize_upper:.1%}: "
                f"{int((y.ne(y_w)).sum()):,} rows clipped"
            )
            y = y_w
            if regime_target_col == self.config.target_col:
                y_regime = y.copy()
            else:
                rlo, rhi = y_regime.quantile(self.config.winsorize_lower), y_regime.quantile(self.config.winsorize_upper)
                y_regime_w = y_regime.clip(rlo, rhi)
                print(
                    f"Regime/HMM target winsorized "
                    f"{self.config.winsorize_lower:.1%}/{self.config.winsorize_upper:.1%}: "
                    f"{int((y_regime.ne(y_regime_w)).sum()):,} rows clipped"
                )
                y_regime = y_regime_w

        if self.config.standardize_features:
            self.scaler = StandardScaler()
            X = pd.DataFrame(
                self.scaler.fit_transform(X_raw),
                columns=X_raw.columns,
                index=X_raw.index,
            )
        else:
            X = X_raw.copy()

        feature_blocks = pd.Series(
            {c: self.feature_types.get(c, fallback_classify_feature(c)) for c in X.columns},
            name="Block",
        )

        sort_cols = [period_col] + ([self.config.id_col] if self.config.id_col in df_model.columns else [])
        df_model = df_model.loc[X.index].copy()
        df_model["MODEL_TARGET_USED"] = y
        df_model["REGIME_TARGET_USED"] = y_regime
        order = df_model.sort_values(sort_cols).index

        self.df_model = df_model.loc[order].copy()
        self.X_raw = X_raw.loc[order].copy()
        self.X = X.loc[order].copy()
        self.y = y.loc[order].copy()
        self.y_regime = y_regime.loc[order].copy()
        self.feature_blocks = feature_blocks.loc[self.X.columns]
        self.period_index = self.df_model[period_col].astype(str)

        print(f"Model sample   : {len(self.y):,} observations")
        print(f"Periods        : {self.period_index.nunique():,} ({period_col})")
        if self.config.id_col in self.df_model.columns:
            print(f"ISINs          : {self.df_model[self.config.id_col].nunique():,}")
        print(f"Final target   : {self.config.target_col}")
        print(f"Regime target  : {regime_target_col}")
        print(f"Features       : {self.X.shape[1]:,}")
        display(pd.concat([
            self.y.describe().rename("MODEL_TARGET_USED"),
            self.y_regime.describe().rename("REGIME_TARGET_USED"),
        ], axis=1).T.round(4))

    # ----------------------------------------------------------
    # Baseline OLS
    # ----------------------------------------------------------

    def fit_baseline_regression(self) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("Call prepare_model_matrix() first.")

        X_const = sm.add_constant(self.X, has_constant="add")
        model = sm.OLS(self.y, X_const).fit(cov_type=self.config.ols_cov_type)
        self.baseline_model = model

        fitted = pd.Series(model.fittedvalues, index=self.X.index, name="Baseline_fitted")
        resid = pd.Series(self.y - fitted, index=self.X.index, name="Baseline_residual")
        self.residuals = resid

        coefs = model.params.drop("const", errors="ignore")
        contrib_raw = self.X.mul(coefs, axis=1)

        block_contrib = pd.DataFrame(index=self.X.index)
        for block in ["Global", "Emerging", "Bonds"]:
            cols = self.feature_blocks[self.feature_blocks.eq(block)].index.intersection(contrib_raw.columns)
            block_contrib[block] = contrib_raw[cols].sum(axis=1) if len(cols) else 0.0
        block_contrib["Intercept"] = float(model.params.get("const", 0.0))
        block_contrib["Fitted"] = fitted
        block_contrib["Actual"] = self.y
        block_contrib["Residual"] = resid
        self.baseline_contributions = block_contrib

        abs_means = block_contrib[["Global", "Emerging", "Bonds"]].abs().mean()
        shares = (abs_means / abs_means.sum()).rename("Share_of_explained_abs_contribution")
        self.baseline_block_shares = shares.reset_index().rename(columns={"index": "Risk_block"})

        self.baseline_top_coefficients = pd.DataFrame({
            "Feature": coefs.index,
            "Coefficient": coefs.values,
            "Abs_coefficient": np.abs(coefs.values),
            "Block": [self.feature_blocks.get(f, fallback_classify_feature(f)) for f in coefs.index],
        }).sort_values("Abs_coefficient", ascending=False)

        self.baseline_summary_table = pd.DataFrame([{
            "Model": "Baseline OLS: observed risk blocks only",
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj,
            "RMSE": compute_rmse(self.y, fitted),
            "MAE": compute_mae(self.y, fitted),
            "N_obs": int(model.nobs),
            "N_features": int(self.X.shape[1]),
        }])

        print("\nBaseline OLS fitted.")
        display(self.baseline_summary_table.round(4))
        print("Baseline block contribution shares:")
        display(self.baseline_block_shares.round(4))

    # ----------------------------------------------------------
    # Residual PCA
    # ----------------------------------------------------------

    def run_residual_pca(self) -> None:
        if self.residuals is None or self.df_model is None:
            raise RuntimeError("Run fit_baseline_regression() first.")
        if self.config.id_col not in self.df_model.columns:
            raise KeyError(f"{self.config.id_col} is required for residual PCA.")

        regime_target = self.y_regime if self.y_regime is not None else self.y

        if (
            self.config.regime_target_col is not None
            and self.config.regime_target_col != self.config.target_col
        ):
            X_const = sm.add_constant(self.X, has_constant="add")
            regime_model = sm.OLS(regime_target, X_const).fit(cov_type=self.config.ols_cov_type)
            regime_residuals = pd.Series(
                regime_target - pd.Series(regime_model.fittedvalues, index=self.X.index),
                index=self.X.index,
            )
            print(f"Residual PCA/HMM uses regime target: {self.config.regime_target_col}")
        else:
            regime_residuals = self.residuals
            print(f"Residual PCA/HMM uses model target: {self.config.target_col}")

        self.regime_residuals = regime_residuals
        valid = regime_residuals.notna() & regime_target.notna()

        tmp = pd.DataFrame({
            "Period": self.df_model.loc[valid, self.period_col].astype(str),
            "ISIN": self.df_model.loc[valid, self.config.id_col].astype(str),
            "Residual": regime_residuals.loc[valid],
            "Target": regime_target.loc[valid],
        })
        resid_wide = tmp.pivot_table(
            index="Period", columns="ISIN", values="Residual", aggfunc="mean"
        )
        resid_wide = resid_wide.loc[
            resid_wide.notna().sum(axis=1) >= self.config.min_isins_per_period
        ]
        resid_wide = resid_wide.loc[:, resid_wide.notna().sum(axis=0) >= 3]
        filled = resid_wide.apply(lambda s: s.fillna(s.median()), axis=0).fillna(0.0)

        E = StandardScaler().fit_transform(filled)
        n_comp = min(self.config.n_pca_components, E.shape[0], E.shape[1])
        if n_comp < 1:
            raise ValueError("Residual panel is too small for PCA.")

        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(E)
        pc = pd.DataFrame(
            scores, index=filled.index,
            columns=[f"PC{i+1}" for i in range(n_comp)]
        )

        period_target = tmp.groupby("Period")["Target"].mean().reindex(pc.index)
        corr = pc["PC1"].corr(period_target)
        if pd.notna(corr) and corr < 0:
            pc["PC1"] *= -1

        self.resid_wide = resid_wide
        self.pca_model = pca
        self.pc_factors = pc
        self.pca_variance_table = pd.DataFrame({
            "Component": pc.columns,
            "Explained_variance_ratio": pca.explained_variance_ratio_,
            "Cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        })

        print("\nResidual PCA complete.")
        display(self.pca_variance_table.round(4))

    # ----------------------------------------------------------
    # HMM
    # ----------------------------------------------------------

    def _period_level_feature_matrix(self) -> pd.DataFrame:
        if self.df_model is None or self.X_raw is None:
            raise RuntimeError("Model data not prepared.")
        tmp = self.X_raw.copy()
        tmp["Period"] = self.df_model[self.period_col].astype(str).values
        return tmp.groupby("Period").mean(numeric_only=True)

    def _select_hmm_features(self, period_features: pd.DataFrame) -> list[str]:
        def allowed(name: str) -> bool:
            name = str(name).upper().strip()
            if is_country_dummy_feature(name):
                return False
            if not self.config.use_latam_specific_inputs and is_latam_specific_feature(name):
                return False
            return True

        explicit = [
            f.upper().strip() for f in self.config.hmm_features
            if f.upper().strip() in period_features.columns and allowed(f)
        ]
        if explicit:
            return explicit[:self.config.hmm_max_features]

        priority = [
            "MA3_VIX_INDEX", "MA3_MOVE_INDEX", "MA3_JPM_EMBI",
            "MA3_MSCI_EM", "MA3_BFCIUS_INDEX", "MA3_BFCIEU_INDEX", "MA3_GPR_LEVEL",
        ]
        if self.config.use_latam_specific_inputs:
            priority.extend([
                "LATAM_CDS_CREDIT_RISK", "LATAM_JPM_SPREAD_CREDIT_RISK",
                "MA3_GPR_LATAM_MOMENTUM",
            ])

        chosen = [f for f in priority if f in period_features.columns and allowed(f)]

        if self.config.hmm_auto_select_features and len(chosen) < self.config.hmm_max_features:
            pc1 = self.pc_factors["PC1"] if self.pc_factors is not None else None
            candidates = []
            for c in period_features.columns:
                if c in chosen or not allowed(c):
                    continue
                if self.feature_types.get(c, fallback_classify_feature(c)) not in {"Global", "Emerging"}:
                    continue
                s = period_features[c].reindex(
                    self.pc_factors.index if self.pc_factors is not None else period_features.index
                )
                if s.nunique(dropna=True) < 4:
                    continue
                score = abs(s.corr(pc1)) if pc1 is not None else s.std()
                if pd.notna(score):
                    candidates.append((c, score))
            for c, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
                chosen.append(c)
                if len(chosen) >= self.config.hmm_max_features:
                    break

        return chosen[:self.config.hmm_max_features]

    def _fit_hmm_with_restarts(self, Y_scaled: np.ndarray) -> GaussianHMM:
        if not HMM_AVAILABLE:
            raise ImportError(
                f"hmmlearn not installed. Run: pip install hmmlearn\n"
                f"Original error: {HMM_IMPORT_ERROR}"
            )
        best_model, best_score = None, -np.inf
        rng = np.random.default_rng(self.config.hmm_random_state)
        seeds = rng.integers(0, 1_000_000, size=self.config.hmm_n_starts)
        for seed in seeds:
            m = GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type=self.config.hmm_covariance_type,
                n_iter=self.config.hmm_n_iter,
                tol=self.config.hmm_tol,
                random_state=int(seed),
            )
            try:
                m.fit(Y_scaled)
                sc = m.score(Y_scaled)
                if np.isfinite(sc) and sc > best_score:
                    best_model, best_score = m, sc
            except Exception:
                continue
        if best_model is None:
            raise RuntimeError(
                "All HMM random starts failed. "
                "Try fewer HMM features or covariance_type='diag'."
            )
        return best_model

    @staticmethod
    def _smooth_label_sequence(labels: pd.Series, min_duration: int) -> tuple[pd.Series, int]:
        if min_duration <= 1 or labels.empty:
            return labels.copy(), 0
        out, n_adjusted = labels.copy().astype(str), 0
        changed, iteration = True, 0
        while changed and iteration < 10:
            changed, iteration = False, iteration + 1
            values, runs, start = out.tolist(), [], 0
            for i in range(1, len(values) + 1):
                if i == len(values) or values[i] != values[start]:
                    runs.append((start, i - 1, values[start], i - start))
                    start = i
            for idx, (s, e, lab, dur) in enumerate(runs):
                if dur >= min_duration:
                    continue
                prev_lab = runs[idx - 1][2] if idx > 0 else None
                next_lab = runs[idx + 1][2] if idx < len(runs) - 1 else None
                if prev_lab is not None and next_lab is not None and prev_lab == next_lab:
                    replacement = prev_lab
                elif prev_lab is not None and next_lab is not None:
                    replacement = prev_lab if runs[idx - 1][3] >= runs[idx + 1][3] else next_lab
                elif prev_lab is not None:
                    replacement = prev_lab
                elif next_lab is not None:
                    replacement = next_lab
                else:
                    replacement = lab
                if replacement != lab:
                    out.iloc[s:e + 1] = replacement
                    n_adjusted += dur
                    changed = True
        return out, n_adjusted

    @staticmethod
    def _duration_table(states: pd.Series) -> pd.DataFrame:
        values, rows, start = states.astype(str).tolist(), [], 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                rows.append({"Regime": values[start], "Duration": i - start})
                start = i
        if not rows:
            return pd.DataFrame(
                columns=["Regime", "N_runs", "Mean_duration", "Median_duration", "Max_duration"]
            )
        return (
            pd.DataFrame(rows).groupby("Regime")["Duration"]
            .agg(N_runs="count", Mean_duration="mean", Median_duration="median", Max_duration="max")
            .reindex(RegimeRiskDecompositionPipeline.REGIME_ORDER)
            .dropna(how="all").reset_index()
        )

    @staticmethod
    def _transition_matrix(states: pd.Series) -> pd.DataFrame:
        labs = RegimeRiskDecompositionPipeline.REGIME_ORDER
        mat = pd.DataFrame(0.0, index=labs, columns=labs)
        s = states.astype(str).tolist()
        for a, b in zip(s[:-1], s[1:]):
            if a in mat.index and b in mat.columns:
                mat.loc[a, b] += 1
        return mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    def fit_compact_hmm(self) -> None:
        if self.pc_factors is None:
            raise RuntimeError("Run run_residual_pca() first.")

        period_features = self._period_level_feature_matrix().reindex(self.pc_factors.index)
        hmm_features = self._select_hmm_features(period_features)
        self.hmm_features_used = (["PC1"] if self.config.hmm_include_pc1 else []) + hmm_features

        Y_parts = []
        if self.config.hmm_include_pc1:
            Y_parts.append(self.pc_factors[["PC1"]])
        if hmm_features:
            Y_parts.append(period_features[hmm_features])
        if not Y_parts:
            raise ValueError("No HMM features selected.")

        Y = pd.concat(Y_parts, axis=1)
        Y = Y.apply(pd.to_numeric, errors="coerce")
        Y = Y.fillna(Y.median(numeric_only=True)).dropna(axis=1, how="all")
        Y = Y.loc[:, Y.nunique(dropna=True) > 1]
        if Y.shape[1] == 0:
            raise ValueError("HMM design matrix has no usable columns.")

        Y_scaled = StandardScaler().fit_transform(Y)
        model = self._fit_hmm_with_restarts(Y_scaled)
        raw_states_num = model.predict(Y_scaled)
        raw_probs = model.predict_proba(Y_scaled)

        state_pc1_mean = (
            pd.Series(raw_states_num, index=Y.index).to_frame("state")
            .join(self.pc_factors["PC1"])
            .groupby("state")["PC1"].mean()
        )
        label_map = {
            s: l for s, l in zip(
                state_pc1_mean.sort_values().index.tolist(),
                self.REGIME_ORDER,
            )
        }

        states_raw = pd.Series(
            [label_map[s] for s in raw_states_num], index=Y.index, name="Raw_regime"
        )
        probs = pd.DataFrame(
            raw_probs, index=Y.index,
            columns=[label_map[i] for i in range(raw_probs.shape[1])],
        ).reindex(columns=self.REGIME_ORDER)

        states_smooth, n_adjusted = (
            self._smooth_label_sequence(states_raw, self.config.min_regime_duration)
            if self.config.smooth_regime_states else (states_raw.copy(), 0)
        )
        states_smooth.name = "Smoothed_regime"

        self.hmm_model = model
        self.regime_probs = probs
        self.regime_states_raw = states_raw
        self.regime_states = states_smooth
        self.regime_adjustment_count = int(n_adjusted)
        self.regime_durations = self._duration_table(states_smooth)
        self.regime_transition_matrix = self._transition_matrix(states_smooth)

        print("\nCompact HMM fitted.")
        print(f"HMM features used              : {self.hmm_features_used}")
        print(f"Smoothed regime points adjusted: {self.regime_adjustment_count}")
        print("Regime durations:")
        display(self.regime_durations.round(3))
        print("Transition matrix:")
        display(self.regime_transition_matrix.round(3))

    # ----------------------------------------------------------
    # HMM-conditioned decomposition
    # ----------------------------------------------------------

    def _join_period_factors_to_rows(self, period_df: pd.DataFrame) -> pd.DataFrame:
        if self.df_model is None:
            raise RuntimeError("Model data not prepared.")
        periods = self.df_model[self.period_col].astype(str)
        return period_df.reindex(periods).set_index(self.df_model.index)

    def _fit_small_ols(self, design: pd.DataFrame, y: pd.Series, name: str):
        X = design.loc[:, ~design.columns.duplicated()].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        yy = pd.to_numeric(pd.Series(y, index=y.index), errors="coerce").replace([np.inf, -np.inf], np.nan)

        X = X[[c for c in X.columns if X[c].notna().any()]]
        if X.empty:
            raise ValueError(f"No usable numeric columns for {name}.")

        valid = yy.notna() & X.notna().all(axis=1)
        X_fit = X.loc[
            valid,
            [c for c in X.columns if X.loc[valid, c].nunique(dropna=True) > 1]
        ].copy()
        y_fit = yy.loc[valid].copy()

        if X_fit.empty:
            raise ValueError(f"No varying columns for {name}.")
        if len(y_fit) <= X_fit.shape[1] + 1:
            raise ValueError(f"Not enough observations for {name}: n={len(y_fit)}, k={X_fit.shape[1]}.")

        model = sm.OLS(
            y_fit, sm.add_constant(X_fit, has_constant="add"), missing="drop"
        ).fit(cov_type=self.config.ols_cov_type)
        fitted = pd.Series(np.nan, index=y.index, dtype=float, name=name)
        fitted.loc[X_fit.index] = model.fittedvalues
        return model, fitted

    def fit_hmm_conditioned_decomposition(self) -> None:
        if self.baseline_contributions is None or self.pc_factors is None or self.regime_probs is None:
            raise RuntimeError("Run baseline, PCA, and HMM first.")

        base_scores = self.baseline_contributions[["Global", "Emerging", "Bonds"]].copy()
        pc_rows = self._join_period_factors_to_rows(self.pc_factors)
        prob_rows = self._join_period_factors_to_rows(self.regime_probs)

        design = pd.DataFrame(index=self.y.index)
        for block in ["Global", "Emerging", "Bonds"]:
            design[f"BASE_{block}"] = base_scores[block]
        for pc in self.pc_factors.columns:
            design[pc] = pc_rows[pc]
        for reg in ["Transitional", "Stress"]:
            if reg in prob_rows.columns:
                design[f"REGIME_PROB_{reg}"] = prob_rows[reg]
        for block in ["Global", "Emerging", "Bonds"]:
            for reg in ["Transitional", "Stress"]:
                col = f"REGIME_PROB_{reg}"
                if col in design.columns:
                    design[f"{block}_x_{reg}"] = base_scores[block] * design[col]

        design = design.apply(pd.to_numeric, errors="coerce")
        design = design.fillna(design.median(numeric_only=True))
        design = design.loc[:, design.nunique(dropna=True) > 1]

        model, fitted = self._fit_small_ols(design, self.y, "HMM_conditioned_fitted")
        self.hmm_decomp_model = model
        self.hmm_design = design

        params = model.params
        components = pd.DataFrame(index=self.y.index)
        for block in ["Global", "Emerging", "Bonds"]:
            components[block] = params.get(f"BASE_{block}", 0.0) * design.get(f"BASE_{block}", 0.0)
        for reg in ["Transitional", "Stress"]:
            for block in ["Global", "Emerging", "Bonds"]:
                col = f"{block}_x_{reg}"
                if col in design.columns:
                    components[block] += params.get(col, 0.0) * design[col]

        components["Latent"] = sum(
            params.get(c, 0.0) * design[c]
            for c in design.columns if c.startswith("PC")
        )
        components["Regime"] = sum(
            params.get(f"REGIME_PROB_{reg}", 0.0) * design.get(f"REGIME_PROB_{reg}", 0.0)
            for reg in ["Transitional", "Stress"]
            if f"REGIME_PROB_{reg}" in design.columns
        )
        components["Intercept"] = float(params.get("const", 0.0))
        components["Fitted"] = fitted
        components["Actual"] = self.y
        components["Residual"] = self.y - fitted
        self.hmm_components = components

        component_cols = ["Global", "Emerging", "Bonds", "Latent", "Regime"]
        abs_means = components[component_cols].abs().mean()
        shares = (abs_means / abs_means.sum()).reset_index()
        shares.columns = ["Risk_component", "Share_of_abs_contribution"]
        self.hmm_component_shares = shares.sort_values(
            "Share_of_abs_contribution", ascending=False
        )

        states_rows = self._join_period_factors_to_rows(
            self.regime_states.to_frame("HMM_Regime")
        )["HMM_Regime"]
        profile = components[component_cols].copy()
        profile[component_cols] = profile[component_cols].apply(pd.to_numeric, errors="coerce")
        profile["HMM_Regime"] = states_rows.values

        def _melt(grp_df, val_name):
            return (
                grp_df.reset_index()
                .melt(
                    id_vars="HMM_Regime",
                    value_vars=component_cols,
                    var_name="Component",
                    value_name=val_name,
                )
                .rename(columns={"HMM_Regime": "Regime_State"})
            )

        prof_abs = _melt(
            profile.assign(**{c: profile[c].abs() for c in component_cols})
            .groupby("HMM_Regime", observed=False)[component_cols]
            .mean().reindex(self.REGIME_ORDER),
            "Mean_abs_contribution",
        )
        prof_mean = _melt(
            profile.groupby("HMM_Regime", observed=False)[component_cols]
            .mean().reindex(self.REGIME_ORDER),
            "Mean_signed_contribution",
        )
        self.regime_component_profile = prof_abs.merge(
            prof_mean, on=["Regime_State", "Component"], how="left"
        )

        self.regime_interaction_table = pd.DataFrame([
            {
                "Block": block,
                "Regime": reg,
                "Interaction_coefficient": params[f"{block}_x_{reg}"],
            }
            for block in ["Global", "Emerging", "Bonds"]
            for reg in ["Transitional", "Stress"]
            if f"{block}_x_{reg}" in params.index
        ])

        print("\nHMM-conditioned decomposition fitted.")
        display(pd.DataFrame([{
            "Model": "HMM-conditioned decomposition OLS",
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj,
            "RMSE": compute_rmse(self.y, fitted),
            "MAE": compute_mae(self.y, fitted),
            "N_obs": int(model.nobs),
            "N_decomposition_terms": int(design.shape[1]),
        }]).round(4))
        print("HMM-conditioned component shares:")
        display(self.hmm_component_shares.round(4))

    # ----------------------------------------------------------
    # Model comparison
    # ----------------------------------------------------------

    def build_model_comparison(self) -> None:
        rows = []
        if self.baseline_model is not None:
            fitted = self.baseline_contributions["Fitted"]
            rows.append({
                "Model": "1. Baseline OLS",
                "Purpose": "Observed Global / Emerging / Bond risk blocks",
                "R2": self.baseline_model.rsquared,
                "Adj_R2": self.baseline_model.rsquared_adj,
                "RMSE": compute_rmse(self.y, fitted),
                "MAE": compute_mae(self.y, fitted),
            })
        if self.pc_factors is not None:
            pc_rows = self._join_period_factors_to_rows(self.pc_factors)
            lm, lf = self._fit_small_ols(
                pd.concat([self.X, pc_rows], axis=1), self.y, "Latent_aug"
            )
            rows.append({
                "Model": "2. Baseline + latent PCs",
                "Purpose": "Adds common residual repricing factors",
                "R2": lm.rsquared,
                "Adj_R2": lm.rsquared_adj,
                "RMSE": compute_rmse(self.y, lf),
                "MAE": compute_mae(self.y, lf),
            })
        if self.hmm_decomp_model is not None:
            fitted = self.hmm_components["Fitted"]
            rows.append({
                "Model": "3. HMM-conditioned decomposition",
                "Purpose": "Adds regime probabilities and block × regime interactions",
                "R2": self.hmm_decomp_model.rsquared,
                "Adj_R2": self.hmm_decomp_model.rsquared_adj,
                "RMSE": compute_rmse(self.y, fitted),
                "MAE": compute_mae(self.y, fitted),
            })
        self.model_comparison = pd.DataFrame(rows)

    def print_key_tables(self) -> None:
        print("\n" + "=" * 78)
        print("KEY RISK-DECOMPOSITION TABLES")
        print("=" * 78)
        if self.model_comparison is not None:
            print("Model comparison:")
            display(self.model_comparison.round(4))
        if self.pca_variance_table is not None:
            print("Residual PCA variance:")
            display(self.pca_variance_table.round(4))
        if self.regime_durations is not None:
            print("HMM regime duration diagnostics:")
            display(self.regime_durations.round(3))
        if self.hmm_component_shares is not None:
            print("Final risk contribution shares:")
            display(self.hmm_component_shares.round(4))
        if self.regime_component_profile is not None:
            print("Average absolute contribution by regime:")
            display(
                self.regime_component_profile
                .pivot(index="Regime_State", columns="Component", values="Mean_abs_contribution")
                .round(4)
            )

    # ----------------------------------------------------------
    # Descriptive helpers
    # ----------------------------------------------------------

    def _build_share_table(self, column: str, label_name: str) -> pd.DataFrame:
        if self.df_model is None:
            raise RuntimeError("Run prepare_model_matrix() first.")
        if column not in self.df_model.columns:
            raise KeyError(f"Column {column!r} not found in final modelling sample.")
        s = (
            self.df_model[column].astype(str).str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN", "NAN": "UNKNOWN"})
        )
        tab = s.value_counts(dropna=False).rename_axis(label_name).reset_index(name="Observations")
        tab["Share"] = tab["Observations"] / tab["Observations"].sum()
        tab["Share_pct"] = 100.0 * tab["Share"]
        return tab

    def plot_country_share(
        self,
        output_dir: Optional[str] = None,
        filename: str = "01_country_share.html",
        show: bool = True,
    ):
        if self.df_model is None:
            return None
        country_col = next(
            (c for c in ["CNTRY_OF_RISK_FULL_NAME", "COUNTRY_FULL_NAME", "CNTRY_OF_RISK"]
             if c in self.df_model.columns), None
        )
        if country_col is None:
            print("No country column available.")
            return None

        tab = self._build_share_table(country_col, "Country")
        fig = px.pie(
            tab,
            names="Country",
            values="Observations",
            hole=0.45,
            title="Final sample — country share (%)",
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Obs: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
        fig.update_layout(template="plotly_white", legend_title="Country")
        return self._finalize_plot(
            fig,
            output_dir=output_dir,
            filename=filename,
            show=show,
        )

    def plot_sector_share(
        self,
        output_dir: Optional[str] = None,
        filename: str = "02_sector_share.html",
        show: bool = True,
    ):
        if self.df_model is None or "INDUSTRY_SECTOR" not in self.df_model.columns:
            print("INDUSTRY_SECTOR not available.")
            return None

        tab = self._build_share_table("INDUSTRY_SECTOR", "Sector")
        fig = px.pie(
            tab,
            names="Sector",
            values="Observations",
            hole=0.45,
            title="Final sample — sector share (%)",
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Obs: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
        fig.update_layout(template="plotly_white", legend_title="Sector")
        return self._finalize_plot(
            fig,
            output_dir=output_dir,
            filename=filename,
            show=show,
        )

    def build_descriptive_diagnostics(self) -> dict[str, pd.DataFrame]:
        if self.df_model is None:
            raise RuntimeError("Run prepare_model_matrix() first.")
        df = self.df_model.copy()
        out = {}

        country_col = next(
            (c for c in ["CNTRY_OF_RISK_FULL_NAME", "COUNTRY_FULL_NAME", "CNTRY_OF_RISK"]
             if c in df.columns), None
        )
        if country_col:
            tab = df[country_col].astype(str).value_counts().rename_axis("Country").reset_index(name="Observations")
            tab["Share"] = tab["Observations"] / len(df)
            out["country_distribution"] = tab

        for c in ["INDUSTRY_SECTOR", "SECTOR_LEVEL_2", "MERRILL_SECTOR_3"]:
            if c in df.columns:
                tab = df[c].astype(str).value_counts().rename_axis("Sector").reset_index(name="Observations")
                tab["Share"] = tab["Observations"] / len(df)
                out["sector_distribution"] = tab
                break

        if "TTM" in df.columns:
            bins = [0, 3, 5, 7, 10, np.inf]
            labels = ["0-3y", "3-5y", "5-7y", "7-10y", "10y+"]
            bucket = pd.cut(pd.to_numeric(df["TTM"], errors="coerce"), bins=bins, labels=labels, include_lowest=True)
            out["maturity_distribution"] = bucket.value_counts(dropna=False).rename_axis("Maturity_bucket").reset_index(name="Observations")

        target_cols = [c for c in ["RAW_MODEL_TARGET", "MODEL_TARGET_USED", "Z_SPREAD_MID", "Z_SPREAD_BPS"] if c in df.columns]
        if target_cols:
            out["target_summary"] = df[target_cols].apply(pd.to_numeric, errors="coerce").describe().T

        feats = [c for c in self.selected_features if c in df.columns]
        if feats:
            missing = df[feats].isna().mean().sort_values(ascending=False)
            out["feature_missingness_top20"] = (
                missing.head(20).rename("Missing_share").reset_index().rename(columns={"index": "Feature"})
            )
        return out

    def display_descriptive_diagnostics(self) -> None:
        for name, table in self.build_descriptive_diagnostics().items():
            print(f"\n{name.replace('_', ' ').title()}")
            display(table.head(20).round(4))

    # ----------------------------------------------------------
    # Monitoring summary table
    # ----------------------------------------------------------

    @staticmethod
    def _fmt(value, pct: bool = False, ndigits: int = 4) -> str:
        if value is None:
            return "n/a"
        try:
            if pd.isna(value):
                return "n/a"
        except Exception:
            pass
        if isinstance(value, (int, np.integer)) and not pct:
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{100.0 * float(value):.1f}%" if pct else f"{float(value):.{ndigits}f}"
        return str(value)

    def build_monitoring_summary_table(self) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        fmt = self._fmt

        def add(area, finding, value, takeaway=""):
            rows.append({
                "Area": area,
                "Finding": finding,
                "Value": str(value),
                "Takeaway": takeaway,
            })

        add("Run", "Universe", self.config.universe_mode,
            "LATAM-specific inputs excluded automatically in ALL_EM mode.")
        add("Run", "Spread target setup",
            f"final={self.config.target_col} | "
            f"HMM={self.config.regime_target_col or self.config.target_col} | "
            f"source={self.config.source_target_col} | "
            f"mode={self.config.target_mode_label}",
            "Final decomposition uses level; PCA/HMM uses regime target.")

        if self.df_model is not None:
            n_obs = len(self.df_model)
            n_periods = self.df_model[self.period_col].astype(str).nunique()
            n_isins = (
                self.df_model[self.config.id_col].nunique()
                if self.config.id_col in self.df_model.columns else None
            )
            add(
                "Sample", "Final panel",
                f"{n_obs:,} rows | {n_periods:,} periods" + (f" | {n_isins:,} ISINs" if n_isins else ""),
                "Final bond-period sample used in the decomposition."
            )

        if self.X is not None:
            add("Sample", "Model features", f"{self.X.shape[1]:,}",
                "Usable numeric inputs after cleaning and screening.")

        if self.y is not None:
            add("Target", "Final target intensity",
                f"mean abs={fmt(float(self.y.abs().mean()))} | std={fmt(float(self.y.std()))}",
                "Computed on the final decomposition target.")

        if self.model_comparison is not None and not self.model_comparison.empty:
            mc = self.model_comparison
            b = mc.loc[mc["Model"].str.contains("Baseline", case=False, na=False)]
            h = mc.loc[mc["Model"].str.contains("HMM", case=False, na=False)]
            if not b.empty:
                add("Fit", "Baseline OLS R²", fmt(float(b.iloc[0]["R2"])),
                    "Fit from observable Global/Emerging/Bond blocks.")
            if not h.empty:
                add("Fit", "Final HMM-conditioned R²", fmt(float(h.iloc[-1]["R2"])),
                    "Explanatory fit after latent and regime conditioning.")

        if self.baseline_block_shares is not None and not self.baseline_block_shares.empty:
            top = self.baseline_block_shares.sort_values(
                "Share_of_explained_abs_contribution", ascending=False
            ).iloc[0]
            add("Baseline", "Dominant observed block",
                f"{top['Risk_block']} ({fmt(float(top['Share_of_explained_abs_contribution']), pct=True)})",
                "Main directly observed risk channel.")

        if self.pca_variance_table is not None and not self.pca_variance_table.empty:
            pc1 = self.pca_variance_table.loc[
                self.pca_variance_table["Component"].eq("PC1"),
                "Explained_variance_ratio",
            ]
            if not pc1.empty:
                add("Latent", "PC1 residual variance", fmt(float(pc1.iloc[0]), pct=True),
                    "Size of the first latent common spread factor.")

        if self.regime_transition_matrix is not None and not self.regime_transition_matrix.empty:
            diag = [
                float(self.regime_transition_matrix.loc[r, r])
                for r in self.REGIME_ORDER
                if r in self.regime_transition_matrix.index
            ]
            if diag:
                add("Regimes", "Average self-transition", fmt(float(np.mean(diag)), pct=True),
                    "Higher means stable regimes.")

        if self.regime_states is not None:
            shares = self.regime_states.astype(str).value_counts(normalize=True)
            top_regime = shares.idxmax()
            add("Regimes", "Dominant regime",
                f"{top_regime} ({fmt(float(shares.max()), pct=True)})",
                "Most frequent smoothed HMM state.")

        if self.hmm_component_shares is not None and not self.hmm_component_shares.empty:
            h = self.hmm_component_shares.sort_values(
                "Share_of_abs_contribution", ascending=False
            )
            top = h.iloc[0]
            add("Final decomposition", "Dominant component",
                f"{top['Risk_component']} ({fmt(float(top['Share_of_abs_contribution']), pct=True)})",
                "Main channel in the HMM-conditioned decomposition.")
            add("Final decomposition", "Component shares",
                " | ".join(
                    f"{r['Risk_component']} {fmt(float(r['Share_of_abs_contribution']), pct=True)}"
                    for _, r in h.iterrows()
                ),
                "Full table exported as final_component_shares.csv.")

        if self.regime_component_profile is not None and not self.regime_component_profile.empty:
            prof = self.regime_component_profile
            totals = prof.groupby("Regime_State")["Mean_abs_contribution"].sum().sort_values(ascending=False)
            if not totals.empty:
                dr = totals.index[0]
                tc = (
                    prof.loc[prof["Regime_State"].eq(dr)]
                    .sort_values("Mean_abs_contribution", ascending=False)
                    .iloc[0]["Component"]
                )
                add("Regime profile", "Highest contribution intensity",
                    f"{dr} | top: {tc}",
                    "Regime where the decomposition is most active.")

        return pd.DataFrame(rows)

    def plot_monitoring_summary_table(
        self,
        output_dir: Optional[str] = None,
        filename: str = "final_monitoring_summary.html",
        show: bool = True,
        export_csv: bool = True,
    ):
        summary = self.build_monitoring_summary_table()
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        visual = summary.copy()
        for col in ["Area", "Finding", "Value", "Takeaway"]:
            visual[col] = visual[col].astype(str)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Area", "Finding", "Value", "Takeaway"],
                fill_color="#1f2937",
                font=dict(color="white", size=12),
                align="left", height=30,
            ),
            cells=dict(
                values=[visual["Area"], visual["Finding"], visual["Value"], visual["Takeaway"]],
                fill_color=["#f8fafc", "#ffffff", "#f8fafc", "#ffffff"],
                align="left", height=27, font=dict(size=11),
            ),
            columnwidth=[0.16, 0.24, 0.27, 0.33],
        )])
        fig.update_layout(
            title="Compact monitoring summary — regime-aware risk decomposition",
            width=1250,
            height=max(430, 80 + 30 * len(summary)),
            margin=dict(l=20, r=20, t=70, b=20),
        )

        html_path = out / filename
        fig.write_html(html_path)
        if export_csv:
            summary.to_csv(out / "final_monitoring_summary.csv", index=False)
        print(f"Monitoring summary saved: {html_path.resolve()}")
        if show:
            fig.show()
        return fig

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_tables(self, output_dir: Optional[str] = None) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tables = {
            "model_comparison": self.model_comparison,
            "baseline_block_shares": self.baseline_block_shares,
            "pca_variance": self.pca_variance_table,
            "regime_durations": self.regime_durations,
            "regime_transition_matrix": self.regime_transition_matrix,
            "final_component_shares": self.hmm_component_shares,
            "regime_component_profile": self.regime_component_profile,
            "regime_interactions": self.regime_interaction_table,
        }
        for name, table in tables.items():
            if table is not None:
                table.to_csv(
                    out / f"{name}.csv",
                    index=(name == "regime_transition_matrix"),
                )

        try:
            self.build_monitoring_summary_table().to_csv(
                out / "final_monitoring_summary.csv", index=False
            )
        except Exception as exc:
            print(f"Could not export final_monitoring_summary.csv: {exc}")

        if self.hmm_components is not None:
            comp = self.hmm_components.copy()
            comp[self.period_col] = self.df_model[self.period_col].astype(str).values
            if self.config.id_col in self.df_model.columns:
                comp[self.config.id_col] = self.df_model[self.config.id_col].astype(str).values
            comp.to_csv(out / "row_level_decomposition.csv", index=False)

        try:
            country_col = next(
                (c for c in ["CNTRY_OF_RISK_FULL_NAME", "COUNTRY_FULL_NAME", "CNTRY_OF_RISK"]
                 if self.df_model is not None and c in self.df_model.columns), None
            )
            if country_col:
                self._build_share_table(country_col, "Country").to_csv(
                    out / "country_share_final_sample.csv", index=False
                )
            if self.df_model is not None and "INDUSTRY_SECTOR" in self.df_model.columns:
                self._build_share_table("INDUSTRY_SECTOR", "Sector").to_csv(
                    out / "sector_share_final_sample.csv", index=False
                )
        except Exception as exc:
            print(f"Could not export country/sector share tables: {exc}")

        print(f"Exported all tables to: {out.resolve()}")
        return out

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------

    @staticmethod
    def _finalize_plot(
        fig: go.Figure,
        *,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        show: bool = True,
    ) -> go.Figure:
        """
        Save one Plotly figure as a complete standalone HTML file and optionally
        display it. Export and display are intentionally independent.
        """
        if output_dir is not None:
            if not filename:
                raise ValueError("filename is required when output_dir is provided.")

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / filename

            fig.write_html(
                str(path),
                include_plotlyjs=True,
                full_html=True,
                auto_open=False,
            )

            if not path.exists() or path.stat().st_size == 0:
                raise IOError(f"Plot export failed or produced an empty file: {path}")

            print(f"Plot saved: {path.resolve()}")

        if show:
            fig.show()

        return fig

    def _regime_color(self, regime: str) -> str:
        return {
            "Calm": "rgba(80, 170, 100, 0.18)",
            "Transitional": "rgba(240, 180, 60, 0.18)",
            "Stress": "rgba(220, 70, 70, 0.18)",
        }.get(regime, "rgba(100,100,100,0.12)")

    def _add_regime_bands(self, fig: go.Figure, states: pd.Series) -> go.Figure:
        vals, idx, start = states.astype(str).tolist(), list(states.index), 0
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] != vals[start]:
                fig.add_vrect(
                    x0=idx[start], x1=idx[i - 1],
                    fillcolor=self._regime_color(vals[start]),
                    line_width=0, layer="below",
                    annotation_text=vals[start] if i - start >= 8 else None,
                    annotation_position="top left",
                )
                start = i
        return fig

    def plot_baseline_block_shares(
        self,
        output_dir: Optional[str] = None,
        filename: str = "03_baseline_block_shares.html",
        show: bool = True,
    ):
        if self.baseline_block_shares is None:
            return None
        fig = px.bar(
            self.baseline_block_shares,
            x="Risk_block", y="Share_of_explained_abs_contribution",
            title="Baseline OLS decomposition — observed risk blocks",
            text="Share_of_explained_abs_contribution",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_yaxes(tickformat=".0%", title="Share of absolute explained contribution")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_pca_variance(
        self,
        output_dir: Optional[str] = None,
        filename: str = "04_pca_explained_variance.html",
        show: bool = True,
    ):
        if self.pca_variance_table is None:
            return None
        fig = px.bar(
            self.pca_variance_table,
            x="Component", y="Explained_variance_ratio",
            title="Residual PCA — explained variance by latent common factor",
            text="Explained_variance_ratio",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_regime_bands(
        self,
        output_dir: Optional[str] = None,
        filename: str = "05_hmm_regime_bands.html",
        show: bool = True,
    ):
        if self.pc_factors is None or self.regime_states is None:
            return None
        tgt = self.y_regime if self.y_regime is not None else self.y
        period_target = pd.DataFrame({
            "Period": self.df_model.loc[tgt.index, self.period_col].astype(str),
            "Target": tgt.values,
        }).groupby("Period")["Target"].mean().reindex(self.pc_factors.index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.pc_factors.index,
            y=self.pc_factors["PC1"],
            mode="lines",
            name="PC1 latent common factor",
        ))
        fig.add_trace(go.Scatter(
            x=period_target.index,
            y=period_target,
            mode="lines",
            name="Average target",
            yaxis="y2",
        ))
        fig = self._add_regime_bands(fig, self.regime_states)
        fig.update_layout(
            title="Compact HMM regimes over latent common spread factor",
            template="plotly_white",
            xaxis_title="Period",
            yaxis=dict(title="PC1"),
            yaxis2=dict(
                title="Average target",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_regime_probabilities(
        self,
        output_dir: Optional[str] = None,
        filename: str = "06_hmm_regime_probabilities.html",
        show: bool = True,
    ):
        if self.regime_probs is None:
            return None
        fig = px.line(
            self.regime_probs.reset_index().rename(columns={"index": "Period"}),
            x="Period",
            y=self.REGIME_ORDER,
            title="Filtered HMM regime probabilities",
            template="plotly_white",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Probability")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_final_component_shares(
        self,
        output_dir: Optional[str] = None,
        filename: str = "07_final_component_shares.html",
        show: bool = True,
    ):
        if self.hmm_component_shares is None:
            return None
        fig = px.bar(
            self.hmm_component_shares,
            x="Risk_component", y="Share_of_abs_contribution",
            title="Final HMM-conditioned risk decomposition",
            text="Share_of_abs_contribution",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_yaxes(tickformat=".0%", title="Share of absolute contribution")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_contributions_over_time(
        self,
        output_dir: Optional[str] = None,
        filename: str = "08_contributions_over_time.html",
        show: bool = True,
    ):
        if self.hmm_components is None:
            return None
        component_cols = ["Global", "Emerging", "Bonds", "Latent", "Regime"]
        tmp = self.hmm_components[component_cols].copy()
        tmp["Period"] = self.df_model[self.period_col].astype(str).values
        fig = px.line(
            tmp.groupby("Period")[component_cols].mean().reset_index(),
            x="Period",
            y=component_cols,
            title="Average signed risk contributions over time",
            template="plotly_white",
        )
        fig.update_yaxes(title="Contribution to modelled target")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_regime_component_profile(
        self,
        output_dir: Optional[str] = None,
        filename: str = "09_regime_component_profile.html",
        show: bool = True,
    ):
        if self.regime_component_profile is None:
            return None
        fig = px.bar(
            self.regime_component_profile,
            x="Regime_State",
            y="Mean_abs_contribution",
            color="Component",
            barmode="group",
            title="Risk contribution intensity by HMM regime",
            template="plotly_white",
        )
        fig.update_yaxes(title="Mean absolute contribution")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_actual_vs_fitted(
        self,
        output_dir: Optional[str] = None,
        filename: str = "10_actual_vs_fitted.html",
        show: bool = True,
    ):
        if self.hmm_components is None:
            return None
        fig = px.scatter(
            self.hmm_components,
            x="Actual",
            y="Fitted",
            title="Observed vs fitted — HMM-conditioned decomposition",
            opacity=0.45,
            trendline="ols",
            template="plotly_white",
        )
        fig.update_xaxes(title="Observed target")
        fig.update_yaxes(title="Fitted decomposition")
        return self._finalize_plot(
            fig, output_dir=output_dir, filename=filename, show=show
        )

    def plot_all(self, show: bool = True) -> None:
        """Display every analytical plot without exporting it."""
        self.plot_country_share(show=show)
        self.plot_sector_share(show=show)
        self.plot_baseline_block_shares(show=show)
        self.plot_pca_variance(show=show)
        self.plot_regime_bands(show=show)
        self.plot_regime_probabilities(show=show)
        self.plot_final_component_shares(show=show)
        self.plot_contributions_over_time(show=show)
        self.plot_regime_component_profile(show=show)
        self.plot_actual_vs_fitted(show=show)

    def export_all_plots(
        self,
        output_dir: str,
        *,
        show: bool = False,
    ) -> Path:
        """
        Export every analytical figure produced by the pipeline.

        The method verifies every written HTML file and creates a manifest.
        A missing optional source column is recorded as ``skipped``. Any actual
        export error is recorded and raised after the manifest is written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        plot_jobs = [
            ("country_share", self.plot_country_share, "01_country_share.html"),
            ("sector_share", self.plot_sector_share, "02_sector_share.html"),
            ("baseline_block_shares", self.plot_baseline_block_shares, "03_baseline_block_shares.html"),
            ("pca_explained_variance", self.plot_pca_variance, "04_pca_explained_variance.html"),
            ("hmm_regime_bands", self.plot_regime_bands, "05_hmm_regime_bands.html"),
            ("hmm_regime_probabilities", self.plot_regime_probabilities, "06_hmm_regime_probabilities.html"),
            ("final_component_shares", self.plot_final_component_shares, "07_final_component_shares.html"),
            ("contributions_over_time", self.plot_contributions_over_time, "08_contributions_over_time.html"),
            ("regime_component_profile", self.plot_regime_component_profile, "09_regime_component_profile.html"),
            ("actual_vs_fitted", self.plot_actual_vs_fitted, "10_actual_vs_fitted.html"),
        ]

        manifest_rows: list[dict[str, Any]] = []
        export_errors: list[str] = []

        for plot_name, plot_function, filename in plot_jobs:
            path = out / filename
            try:
                fig = plot_function(
                    output_dir=str(out),
                    filename=filename,
                    show=show,
                )
                if fig is None:
                    manifest_rows.append({
                        "plot": plot_name,
                        "filename": filename,
                        "status": "skipped",
                        "bytes": 0,
                        "message": "Required source data/column was unavailable.",
                    })
                    continue

                valid = path.exists() and path.stat().st_size > 0
                if not valid:
                    raise IOError(f"Expected plot file was not created: {path}")

                manifest_rows.append({
                    "plot": plot_name,
                    "filename": filename,
                    "status": "saved",
                    "bytes": int(path.stat().st_size),
                    "message": "",
                })
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                manifest_rows.append({
                    "plot": plot_name,
                    "filename": filename,
                    "status": "error",
                    "bytes": 0,
                    "message": message,
                })
                export_errors.append(f"{plot_name}: {message}")
                print(f"Could not export {plot_name}: {message}")

        # The monitoring summary is also a Plotly figure and belongs with plots.
        monitoring_filename = "11_final_monitoring_summary.html"
        monitoring_path = out / monitoring_filename
        try:
            fig = self.plot_monitoring_summary_table(
                output_dir=str(out),
                filename=monitoring_filename,
                show=show,
                export_csv=False,
            )
            valid = (
                fig is not None
                and monitoring_path.exists()
                and monitoring_path.stat().st_size > 0
            )
            if not valid:
                raise IOError(
                    f"Expected monitoring summary was not created: {monitoring_path}"
                )
            manifest_rows.append({
                "plot": "final_monitoring_summary",
                "filename": monitoring_filename,
                "status": "saved",
                "bytes": int(monitoring_path.stat().st_size),
                "message": "",
            })
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            manifest_rows.append({
                "plot": "final_monitoring_summary",
                "filename": monitoring_filename,
                "status": "error",
                "bytes": 0,
                "message": message,
            })
            export_errors.append(f"final_monitoring_summary: {message}")
            print(f"Could not export final_monitoring_summary: {message}")

        manifest = pd.DataFrame(manifest_rows)
        manifest_path = out / "plot_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        saved_count = int(manifest["status"].eq("saved").sum())
        skipped_count = int(manifest["status"].eq("skipped").sum())
        error_count = int(manifest["status"].eq("error").sum())

        print(
            f"Plot export complete: {saved_count} saved, "
            f"{skipped_count} skipped, {error_count} error(s)."
        )
        print(f"Plot folder  : {out.resolve()}")
        print(f"Plot manifest: {manifest_path.resolve()}")

        if export_errors:
            raise RuntimeError(
                "One or more plots failed to export:\n  - "
                + "\n  - ".join(export_errors)
            )

        return out


# ==========================================================
# Search-space definition
# ==========================================================

HMM_CANDIDATE_POOL_LATAM: list[str] = [
    "MA3_VIX_INDEX", "MA3_MOVE_INDEX", "MA3_JPM_EMBI", "MA3_MSCI_EM",
    "MA3_BFCIUS_INDEX", "MA3_BFCIEU_INDEX", "MA3_GPR_LEVEL",
    "LATAM_CDS_CREDIT_RISK", "LATAM_JPM_SPREAD_CREDIT_RISK",
    "MA3_GPR_LATAM_MOMENTUM", "MA3_CDX_IG_CDSI_GEN_5Y_CORP",
    "MA3_ITRX_EUR_CDSI_GEN_5Y_CORP", "MA3_STOXX600", "MA3_SP500",
    "MA3_EPUCEMEC_INDEX", "MA3_AONILS_INDEX",
]

HMM_CANDIDATE_POOL_ALL_EM: list[str] = [
    f for f in HMM_CANDIDATE_POOL_LATAM
    if not is_latam_specific_feature(f)
]

SEARCH_SPACE: dict[str, list[Any]] = {
    "hmm_covariance_type": ["diag", "full", "tied", "spherical"],
    "hmm_n_starts": [10, 20, 30, 50],
    "hmm_max_features": [4, 5, 6, 7, 8],
    "n_pca_components": [2, 3, 4],
    "min_isins_per_period": [3, 5, 7],
    "min_regime_duration": [2, 3, 4, 5],
    "ols_cov_type": ["HC3", "HC1", "HC0"],
    "winsorize_lower": [0.005, 0.01, 0.02],
    "winsorize_upper": [0.98, 0.99, 0.995],
    "standardize_features": [True, False],
    "hmm_include_pc1": [True, False],
    "n_features_to_drop": [0, 0, 0, 5, 10, 15, 20],
}


# ==========================================================
# Scoring function
# ==========================================================

def composite_score(
    r2_hmm: float,
    adj_r2_hmm: float,
    pct_significant: float,
    avg_self_transition: float,
    w_r2: float = 0.40,
    w_adj: float = 0.20,
    w_sig: float = 0.25,
    w_persist: float = 0.15,
) -> float:
    def _clip(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    return (
        w_r2 * _clip(r2_hmm)
        + w_adj * _clip(adj_r2_hmm)
        + w_sig * _clip(pct_significant)
        + w_persist * _clip(avg_self_transition)
    )


# ==========================================================
# Single-trial runner
# ==========================================================

def _run_single_trial(
    trial_id: int,
    params: dict[str, Any],
    base_features: list[str],
    hmm_pool: list[str],
    universe_mode: str,
    use_latam: bool,
    data_path: str,
    features_path: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trial_id": trial_id,
        "status": "error",
        "score": np.nan,
        "baseline_r2": np.nan,
        "baseline_adj_r2": np.nan,
        "hmm_r2": np.nan,
        "hmm_adj_r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "pct_significant_p05": np.nan,
        "pct_significant_p10": np.nan,
        "avg_self_transition": np.nan,
        "n_features_used": np.nan,
        "hmm_features_used": "",
        "n_regimes_identified": np.nan,
        "pc1_variance_share": np.nan,
        "error_message": "",
        **{k: str(v) for k, v in params.items()},
    }

    try:
        n_drop = int(params.get("n_features_to_drop", 0))
        features = list(base_features)
        if n_drop > 0 and n_drop < len(features):
            drop_idx = rng.choice(len(features), size=n_drop, replace=False)
            features = [f for i, f in enumerate(features) if i not in drop_idx]

        if not features:
            result["error_message"] = "Empty feature set after dropping."
            return result

        max_hmm = int(params.get("hmm_max_features", 6))
        available = [f for f in hmm_pool if f in features or f not in base_features]
        n_hmm = min(max_hmm, len(available))
        if n_hmm < 2:
            result["error_message"] = "Too few HMM features available."
            return result

        hmm_feat_idx = rng.choice(len(available), size=n_hmm, replace=False)
        hmm_features = [available[i] for i in sorted(hmm_feat_idx)]

        tmp_feat_path = OPTIMIZER_DATA_DIR / f"_tmp_features_{trial_id}.csv"
        feat_df = pd.read_csv(features_path)
        feat_df.columns = [str(c).upper().strip() for c in feat_df.columns]
        feat_df["FEATURE"] = feat_df["FEATURE"].astype(str).str.upper().str.strip()
        feat_df_filtered = feat_df.loc[feat_df["FEATURE"].isin(features)].copy()
        feat_df_filtered.to_csv(tmp_feat_path, index=False)

        config = RiskDecompositionConfig(
            data_path=data_path,
            features_path=str(tmp_feat_path),
            sheet_name=0,

            universe_mode=universe_mode,
            use_latam_specific_inputs=use_latam,

            target_col="MODEL_TARGET_LEVEL",
            regime_target_col="MODEL_TARGET",
            source_target_col="Z_SPREAD_MID",
            target_mode_label="weekly_pct_change",

            winsorize_target=True,
            winsorize_lower=float(params["winsorize_lower"]),
            winsorize_upper=float(params["winsorize_upper"]),

            ols_cov_type=str(params["ols_cov_type"]),
            standardize_features=bool(params["standardize_features"]),

            n_pca_components=int(params["n_pca_components"]),
            min_isins_per_period=int(params["min_isins_per_period"]),

            n_regimes=3,
            hmm_covariance_type=str(params["hmm_covariance_type"]),
            hmm_n_iter=300,
            hmm_tol=1e-4,
            hmm_n_starts=int(params["hmm_n_starts"]),
            hmm_random_state=trial_id % 100_000,
            hmm_include_pc1=bool(params["hmm_include_pc1"]),
            hmm_features=hmm_features,
            hmm_auto_select_features=False,
            hmm_max_features=max_hmm,

            smooth_regime_states=True,
            min_regime_duration=int(params["min_regime_duration"]),

            output_dir=str(OPTIMIZER_DATA_DIR / "_tmp"),
        )

        buf = io.StringIO()
        pipeline = RegimeRiskDecompositionPipeline(config)

        def _noop(*a, **k):
            return None

        pipeline.print_key_tables = _noop

        with contextlib.redirect_stdout(buf):
            pipeline.load_data()
            pipeline.prepare_model_matrix()
            pipeline.fit_baseline_regression()
            pipeline.run_residual_pca()
            pipeline.fit_compact_hmm()
            pipeline.fit_hmm_conditioned_decomposition()
            pipeline.build_model_comparison()

        mc = pipeline.model_comparison
        if mc is None or mc.empty:
            result["error_message"] = "model_comparison is empty."
            return result

        b_row = mc.loc[mc["Model"].str.contains("Baseline", case=False, na=False)]
        h_row = mc.loc[mc["Model"].str.contains("HMM", case=False, na=False)]

        baseline_r2 = float(b_row.iloc[0]["R2"]) if not b_row.empty else np.nan
        baseline_adj = float(b_row.iloc[0]["Adj_R2"]) if not b_row.empty else np.nan
        hmm_r2 = float(h_row.iloc[-1]["R2"]) if not h_row.empty else np.nan
        hmm_adj = float(h_row.iloc[-1]["Adj_R2"]) if not h_row.empty else np.nan
        hmm_rmse = float(h_row.iloc[-1]["RMSE"]) if not h_row.empty else np.nan
        hmm_mae = float(h_row.iloc[-1]["MAE"]) if not h_row.empty else np.nan

        pct_sig_05 = np.nan
        pct_sig_10 = np.nan
        if pipeline.hmm_decomp_model is not None:
            pvals = pipeline.hmm_decomp_model.pvalues.drop("const", errors="ignore")
            if len(pvals) > 0:
                pct_sig_05 = float((pvals < 0.05).mean())
                pct_sig_10 = float((pvals < 0.10).mean())

        avg_self = np.nan
        if pipeline.regime_transition_matrix is not None:
            tm = pipeline.regime_transition_matrix
            diag = [float(tm.loc[r, r]) for r in ["Calm", "Transitional", "Stress"]
                    if r in tm.index and r in tm.columns]
            if diag:
                avg_self = float(np.mean(diag))

        pc1_var = np.nan
        if pipeline.pca_variance_table is not None:
            pc1_row = pipeline.pca_variance_table.loc[
                pipeline.pca_variance_table["Component"].eq("PC1"),
                "Explained_variance_ratio"
            ]
            if not pc1_row.empty:
                pc1_var = float(pc1_row.iloc[0])

        n_reg = pipeline.regime_states.nunique() if pipeline.regime_states is not None else np.nan

        score = composite_score(
            r2_hmm=hmm_r2 if pd.notna(hmm_r2) else 0.0,
            adj_r2_hmm=hmm_adj if pd.notna(hmm_adj) else 0.0,
            pct_significant=pct_sig_05 if pd.notna(pct_sig_05) else 0.0,
            avg_self_transition=avg_self if pd.notna(avg_self) else 0.0,
        )

        result.update({
            "status": "ok",
            "score": round(score, 6),
            "baseline_r2": round(baseline_r2, 6),
            "baseline_adj_r2": round(baseline_adj, 6),
            "hmm_r2": round(hmm_r2, 6),
            "hmm_adj_r2": round(hmm_adj, 6),
            "rmse": round(hmm_rmse, 6),
            "mae": round(hmm_mae, 6),
            "pct_significant_p05": round(pct_sig_05, 4) if pd.notna(pct_sig_05) else np.nan,
            "pct_significant_p10": round(pct_sig_10, 4) if pd.notna(pct_sig_10) else np.nan,
            "avg_self_transition": round(avg_self, 4) if pd.notna(avg_self) else np.nan,
            "n_features_used": int(pipeline.X.shape[1]) if pipeline.X is not None else np.nan,
            "hmm_features_used": " | ".join(pipeline.hmm_features_used),
            "n_regimes_identified": int(n_reg) if pd.notna(n_reg) else np.nan,
            "pc1_variance_share": round(pc1_var, 4) if pd.notna(pc1_var) else np.nan,
        })

    except Exception:
        result["error_message"] = traceback.format_exc(limit=3)
    finally:
        try:
            if "tmp_feat_path" in dir() and tmp_feat_path.exists():
                tmp_feat_path.unlink()
        except Exception:
            pass

    return result


# ==========================================================
# Optimizer
# ==========================================================

class PipelineOptimizer:
    """
    Randomised search optimizer for RegimeRiskDecompositionPipeline.
    """

    LEADERBOARD_COLS = [
        "rank", "trial_id", "status", "score",
        "hmm_r2", "hmm_adj_r2", "baseline_r2",
        "pct_significant_p05", "pct_significant_p10",
        "avg_self_transition", "pc1_variance_share",
        "n_features_used", "n_regimes_identified",
        "rmse", "mae",
        "hmm_features_used",
        "hmm_covariance_type", "hmm_n_starts", "hmm_max_features",
        "n_pca_components", "min_isins_per_period", "min_regime_duration",
        "ols_cov_type", "winsorize_lower", "winsorize_upper",
        "standardize_features", "hmm_include_pc1", "n_features_to_drop",
    ]

    def __init__(
        self,
        n_iterations: int,
        universe_mode: str = "LATAM",
        random_seed: int = 42,
        score_weights: Optional[dict] = None,
    ) -> None:
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        self.n_iterations = n_iterations
        self.universe_mode = str(universe_mode).upper().strip()
        self.random_seed = random_seed
        self.score_weights = score_weights or {}

        self.use_latam = (self.universe_mode == "LATAM")
        self.hmm_pool = HMM_CANDIDATE_POOL_LATAM if self.use_latam else HMM_CANDIDATE_POOL_ALL_EM

        self.results: list[dict] = []
        self.leaderboard: Optional[pd.DataFrame] = None
        self.best_params: Optional[dict] = None
        self.best_score: float = -np.inf

        OPTIMIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OPTIMIZER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        OPTIMIZER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        (OPTIMIZER_DATA_DIR / "_tmp").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self) -> pd.DataFrame:
        print("\n" + "=" * 78)
        print("PIPELINE OPTIMIZER — RANDOMISED SEARCH")
        print("=" * 78)
        print(f"Iterations     : {self.n_iterations}")
        print(f"Universe       : {self.universe_mode}")
        print(f"Random seed    : {self.random_seed}")
        print(f"Output dir     : {OPTIMIZER_OUTPUT_DIR}")
        print(f"  ├─ data/     : {OPTIMIZER_DATA_DIR}")
        print(f"  └─ plots/    : {OPTIMIZER_PLOTS_DIR}")
        print("=" * 78)

        base_features = self._load_base_features()
        print(f"Base features available : {len(base_features):,}")

        rng = np.random.default_rng(self.random_seed)
        t0 = time.time()

        for i in range(1, self.n_iterations + 1):
            params = self._sample_params(rng)
            elapsed = time.time() - t0
            eta = (elapsed / i) * (self.n_iterations - i) if i > 1 else 0.0

            print(
                f"\n[{i:>4}/{self.n_iterations}] "
                f"cov={params['hmm_covariance_type']:<10} "
                f"pca={params['n_pca_components']} "
                f"dur={params['min_regime_duration']} "
                f"ols={params['ols_cov_type']:<4} "
                f"drop={params['n_features_to_drop']} "
                f"| ETA {eta/60:.1f} min",
                end="  ",
            )

            result = _run_single_trial(
                trial_id=i,
                params=params,
                base_features=base_features,
                hmm_pool=self.hmm_pool,
                universe_mode=self.universe_mode,
                use_latam=self.use_latam,
                data_path=str(MASTER_CLEAN_PATH),
                features_path=str(SELECTED_FEATURES_PATH),
                rng=rng,
            )

            self.results.append(result)

            if result["status"] == "ok":
                sc = float(result["score"])
                print(
                    f"score={sc:.4f}  R²={result['hmm_r2']:.4f}  "
                    f"sig%={result['pct_significant_p05']:.2f}  "
                    f"persist={result['avg_self_transition']:.2f}"
                )
                if sc > self.best_score:
                    self.best_score = sc
                    self.best_params = {**params, "hmm_features_used": result["hmm_features_used"]}
                    print(f"  ★ New best score: {sc:.4f}")
            else:
                msg = str(result.get("error_message", ""))[:120]
                print(f"ERROR — {msg}")

            if i % 10 == 0 or i == self.n_iterations:
                self._save_leaderboard()

        self._save_leaderboard()
        self._print_summary()
        return self.leaderboard

    # ----------------------------------------------------------
    # Best re-run
    # ----------------------------------------------------------

    def rerun_best(self, show_plots: bool = True) -> RegimeRiskDecompositionPipeline:
        if self.best_params is None:
            raise RuntimeError("No successful trial found. Cannot re-run best.")

        print("\n" + "=" * 78)
        print("RE-RUNNING BEST CONFIGURATION")
        print("=" * 78)
        print(f"Best score     : {self.best_score:.4f}")
        for k, v in self.best_params.items():
            print(f"  {k:<30} : {v}")
        print("=" * 78)

        base_features = self._load_base_features()
        n_drop = int(self.best_params.get("n_features_to_drop", 0))
        rng = np.random.default_rng(self.random_seed)

        best_rank = int(
            self.leaderboard.loc[self.leaderboard["score"].eq(self.best_score), "trial_id"].iloc[0]
        )
        for _ in range(1, best_rank):
            self._sample_params(rng)
        _ = self._sample_params(rng)

        features = list(base_features)
        if n_drop > 0 and n_drop < len(features):
            drop_idx = rng.choice(len(features), size=n_drop, replace=False)
            features = [f for j, f in enumerate(features) if j not in drop_idx]

        feat_df = pd.read_csv(SELECTED_FEATURES_PATH)
        feat_df.columns = [str(c).upper().strip() for c in feat_df.columns]
        feat_df["FEATURE"] = feat_df["FEATURE"].astype(str).str.upper().str.strip()
        feat_df_best = feat_df.loc[feat_df["FEATURE"].isin(features)].copy()
        best_feat_path = OPTIMIZER_DATA_DIR / "best_selected_features.csv"
        feat_df_best.to_csv(best_feat_path, index=False)

        hmm_features_str = str(self.best_params.get("hmm_features_used", ""))
        hmm_features = [
            f.strip() for f in hmm_features_str.split("|")
            if f.strip() and not f.strip().startswith("PC")
        ]

        best_data_dir = OPTIMIZER_DATA_DIR / "best_run"
        best_plots_dir = OPTIMIZER_PLOTS_DIR / "best_run"
        best_data_dir.mkdir(parents=True, exist_ok=True)
        best_plots_dir.mkdir(parents=True, exist_ok=True)

        config = RiskDecompositionConfig(
            data_path=str(MASTER_CLEAN_PATH),
            features_path=str(best_feat_path),
            sheet_name=0,

            universe_mode=self.universe_mode,
            use_latam_specific_inputs=self.use_latam,

            target_col="MODEL_TARGET_LEVEL",
            regime_target_col="MODEL_TARGET",
            source_target_col="Z_SPREAD_MID",
            target_mode_label="weekly_pct_change",

            winsorize_target=True,
            winsorize_lower=float(self.best_params["winsorize_lower"]),
            winsorize_upper=float(self.best_params["winsorize_upper"]),

            ols_cov_type=str(self.best_params["ols_cov_type"]),
            standardize_features=bool(self.best_params["standardize_features"]),

            n_pca_components=int(self.best_params["n_pca_components"]),
            min_isins_per_period=int(self.best_params["min_isins_per_period"]),

            n_regimes=3,
            hmm_covariance_type=str(self.best_params["hmm_covariance_type"]),
            hmm_n_iter=500,
            hmm_tol=1e-4,
            hmm_n_starts=int(self.best_params["hmm_n_starts"]),
            hmm_random_state=best_rank % 100_000,
            hmm_include_pc1=bool(self.best_params["hmm_include_pc1"]),
            hmm_features=hmm_features,
            hmm_auto_select_features=False,
            hmm_max_features=int(self.best_params["hmm_max_features"]),

            smooth_regime_states=True,
            min_regime_duration=int(self.best_params["min_regime_duration"]),

            output_dir=str(best_data_dir),
        )

        pipeline = RegimeRiskDecompositionPipeline(config)

        # Compute the complete best-run pipeline first. Plot display is handled
        # by export_all_plots so every displayed figure is also written to disk.
        pipeline.run_all(show_plots=False)

        pipeline.export_tables(str(best_data_dir))
        pipeline.export_all_plots(
            output_dir=str(best_plots_dir),
            show=show_plots,
        )

        print(f"\nBest-run data  saved to : {best_data_dir.resolve()}")
        print(f"Best-run plots saved to : {best_plots_dir.resolve()}")
        return pipeline

    # ----------------------------------------------------------
    # Visualisation
    # ----------------------------------------------------------

    def plot_leaderboard(self) -> None:
        if self.leaderboard is None or self.leaderboard.empty:
            print("No results to plot.")
            return

        ok = self.leaderboard.loc[self.leaderboard["status"].eq("ok")].copy()
        if ok.empty:
            print("No successful trials to plot.")
            return

        fig1 = px.histogram(
            ok, x="score", nbins=30,
            title="Composite score distribution across trials",
            labels={"score": "Composite score"},
            template="plotly_white",
        )
        fig1.add_vline(
            x=float(ok["score"].max()), line_dash="dash", line_color="red",
            annotation_text="Best", annotation_position="top right",
        )
        fig1.write_html(str(OPTIMIZER_PLOTS_DIR / "score_distribution.html"))
        fig1.show()

        fig2 = px.scatter(
            ok, x="hmm_r2", y="pct_significant_p05",
            color="hmm_covariance_type", size="score",
            hover_data=["trial_id", "score", "n_features_used", "hmm_features_used"],
            title="HMM R² vs fraction of significant coefficients (p < 0.05)",
            labels={"hmm_r2": "HMM R²", "pct_significant_p05": "% significant (p<0.05)"},
            template="plotly_white",
        )
        fig2.write_html(str(OPTIMIZER_PLOTS_DIR / "r2_vs_significance.html"))
        fig2.show()

        for dim in ["hmm_covariance_type", "ols_cov_type", "n_pca_components",
                    "min_regime_duration", "hmm_include_pc1", "standardize_features"]:
            if dim not in ok.columns:
                continue
            fig = px.box(
                ok, x=dim, y="score", points="all",
                title=f"Score distribution by {dim}",
                template="plotly_white",
            )
            fig.write_html(str(OPTIMIZER_PLOTS_DIR / f"score_by_{dim}.html"))
            fig.show()

        ok_sorted = ok.sort_values("trial_id")
        ok_sorted["running_best"] = ok_sorted["score"].cummax()
        fig4 = px.line(
            ok_sorted, x="trial_id", y=["score", "running_best"],
            title="Composite score per trial and running best",
            labels={"trial_id": "Trial", "value": "Score"},
            template="plotly_white",
        )
        fig4.write_html(str(OPTIMIZER_PLOTS_DIR / "score_convergence.html"))
        fig4.show()

        top10 = ok.head(10)[
            ["rank", "score", "hmm_r2", "pct_significant_p05",
             "avg_self_transition", "n_features_used",
             "hmm_covariance_type", "ols_cov_type",
             "n_pca_components", "min_regime_duration"]
        ].round(4)

        fig5 = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in top10.columns],
                fill_color="#1f2937",
                font=dict(color="white", size=11),
                align="center", height=28,
            ),
            cells=dict(
                values=[top10[c].tolist() for c in top10.columns],
                fill_color=["#f8fafc" if i % 2 == 0 else "#ffffff" for i in range(len(top10.columns))],
                align="center", height=25, font=dict(size=11),
            ),
        )])
        fig5.update_layout(
            title="Top-10 trials leaderboard",
            width=1300,
            height=max(380, 60 + 28 * len(top10)),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig5.write_html(str(OPTIMIZER_PLOTS_DIR / "top10_leaderboard_table.html"))
        fig5.show()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _load_base_features(self) -> list[str]:
        feat_df = pd.read_csv(SELECTED_FEATURES_PATH)
        feat_df.columns = [str(c).upper().strip() for c in feat_df.columns]
        feat_df["FEATURE"] = feat_df["FEATURE"].astype(str).str.upper().str.strip()
        features = [
            f for f in feat_df["FEATURE"].tolist()
            if is_allowed_model_input_feature(f)
            and (self.use_latam or not is_latam_specific_feature(f))
        ]
        return features

    def _sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            k: rng.choice(v).item() if isinstance(rng.choice(v), np.generic)
            else rng.choice(v)
            for k, v in SEARCH_SPACE.items()
        }

    def _save_leaderboard(self) -> None:
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        ok = df.loc[df["status"].eq("ok")].sort_values("score", ascending=False).copy()
        ok.insert(0, "rank", range(1, len(ok) + 1))
        err = df.loc[df["status"].ne("ok")].copy()
        err.insert(0, "rank", np.nan)
        self.leaderboard = pd.concat([ok, err], ignore_index=True)

        avail = [c for c in self.LEADERBOARD_COLS if c in self.leaderboard.columns]
        extra = [c for c in self.leaderboard.columns if c not in avail]
        self.leaderboard = self.leaderboard[avail + extra].copy()

        path = OPTIMIZER_DATA_DIR / "optimizer_leaderboard.csv"
        self.leaderboard.to_csv(path, index=False)

    def _print_summary(self) -> None:
        ok = [r for r in self.results if r["status"] == "ok"]
        print("\n" + "=" * 78)
        print("OPTIMIZER SUMMARY")
        print("=" * 78)
        print(f"Total trials      : {self.n_iterations}")
        print(f"Successful trials : {len(ok)}")
        print(f"Failed trials     : {self.n_iterations - len(ok)}")
        if ok:
            scores = [r["score"] for r in ok]
            print(f"Score  mean/std   : {np.mean(scores):.4f} / {np.std(scores):.4f}")
            print(f"Score  min/max    : {np.min(scores):.4f} / {np.max(scores):.4f}")
            print(f"Best score        : {self.best_score:.4f}")
            print("Best parameters:")
            if self.best_params:
                for k, v in self.best_params.items():
                    print(f"  {k:<30} : {v}")
        print(f"\nLeaderboard saved : {OPTIMIZER_DATA_DIR / 'optimizer_leaderboard.csv'}")
        print("=" * 78)


# ==========================================================
# MAIN — ONLY OPTIMIZER ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    if not MASTER_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"MASTER_CLEAN.xlsx not found:\n  {MASTER_CLEAN_PATH}\n"
            "Run data_cleaning.py first."
        )
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"selected_features.csv not found:\n  {SELECTED_FEATURES_PATH}\n"
            "Run data_cleaning.py first."
        )

    # ======================================================
    # USER SETTINGS — EDIT HERE
    # ======================================================

    N_ITERATIONS: int = 20
    UNIVERSE_MODE: str = "LATAM"
    RANDOM_SEED: int = 42

    RERUN_BEST: bool = True
    SHOW_PLOTS: bool = True

    SCORE_WEIGHTS: dict = {}
    # Example:
    # SCORE_WEIGHTS = {"w_r2": 0.50, "w_adj": 0.10, "w_sig": 0.30, "w_persist": 0.10}

    # ======================================================
    # Run
    # ======================================================

    OPTIMIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\npath_config loaded        : OK")
    print(f"Input  MASTER_CLEAN       : {MASTER_CLEAN_PATH}")
    print(f"Input  selected_features  : {SELECTED_FEATURES_PATH}")
    print(f"Output optimizer_results/ : {OPTIMIZER_OUTPUT_DIR}")
    print(f"  ├─ data/               : {OPTIMIZER_DATA_DIR}")
    print(f"  └─ plots/              : {OPTIMIZER_PLOTS_DIR}")

    optimizer = PipelineOptimizer(
        n_iterations=N_ITERATIONS,
        universe_mode=UNIVERSE_MODE,
        random_seed=RANDOM_SEED,
        score_weights=SCORE_WEIGHTS,
    )

    leaderboard = optimizer.run()
    optimizer.plot_leaderboard()

    if RERUN_BEST:
        best_pipeline = optimizer.rerun_best(show_plots=SHOW_PLOTS)

    print("\nOptimizer complete.")
    print(f"Leaderboard : {OPTIMIZER_DATA_DIR / 'optimizer_leaderboard.csv'}")
    if RERUN_BEST:
        print(f"Best run data  : {OPTIMIZER_DATA_DIR / 'best_run/'}")
        print(f"Best run plots : {OPTIMIZER_PLOTS_DIR / 'best_run/'}")

# %%