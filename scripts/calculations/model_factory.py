"""Construct the regression models used by every controlled comparison.

Keeping model construction here prevents the production, DFT-ablation, and
DFT-control scripts from drifting to different hyperparameters. Optional
third-party estimators are included only when their packages are installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class ModelFactory:
    """Build an identical estimator registry for all comparison scripts."""

    random_state: int = 42
    n_estimators: int = 200
    gpr_max_samples: int = 500

    def build(
        self,
        n_train: int,
        logger: logging.Logger | None = None,
    ) -> dict[str, object]:
        models: dict[str, object] = {
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=10_000),
            "ElasticNet": ElasticNet(max_iter=10_000),
            "BayesianRidge": BayesianRidge(),
            "SVR": Pipeline(
                [("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))]
            ),
            "DecisionTree": DecisionTreeRegressor(random_state=self.random_state),
            "RandomForest": RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            "MLP": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(256, 128, 64),
                            max_iter=500,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
        }
        self._add_optional_models(models, logger)

        if n_train <= self.gpr_max_samples:
            models["GPR"] = GaussianProcessRegressor(
                kernel=DotProduct() + WhiteKernel(),
                random_state=self.random_state,
                normalize_y=True,
            )
        elif logger:
            logger.info(
                "  GPR skipped (n_train=%s > %s)",
                n_train,
                self.gpr_max_samples,
            )
        return models

    def _add_optional_models(
        self,
        models: dict[str, object],
        logger: logging.Logger | None,
    ) -> None:
        """Add optional estimators without changing the core registry."""
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            )
        except ImportError:
            if logger:
                logger.warning("xgboost is not installed; XGBoost is skipped")

        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            if logger:
                logger.warning("lightgbm is not installed; LightGBM is skipped")

        try:
            from catboost import CatBoostRegressor

            models["CatBoost"] = CatBoostRegressor(
                iterations=self.n_estimators,
                random_state=self.random_state,
                verbose=False,
            )
        except ImportError:
            if logger:
                logger.warning("catboost is not installed; CatBoost is skipped")
