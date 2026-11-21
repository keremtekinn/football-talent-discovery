"""Model definitions and native XGBoost TreeSHAP helper."""

import lightgbm as lgb
import shap
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class WeightedPipeline(Pipeline):
    """Pipeline that routes sample_weight to the final estimator step."""

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if sample_weight is not None:
            last_step = self.steps[-1][0]
            fit_params[f"{last_step}__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def make_xgb(random_state=42):
    return WeightedPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            xgb.XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.5,
                min_child_weight=15,
                reg_alpha=1.0,
                reg_lambda=15.0,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])


def make_logreg(random_state=42):
    return WeightedPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, penalty="l2", max_iter=1000, random_state=random_state)),
    ])


def make_rf(random_state=42):
    return WeightedPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])


def make_lgbm(random_state=42):
    return WeightedPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            lgb.LGBMClassifier(
                n_estimators=250,
                max_depth=3,
                num_leaves=8,
                learning_rate=0.03,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                min_child_samples=30,
                reg_alpha=1.0,
                reg_lambda=15.0,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
    ])


def make_ensemble(random_state=42):
    return VotingClassifier(
        estimators=[
            ("xgb", make_xgb(random_state)),
            ("logreg", make_logreg(random_state)),
            ("rf", make_rf(random_state)),
        ],
        voting="soft",
        weights=[2, 2, 1],
    )


def get_models(random_state=42):
    return {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic Regression (L2)": make_logreg(random_state),
        "Random Forest": make_rf(random_state),
        "LightGBM": make_lgbm(random_state),
        "XGBoost": make_xgb(random_state),
        "Ensemble (XGB+LR+RF)": make_ensemble(random_state),
    }


def get_xgb_shap(xgb_pipeline, X_df, feature_names):
    # XGBoost 3.x base_score format breaks shap.TreeExplainer, so use native pred_contribs
    imputer = xgb_pipeline.named_steps["imputer"]
    clf = xgb_pipeline.named_steps["clf"]

    X_imp = imputer.transform(X_df[feature_names].to_numpy(dtype=float))
    dmat = xgb.DMatrix(X_imp, feature_names=feature_names)
    contribs = clf.get_booster().predict(dmat, pred_contribs=True)

    return shap.Explanation(
        values=contribs[:, :-1],
        base_values=contribs[:, -1],
        data=X_imp,
        feature_names=feature_names,
    )
