import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, target_col: str = "Class"):
        self.target_col = target_col
        self.pipeline = None
        self.feature_names_in_ = None

    def fit(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        self.feature_names_in_ = X.columns.tolist()
        numeric_features = [f for f in ['Time', 'Amount'] if f in self.feature_names_in_]
        pass_through_features = [f for f in self.feature_names_in_ if f not in numeric_features]
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if pass_through_features:
            transformers.append(('passthrough', 'passthrough', pass_through_features))
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop'))
        ])
        self.pipeline.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet.")
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        X = X[self.feature_names_in_]
        X_transformed = self.pipeline.transform(X)
        output_features = []
        for name, transformer, features in self.pipeline.named_steps['preprocessor'].transformers_:
            if name != 'remainder':
                output_features.extend(features)
        return pd.DataFrame(X_transformed, columns=output_features, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
