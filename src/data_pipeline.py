import pandas as pd
import numpy as np
from typing import Optional
import time
from functools import wraps
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Le Décorateur (Monitoring du pipeline)
def log_step(func):
    """Décorateur pour logger le temps d'exécution et la dimension du DataFrame à chaque étape."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        # args[0] est 'self', args[1] est le DataFrame
        print(f"[{func.__name__}] exécuté en {time.time() - start:.3f}s | Lignes restantes: {result.shape[0]}")
        return result
    return wrapper

# 2. La Classe Orientée Objet (Scikit-Learn compatible)
class PortNetDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop = ['ID', 'NUM_REGISTRO', 'DATE_IMPUTATION', 'DATE_ENREG_IMP']
        self.cols_to_date_time = ['DATE_DOMICILIATION', 'DATE_ENREG_DO']
        self.cols_to_int = ['ANNEE', 'MOIS']
        self.cols_to_category = ['CODE_SH']

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PortNetDataCleaner':
        return self

    @log_step
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        float_columns = df.select_dtypes(include=['float64']).columns
        int_columns = df.select_dtypes(include=['int64']).columns
        df[float_columns] = df[float_columns].astype('float32')
        df[int_columns] = df[int_columns].astype('int32')
        return df

    @log_step
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.cols_to_drop, errors='ignore')

    @log_step
    def _filter_valid_currencies(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'DEVISE' in df.columns:
            df = df[df['DEVISE'].isin(['EUR', 'USD'])].copy()
        return df

    @log_step
    def _set_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.cols_to_date_time:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors='coerce')

        for col in self.cols_to_int:
            if col in df.columns:
                df[col] = (pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32'))

        for q_col in ['QUANTITE_DOMICILE', 'QTE_IMPUTE']:
            if q_col in df.columns:
                df[q_col] = pd.to_numeric(df[q_col], errors='coerce')

        for col in self.cols_to_category:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    @log_step
    def _clean_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        pattern = r'^9{7,}(\.0+)?$'
        seuil_absurde = 20_000_000

        if 'QUANTITE_DOMICILE' in df.columns:
            col_str = df['QUANTITE_DOMICILE'].astype(str)
            is_sentinel = col_str.str.match(pattern, na=False)
            df.loc[is_sentinel, 'QUANTITE_DOMICILE'] = np.nan
            df.loc[df['QUANTITE_DOMICILE'] > seuil_absurde, 'QUANTITE_DOMICILE'] = (np.nan)

        if 'QTE_IMPUTE' in df.columns:
            col_str_imp = df['QTE_IMPUTE'].astype(str)
            is_sentinel_imp = col_str_imp.str.match(pattern, na=False)
            df.loc[is_sentinel_imp, 'QTE_IMPUTE'] = np.nan
            df.loc[df['QTE_IMPUTE'] > seuil_absurde, 'QTE_IMPUTE'] = np.nan

        return df

    @log_step
    def _filter_quantity(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'QUANTITE_DOMICILE' in df.columns:
            df = df[
                (df['QUANTITE_DOMICILE'] > 0) &
                (df['QUANTITE_DOMICILE'] <= 20_000_000)]

        if 'QTE_IMPUTE' in df.columns:
            df = df[
                (df['QTE_IMPUTE'] > 0) &
                (df['QTE_IMPUTE'] <= 20_000_000)]

        return df

    @log_step
    def _clean_null_date(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_mask = (
            (df['MOIS'] >= 1) &
            (df['MOIS'] <= 12) &
            (df['ANNEE'] >= 2010) &
            (df['ANNEE'] <= 2024)
            )
        return df[valid_mask].copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.copy()
            .pipe(self._drop_unnecessary_columns)
            .pipe(self._filter_valid_currencies)
            .pipe(self._set_datatypes)
            .pipe(self._clean_quantities)
            .pipe(self._filter_quantity)
            .pipe(self._clean_null_date)
            .pipe(self._optimize_memory)
            )
            
class PortNetFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, taux_conversion = 0.85):
        self.date_cols = ['DATE_DOMICILIATION', 'DATE_ENREG_DO']
        self.taux_conversion = taux_conversion

        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None) -> 'PortNetFeatureEngineering':
        return self
    
    @log_step
    def _delai_extracting(self, df: pd.DataFrame) -> pd.DataFrame:
      df['DELAI_ENREGISTREMENT'] = (df['DATE_ENREG_DO'] - df['DATE_DOMICILIATION']).dt.days

      valid_mask = (
          (df['DELAI_ENREGISTREMENT'] >= 0) &
          (df['DELAI_ENREGISTREMENT'] <= 365) &
          (df['DELAI_ENREGISTREMENT'].notna())
          )
      df = df[valid_mask].copy()

      return df.drop(columns=['DATE_ENREG_DO', 'DATE_DOMICILIATION'])
    
    @log_step
    def _strategic_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        df['TRIMESTRE'] = np.ceil(df['MOIS'] / 3).astype('int32')
        return df
    
    @log_step
    def _temporal_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # 2pi * MOIS\12
        df['MOIS_SIN'] = np.sin(2 * np.pi * df['MOIS'] / 12).astype('float32')
        df['MOIS_COS'] = np.cos(2 * np.pi * df['MOIS'] / 12).astype('float32')
        
        return df
    
    @log_step
    def _flags_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        df['A_UNE_ASSURANCE'] = (df['ASSURANCE'] > 0).astype('int8')
        df['A_FRAIS_ACCESSOIRE'] = (df['FRAIS_ACCESSOIRE'] > 0).astype('int8')
        
        return df
    
    @log_step
    def _unify_currency_to_eur(self, df: pd.DataFrame) -> pd.DataFrame:
      cols_to_convert = ['FRET', 'FOB', 'ASSURANCE', 'FRAIS_ACCESSOIRE']

      if 'DEVISE' in df.columns:
        mask_usd = df['DEVISE'] == 'USD'
        df.loc[mask_usd, cols_to_convert] = (df.loc[mask_usd, cols_to_convert] * self.taux_conversion)
        df = df.drop(columns=['DEVISE'])

      return df
    
    def transform(self, X :pd.DataFrame) -> pd.DataFrame:
            return(X.copy()
                   .pipe(self._delai_extracting)
                   .pipe(self._strategic_grouping)
                   .pipe(self._temporal_engineering)
                   .pipe(self._flags_creation)
                   .pipe(self._unify_currency_to_eur)
                   )

class PortNetDataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median') 
        self.scaler = StandardScaler()
        self.encoder = TargetEncoder(smooth="auto")
        
        self.columns_to_transform = ['FRET', 'FOB', 'QUANTITE_DOMICILE']
        self.columns_to_encode = ['CODE_SH']


    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PortNetDataPreprocessing':
        X_temp = np.log1p(X[self.columns_to_transform].copy())
        X_temp_imputed = self.imputer.fit_transform(X_temp)
        self.scaler.fit(X_temp_imputed)
        self.encoder.fit(X[self.columns_to_encode], y)
        return self
        
    @log_step
    def _logarithmic_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.columns_to_transform] = np.log1p(df[self.columns_to_transform])
        return df
    
    @log_step
    def _target_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.columns_to_encode] = self.encoder.transform(df[self.columns_to_encode])
        return df
    
    @log_step
    def _impute_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        donnees_imputees = self.imputer.transform(df[self.columns_to_transform])
        df[self.columns_to_transform] = self.scaler.transform(donnees_imputees)
        return df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X.copy()
                .pipe(self._logarithmic_transform)
                .pipe(self._impute_and_scale)
                .pipe(self._target_encode))
        

cleaning_fe_pipeline = Pipeline([
    ('cleaner', PortNetDataCleaner()),
    ('feature_engineer', PortNetFeatureEngineering(taux_conversion=0.85))
])

modeling_pipeline = Pipeline([
    ('preprocessor', PortNetDataPreprocessing()),
    ('algorithme', 'passthrough')
])