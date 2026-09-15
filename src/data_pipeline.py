import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import time
from functools import wraps
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
        self.cols_to_drop =  ['ID', 'NUM_REGISTRO', 'DATE_IMPUTATION', 'DATE_ENREG_IMP']
        self.cols_to_date_time = ['DATE_DOMICILIATION', 'DATE_ENREG_DO']
        self.cols_to_int = ['ANNEE', 'MOIS']
        self.cols_to_category = ['CODE_SH']
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None) -> 'PortNetDataCleaner':
        return self
    
    @log_step 
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        float_columns = df.select_dtypes(include =['float64']).columns
        int_columns = df.select_dtypes(include =['int64']).columns
            
        df[float_columns] = df[float_columns].astype('float32')
        df[int_columns] = df[int_columns].astype('int32')
            
        return df
    
    @log_step     
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns = self.cols_to_drop, errors= 'ignore')
    
    @log_step     
    def _set_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.cols_to_date_time:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors= 'coerce')
        
        for col in self.cols_to_int:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
                
        for col in self.cols_to_category:
            if col in df.columns:
                df[col] = df[col].astype('category')
                    
        return df

    @log_step 
    def _clean_quantite_domicile(self, df: pd.DataFrame) -> pd.DataFrame:
        df['QUANTITE_DOMICILE'] = df['QUANTITE_DOMICILE'].replace(999999999999, np.nan)
        return df
        
    def transform(self, X :pd.DataFrame) -> pd.DataFrame:
        return(X.copy()
               .pipe(self._drop_unnecessary_columns)
               .pipe(self._clean_quantite_domicile)
               .pipe(self._set_datatypes)
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
        return df.drop(columns=['DATE_ENREG_DO','DATE_DOMICILIATION'])
    
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
        mask_usd = df['DEVISE'] == 'USD'
        cols_to_convert = ['FRET', 'FOB', 'ASSURANCE', 'FRAIS_ACCESSOIRE']
        
        df.loc[mask_usd, cols_to_convert] = df.loc[mask_usd, cols_to_convert] * self.taux_conversion
        return df.drop(columns=['DEVISE'])
    
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
        self.columns_to_drop = ['DESCRIPTION_LIBRE']


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
    def _drop_useless_text(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop, errors='ignore')
    
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
                .pipe(self._drop_useless_text)
                .pipe(self._target_encode))
        
        
portnet_pipeline = Pipeline([
    ('cleaner', PortNetDataCleaner()),
    ('feature_engineer', PortNetFeatureEngineering(taux_conversion=0.85)),
    ('preprocessor', PortNetDataPreprocessing()),
    ('algorithme', 'passthrough')
])
