import math
import numpy as np
import pandas as pd

from scipy.stats import beta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K

pd.set_option('display.max_columns', None)

FEATURES   = ["CaseLocation", "CaseAttorneyJuris", "CaseDispositionJudgeJurisNo", "MotionFilingParty"]
OTHER_COLS = ["CaseMajorCode"]
TARGET     = "MotionResultCode"


# ONE-HOT ENCODING

"""def onehot(df, cols):
    df[cols] = df[cols].fillna('nan')
    encoded_df = pd.DataFrame()

    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each)
        encoded_df = pd.concat([encoded_df, dummies], axis=1)

    final_df = pd.concat([encoded_df, df[OTHER_COLS], df[TARGET]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(final_df.drop(columns=[TARGET]), final_df[TARGET], test_size=0.2, stratify=final_df[TARGET], random_state=42)
    return X_train, y_train, X_test, y_test"""


# BIN COUNTING

# Helper function for beta-binomial modeling for motion results
def motion_result_beta_binomial(df, feature, target="MotionResultCode"):
    counts = df.groupby([feature, target]).size().unstack(fill_value=0)
    counts["beta_mean"] = counts.apply(lambda row: beta.stats(1 + row["GR"], 1 + row["DN"])[0], axis=1)
    return counts["beta_mean"].to_dict()

# Helper function for beta-binomial modeling for case major codes
def case_major_code_beta_binomial(df, feature, case_major_code="CaseMajorCode"):
    counts = df.groupby([feature, case_major_code]).size().unstack(fill_value=0)
    counts["beta_mean"] = counts.apply(lambda row: beta.stats(1 + row.get("T", 0), 1 + row.get("V", 0))[0], axis=1)
    return counts["beta_mean"].to_dict()

def bincount(df, features, target):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target]), df[target], test_size=0.2, stratify=df[target], random_state=42)
    train_df = X_train.join(y_train)
    test_df = X_test.join(y_test)
    
    for feature in features:
        motion_beta_feature = motion_result_beta_binomial(train_df, feature, target)
        train_df[f"{feature}_motion_beta"] = train_df[feature].map(motion_beta_feature).fillna(0)
        test_df[f"{feature}_motion_beta"] = test_df[feature].map(motion_beta_feature).fillna(0)
        
        case_code_beta_feature = case_major_code_beta_binomial(train_df, feature)
        train_df[f"{feature}_case_code_beta"] = train_df[feature].map(case_code_beta_feature).fillna(0)
        test_df[f"{feature}_case_code_beta"] = test_df[feature].map(case_code_beta_feature).fillna(0)
    
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    
    return X_train, y_train, X_test, y_test


# NEURAL EMBEDDING

def neural_embedding(train, test):
    # Map labels for train and test datasets
    train['MotionResultCode'] = train['MotionResultCode'].map({'GR': 1, 'DN': 0})
    test['MotionResultCode'] = test['MotionResultCode'].map({'GR': 1, 'DN': 0})
    
    # Label encode features
    lbl_encoders = {}
    for feature in FEATURES:
        lbl_enc = LabelEncoder()
        train[feature] = lbl_enc.fit_transform(train[feature].fillna("-1").astype(str).values)
        test[feature] = lbl_enc.transform(test[feature].fillna("-1").astype(str).values)
        lbl_encoders[feature] = lbl_enc  # Store encoder

    # Prepare the full dataset (combine training and validation sets)
    X_full = train[FEATURES].copy()
    y_full = train[TARGET].copy()
    
    # Prepare the test set
    X_test = test[FEATURES].copy()
    y_test = test[TARGET].copy()

    # Model architecture for feature extraction
    input_models = []
    output_embeddings = []
    for categorical_var in FEATURES:
        cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"
        no_of_unique_cat = X_full[categorical_var].nunique()
        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50))

        input_model = Input(shape=(1,), name=f"{cat_emb_name}_input")
        output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
        output_model = Reshape(target_shape=(embedding_size,))(output_model)
        
        input_models.append(input_model)
        output_embeddings.append(output_model)

    # Concatenate all embeddings to create a unified feature set
    if len(output_embeddings) > 1:
        final_output = Concatenate()(output_embeddings)
    else:
        final_output = output_embeddings[0]  # If only one feature

    # Define the model
    model = Model(inputs=input_models, outputs=final_output)

    # Preparing input lists for model prediction
    X_full_list = [X_full[feature].values for feature in FEATURES]
    X_test_list = [X_test[feature].values for feature in FEATURES]

    # Use the model to transform data
    X_full_transformed = model.predict(X_full_list)
    X_test_transformed = model.predict(X_test_list)

    feature_names = [f'feature_{i}' for i in range(X_full_transformed.shape[1])]

    X_full_df = pd.DataFrame(X_full_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Return transformed features and labels for use in other classifiers
    return X_full_df, y_full, X_test_df, y_test


# PREPROCESSING FOR NEURAL EMBEDDING

def preproc(X_train, X_val, X_test=None):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    for c in FEATURES:
        raw_vals = np.unique(X_train[c])
        val_map = {v: i for i, v in enumerate(raw_vals)}
        input_list_train.append(np.vectorize(val_map.get)(X_train[c]))
        input_list_val.append(np.vectorize(val_map.get)(X_val[c], 0))
        if X_test is not None:
            input_list_test.append(np.vectorize(val_map.get)(X_test[c], 0))

    if X_test is not None:
        return input_list_train, input_list_val, input_list_test
    else:
        return input_list_train, input_list_val


# FEATURE ENGINEERING

def feature_engineering(df, method):
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES + OTHER_COLS], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42)
    train = X_train.join(y_train)
    test  = X_test.join(y_test)

    for feature in FEATURES:
        only_test = set(X_test[feature]) - set(X_train[feature])
        test = test[~test[feature].isin(only_test)]

    if method == 'onehot':   
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES)], remainder='passthrough')
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
    elif method == 'bincount': 
        X_train, y_train, X_test, y_test = bincount(df, FEATURES, TARGET)
    elif method == 'neural':   
        X_train, y_train, X_test, y_test = neural_embedding(train, test)

    else: pass

    return X_train, y_train, X_test, y_test