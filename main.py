import pandas as pd
import numpy as np
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def engineer_features(df):
    """Feature engineering, pulled into one function so it isn't duplicated
    between the 'train' branch and the 'load' branch below."""
    chain_counts = df['Restaurant Name'].value_counts()

    df['Is_Chain'] = df['Restaurant Name'].map(lambda x: 1 if chain_counts[x] > 1 else 0)
    df['Multi_Cuisines'] = (df['Cuisines'].astype(str).str.split(',').str.len() > 1).astype(int)
    city_mean_cost = df.groupby('City')['Average Cost for two'].transform('mean')
    df['Relative_Cost_City'] = df['Average Cost for two'] / (city_mean_cost + 0.001)
    return df


def build_pipeline():
    te_cols = ['City', 'Locality']
    ohe_cols = ['Has Table booking', 'Has Online delivery']

    num_cols = ['Price range', 'Votes', 'Is_Chain', 'Multi_Cuisines', 'Relative_Cost_City']

    preprocessor = ColumnTransformer(transformers=[
        ('te', TargetEncoder(), te_cols),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), ohe_cols),
        ('num', 'passthrough', num_cols)
    ], verbose_feature_names_out=False)

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    return final_pipeline

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

features = ['City', 'Locality', 'Average Cost for two', 'Has Table booking',
            'Has Online delivery', 'Price range', 'Votes',
            'Is_Chain', 'Multi_Cuisines', 'Relative_Cost_City']

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('Dataset .csv')
    df = engineer_features(df)

    X = df[features]
    y = df['Aggregate rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    final_pipeline = build_pipeline()
    X_train_transformed = final_pipeline.fit_transform(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_transformed, y_train)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(PIPELINE_FILE, 'wb') as f:
        pickle.dump(final_pipeline, f)

    X_test_transformed = final_pipeline.transform(X_test)
    predictions = xgb_model.predict(X_test_transformed)

    print("Trained a new model and saved it to", MODEL_FILE, "and", PIPELINE_FILE)
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
    print(f"R2:   {r2_score(y_test, predictions):.4f}")

else:
    df = pd.read_csv('Dataset .csv')
    df = engineer_features(df)

    X = df[features]
    y = df['Aggregate rating']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(MODEL_FILE, 'rb') as f:
        xgb_model = pickle.load(f)
    with open(PIPELINE_FILE, 'rb') as f:
        final_pipeline = pickle.load(f)

    X_test_transformed = final_pipeline.transform(X_test)
    predictions = xgb_model.predict(X_test_transformed)

    print("Loaded existing model from", MODEL_FILE)
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
    print(f"R2:   {r2_score(y_test, predictions):.4f}")

    output_data = pd.DataFrame({
        "Restaurant Name": df.loc[X_test.index, 'Restaurant Name'],
        "Actual_Rating": y_test,
        "Predicted_Rating": predictions
    })
    output_data.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")