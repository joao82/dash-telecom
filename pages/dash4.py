import pathlib
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import scipy.optimize as opt

import dash
from dash import Input, Output, callback, dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
import itertools


dash.register_page(__name__, name='Model')


# Data reading convert a CSV file to a pandas data frame
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
churn_df = pd.read_csv(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))
df = churn_df.copy()


# DATA CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].replace(np.nan, 0)
df['Churn'].replace(['Yes', 'No'], [1, 0], inplace=True)
df = df.drop(['customerID'], axis=1 )
df_uniques = pd.DataFrame(
  [[i, len(df[i].unique())] for i in df.columns], 
  columns=['Variable', 'Unique Values']).set_index('Variable')


# DATA PROCESSING
binary_variables = list(df_uniques[df_uniques['Unique Values'] == 2].index)
categorical_variables = list(df_uniques[
  (6 >= df_uniques['Unique Values']) & 
  (df_uniques['Unique Values'] > 2)
  ].index)
numeric_variables = list(set(df.columns) - set(categorical_variables) - set(binary_variables))

# one-hot-encoded: convert binary variables into numeric variables
dummies = pd.get_dummies(df[['gender', 'Partner','Dependents', 'PhoneService','PaperlessBilling']])
merged = pd.concat([df, dummies], axis='columns')
merged = merged.drop(['gender', 'Partner','Dependents', 'PhoneService','PaperlessBilling'], axis=1)

dummies = pd.get_dummies(merged[['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']])
merged = pd.concat([merged, dummies], axis='columns')
merged = merged.drop(
  ['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup','DeviceProtection', 
  'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'], axis=1)

# NORMALIZATION of the features with skewness higher than the skew limit (0.075) - LOG method
# create a list of float columns to check for skewing
float_mask = df.dtypes == np.float64
float_cols = df.columns[float_mask]

skew_limit = 0.075 # define a limit above which we will log transform
skew_vals = merged[float_cols].skew()

skew_cols = (skew_vals.sort_values(ascending=False)
              .to_frame().rename(columns={0:'Skew'}).query('abs(Skew) > {}'.format(skew_limit)))

skew_cols.index.to_list()
skew_vals.sort_values(ascending=False).to_frame().rename(columns={0:'Skew'}).query(f'abs(Skew) > {skew_limit}')

for col in skew_cols.index.values:
    if col == "Churn":
        continue
    merged[col] = merged[col].apply(np.log1p)




header = html.H3(
    "Machine Learning Model - Logistic Regression Algorithm", className="text-primary p-2 mb-4 text-center"
)

Controls_title = html.H5(
    "Refine the LR Model", className="text-primary p-1 mb-2 text-center"
)

splitter = html.Div(
    [
        dbc.Label(["Select Ratio test/train data "], className="fw-bold w-100"),
        dbc.RadioItems(
            id="splitter",
            options=[
              {'label': '80/20', 'value': 0.8},
              {'label': '50/50', 'value': 0.5},
              {'label': '20/80', 'value': 0.2},
              ],
            value=0.2,
            inline=True,
        ),
    ],
    className="mb-4",
)

checklist_numerical_optimizers = html.Div(
    [
        dbc.Label(["Select a Numerical Optimizer"], className="fw-bold w-100"),
        dbc.RadioItems(
            id="optimizer",
            options=[
              {'label': 'Newton`s method', 'value': 'newton-cg'},
              {'label': 'lbfgs', 'value': 'lbfgs'},
              {'label': 'lib linear', 'value': 'liblinear'},
              {'label': 'Sag', 'value': 'sag'},
              {'label': 'Saga', 'value': 'saga'},
              ],
            value='liblinear',
            inline=False,
        ),
    ],
    className="mb-4",
)

slider = html.Div(
    [
        dbc.Label(["Select C"], className="fw-bold w-100"),
        dcc.Slider(
            0,
            20,
            0.1,
            id="cfactor",
            marks={
              0: '0',
              5: '5',
              10: '10',
              15: '15',
              20: '20'
              },
            tooltip={"placement": "bottom", "always_visible": True},
            value=1,
            className="p-0",
        ),
    ],
    className="mb-4",
)

controls = dbc.Card(
    [Controls_title, splitter, checklist_numerical_optimizers, slider],
    body=True,
)


score = html.Div(id='score')
intercept = html.Div(id='intercept')
precision = html.Div(id='precision')
recall = html.Div(id='recall')
f1 = html.Div(id='f1')


card_1 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "LR Intercept value",
                          className="card_title text-secondary",
                      ),
                      intercept,
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )


card_2 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "LR Score",
                          className="card_title text-secondary",
                      ),
                      score
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )


card_3 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Confusion Matrix - Precision",
                          className="card_title text-secondary",
                      ),
                      precision,
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )

card_4 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Confusion Matrix - Recall",
                          className="card_title text-secondary",
                      ),
                      recall,
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )

card_5 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Confusion Matrix F1 Score",
                          className="card_title text-secondary",
                      ),
                      f1,
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )

graph_1 = dcc.Graph(
            id="confusionMatrix",
            config={"displayModeBar": False})



layout = html.Div(
  [
    header,
    html.Div(
    [
      dbc.Row([
        dbc.Col(html.Div(controls), xs=12, sm=12, md=3, lg=3, xl=3, className="mb-2"),
        dbc.Col([
          dbc.Row([
            dbc.Col(html.Div(card_1), xs=12, sm=12, md=6, lg=6, xl=4, className="mb-2"),
            dbc.Col(html.Div(card_2), xs=12, sm=12, md=6, lg=6, xl=4, className="mb-2"),
            dbc.Col(html.Div(card_3), xs=12, sm=12, md=6, lg=6, xl=4, className="mb-2"),  
          ]),
          dbc.Row([
            dbc.Col(html.Div(card_4), xs=12, sm=12, md=6, lg=6, xl=4, className="mb-2"),
            dbc.Col(html.Div(card_5), xs=12, sm=12, md=6, lg=6, xl=4, className="mb-2"),  
          ]),
          dbc.Row([
            dbc.Col(html.Div(graph_1), xs=12, sm=12,md=12, lg=12, xl=12, className="mb-2"),
      ])
        ])
      ],
        className="m-2"),
    ]),
  ],
    className="bg-light m-0 p-0"
)




@callback(
    Output("confusionMatrix", "figure"),
    Output("score", 'children'),
    Output("intercept", "children"),
    Output("precision", "children"),
    Output("recall", "children"),
    Output("f1", "children"),
    Input("splitter", "value"),
    Input("optimizer", "value"),
    Input("cfactor", "value"),
)
def make_line_chart(split, optimizer, c):
    
    y_col = 'Churn'
    feature_cols = [x for x in merged.columns if x != y_col]
    X_data = merged[feature_cols]
    y_data = merged[y_col]

    X = np.asarray(X_data)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = np.asarray(y_data)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=split, random_state=4)
    
    # Compute the Logistic Regression Model
    LR = LogisticRegression(
      C=c, 
      solver=optimizer,
      class_weight=None, dual=False, fit_intercept=True,
      intercept_scaling=1, l1_ratio=None, max_iter=100,
      multi_class='ovr', n_jobs=None, penalty='l2',
      random_state=0, tol=0.0001, verbose=0, warm_start=False).fit(X_train,y_train)
    
    INTERCEPT = LR.intercept_
    SCORE = LR.score(X_train,y_train)
    
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)
    
    # Compute the Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
    matrix_cnf = pd.DataFrame(cnf_matrix, columns=['churn=1','churn=0'], index=['churn=1','churn=0'])
    
    # Compute the Classification Report
    CLASSIFICATION_REPORT = classification_report(y_test, yhat, output_dict=True)
    classification_report_df = pd.DataFrame.from_dict(CLASSIFICATION_REPORT)
    
    precision_churn = classification_report_df['0'][0]
    recall_churn = classification_report_df['0'][1]
    f1_score_churn = classification_report_df['0'][2]
    
    
    fig = ff.create_annotated_heatmap(
        cnf_matrix, 
        x=['churn=1','churn=0'], y=['churn=1','churn=0'], 
        colorscale='Viridis'
    )
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    )
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    
    return fig, SCORE, INTERCEPT, precision_churn, recall_churn, f1_score_churn