import pathlib
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing

import dash
from dash import Input, Output, callback, dcc, html, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
import itertools


dash.register_page(__name__, name='Prediction')


# Data reading convert a CSV file to a pandas data frame
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
churn_df = pd.read_csv(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))
df = churn_df.copy()

# DATA CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].replace(np.nan, 0)
df['Churn'].replace(['Yes', 'No'], [1, 0], inplace=True)
df = df.drop(['customerID','PaperlessBilling','StreamingTV','StreamingMovies','DeviceProtection'], axis=1 )


df['gender'].replace(['Female', 'Male'], [1, 0], inplace=True)
df['Partner'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['Dependents'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['PhoneService'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['MultipleLines'].replace(['No phone service', 'No', 'Yes'], [2, 1, 0], inplace=True)
df['InternetService'].replace(['DSL', 'Fiber optic', 'No'], [2, 1, 0], inplace=True)
df['OnlineSecurity'].replace(['No', 'Yes', 'No internet service'], [1, 2, 0], inplace=True)
df['OnlineBackup'].replace(['No', 'Yes', 'No internet service'], [1, 2,0], inplace=True)
df['TechSupport'].replace(['No', 'Yes', 'No internet service'], [1, 2,0], inplace=True)
df['Contract'].replace(['Month-to-month', 'One year', 'Two year'], [1, 2,0], inplace=True)
df['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], [3, 2, 1, 0], inplace=True)

y_col = 'Churn'
feature_cols = [x for x in df.columns if x != y_col]

X_data = df[feature_cols]
y_data = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=4)

# Compute the Logistic Regression Model
LR = LogisticRegression(
    C=0.01, 
    solver='liblinear',
    class_weight=None, dual=False, fit_intercept=True,
    intercept_scaling=1, l1_ratio=None, max_iter=100,
    multi_class='ovr', n_jobs=None, penalty='l2',
    random_state=0, tol=0.0001, verbose=0, warm_start=False).fit(X_train,y_train)


header = html.H3(
    "Telecom Customer Churn Prediction Model", className="text-primary p-2 mb-4 text-center"
)

Controls_title = html.H5(
    "Demographic controls", className="text-primary p-1 mb-2 text-center"
)

gender = html.Div(
    [
        dbc.Label(["Select the Customer Gender"], className="fw-bold"),
        dbc.RadioItems(
            id="gender",
            options=[
              {'label': 'Male', 'value': 'Male'},
              {'label': 'Female', 'value': 'Female'}
              ],
            value='Male',
            inline=True,
        ),
    ],
    className="mb-4",
)

SeniorCitizen = html.Div(
    [
        dbc.Label(["Select the Customer Age"], className="fw-bold"),
        dbc.RadioItems(
            id="SeniorCitizen",
            options=[
              {'label': '> 65 years', 'value': 0},
              {'label': '< 65 years', 'value': 1}
              ],
            value=0,
            inline=True,
        ),
    ],
    className="mb-4",
)

Partner = html.Div(
    [
        dbc.Label(["Select Marital Status"], className="fw-bold"),
        dbc.RadioItems(
            id="Partner",
            options=[
              {'label': 'Married', 'value': 'Yes'},
              {'label': 'Single', 'value': 'No'}
              ],
            value='Yes',
            inline=True,
        ),
    ],
    className="mb-4",
)

Dependents = html.Div(
    [
        dbc.Label(["Have the Customer Dependents?"], className="fw-bold"),
        dbc.RadioItems(
            id="Dependents",
            options=[
              {'label': 'Yes', 'value': 'Yes'},
              {'label': 'No', 'value': 'No'}
              ],
            value='Yes',
            inline=True,
        ),
    ],
    className="mb-4",
)


slider = html.Div(
    [
        dbc.Label(["Select the expected customer tenure"], className="fw-bold"),
        dcc.Slider(
            0,
            80,
            1,
            id="tenure",
            marks={
              0: '0',
              20: '20',
              40: '40',
              60: '60',
              80: '80'
              },
            tooltip={"placement": "bottom", "always_visible": True},
            value=20,
            className="p-0",
        ),
    ],
    className="mb-4",
)


# ------ Service --------
PhoneService = html.Div(
    [
        dbc.Label(["Customer has Phone Service?"], className="fw-bold"),
        dbc.RadioItems(
            id="PhoneService",
            options=[
              {'label': 'Yes', 'value': 'Yes'},
              {'label': 'No', 'value': 'No'}
              ],
            value='No',
            inline=True,
        ),
    ],
    className="mb-4",
)

MultipleLines = html.Div(
    [
        dbc.Label(["Customer has Multiple Lines?"], className="fw-bold"),
        dcc.Dropdown(['Yes', 'No', 'No phone service'], 'Yes', id='MultipleLines'),
    ],
    className="mb-4",
)

InternetService = html.Div(
    [
        dbc.Label(["What Internet Service?"], className="fw-bold"),
        dcc.Dropdown(['DSL', 'No', 'Fiber optic'], 'DSL', id='InternetService'),
    ],
    className="mb-4",
)

OnlineSecurity = html.Div(
    [
        dbc.Label(["Customer has online Security?"], className="fw-bold"),
        dcc.Dropdown(['Yes', 'No', 'No internet service'], 'Yes', id='OnlineSecurity'),
    ],
    className="mb-4",
)

OnlineBackup = html.Div(
    [
        dbc.Label(["Customer has online Security?"], className="fw-bold"),
        dcc.Dropdown(['Yes', 'No', 'No internet service'], 'Yes', id='OnlineBackup'),
    ],
    className="mb-4",
)

TechSupport = html.Div(
    [
        dbc.Label(["Customer has Tech Support?"], className="fw-bold"),
        dcc.Dropdown(['Yes', 'No', 'No internet service'], 'Yes', id='TechSupport'),
    ],
    className="mb-4",
)

Contract = html.Div(
    [
        dbc.Label(["What type of contract?"], className="fw-bold"),
        dcc.Dropdown(['Month-to-month', 'One year', 'Two year'], 'One year', id='Contract'),
    ],
    className="mb-4",
)

PaymentMethod = html.Div(
    [
        dbc.Label(["Customer Payment Method?"], className="fw-bold"),
        dcc.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 'Bank transfer (automatic)', id='PaymentMethod'),
    ],
    className="mb-4",
)

MonthlyCharges = html.Div(
    [
        dbc.Label(["Expected Customer Monthly Charges"], className="fw-bold"),
        dcc.Input(id="MonthlyCharges", type="number", placeholder="1000"),
    ],
    className="mb-4",
)

TotalCharges = html.Div(
    [
        dbc.Label(["Expected Customer Total Charges"], className="fw-bold"),
        dcc.Input(id="TotalCharges", type="number", placeholder="1000"),
    ],
    className="mb-4",
)



control_demographic = dbc.Card(
    [gender, SeniorCitizen, Partner, Dependents, slider],
    body=True,
)

control_services1 = dbc.Card(
    [PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup],
    body=True,
)

control_services2 = dbc.Card(
    [TechSupport, Contract, PaymentMethod, MonthlyCharges, TotalCharges],
    body=True,
)


form = dbc.Form([
    dbc.Row([
        dbc.Col(html.Div(control_demographic), xs=12, sm=12, md=4, lg=4, xl=4, className="mb-2"),
        dbc.Col(html.Div(control_services1), xs=12, sm=12, md=4, lg=4, xl=4, className="mb-2"),
        dbc.Col(html.Div(control_services2), xs=12, sm=12, md=4, lg=4, xl=4, className="mb-2")
        ]),
    dbc.Row([
        dbc.Col( html.Button('Submit', id='btn', n_clicks=0), xs=12, sm=6, md=2, lg=2, xl=2, className="")
        ],
        className="d-flex justify-content-center m-2"),
])

layout = dbc.Container([
    header,
    html.Div(
    [
      form,
      dbc.Row([
        dbc.Col(html.Div(id='result'), xs=12, sm=12, md=12, lg=12, xl=12, className="d-flex justify-content-center m-4")
        ],
        className="m-2")
    ]),
  ])


@callback(
    Output('result', 'children'),
    Input('btn', 'n_clicks'),
    State('gender', 'value'),
    State('SeniorCitizen', 'value'),
    State('Partner', 'value'),
    State('Dependents', 'value'),
    State('tenure', 'value'),
    State('PhoneService', 'value'),
    State('MultipleLines', 'value'),
    State('InternetService', 'value'),
    State('OnlineSecurity', 'value'),
    State('OnlineBackup', 'value'),
    State('TechSupport', 'value'),
    State('Contract', 'value'),
    State('PaymentMethod', 'value'),
    State('MonthlyCharges', 'value'),
    State('TotalCharges', 'value'),
)
def update_output(n_clicks, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,  OnlineBackup, TechSupport, Contract, PaymentMethod, MonthlyCharges, TotalCharges):
    
    if MonthlyCharges != '':
        form_data = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, TechSupport, Contract, PaymentMethod, MonthlyCharges, TotalCharges]
        
        columns_data = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        
        res = {}
        for key in columns_data:
            for value in form_data:
                res[key] = value
                form_data.remove(value)
                break
            
        X = pd.DataFrame(res, index=[0])
        
        X['gender'].replace(['Female', 'Male'], [1, 0], inplace=True)
        X['Partner'].replace(['Yes', 'No'], [1, 0], inplace=True)
        X['Dependents'].replace(['Yes', 'No'], [1, 0], inplace=True)
        X['PhoneService'].replace(['Yes', 'No'], [1, 0], inplace=True)
        X['MultipleLines'].replace(['No phone service', 'No', 'Yes'], [2,1, 0], inplace=True)
        X['InternetService'].replace(['DSL', 'Fiber optic', 'No'], [2,1, 0], inplace=True)
        X['OnlineSecurity'].replace(['No', 'Yes', 'No internet service'], [1,2, 0], inplace=True)
        X['OnlineBackup'].replace(['No', 'Yes', 'No internet service'], [1,2, 0], inplace=True)
        X['TechSupport'].replace(['No', 'Yes', 'No internet service'], [1, 2,0], inplace=True)
        X['Contract'].replace(['Month-to-month', 'One year', 'Two year'], [1, 2,0], inplace=True)
        X['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], [3, 2, 1, 0], inplace=True)
        
        result = LR.predict(X)
        if result == 1:
            return html.Div([
                html.I(className="bi bi-x-lg text-danger"),
                html.H2("Customer Churn", className="text-danger")
                ])
        else:
            return html.Div([
                html.I(className="bi bi-check2-circle text-danger"),
                html.H2("Customer Not Churn", className="text-success")
                ])
    else:
        return html.Div([html.P("Complete all the fields")])
