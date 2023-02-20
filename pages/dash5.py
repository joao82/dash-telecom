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
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import mutual_info_score
import itertools


dash.register_page(__name__, name='Prediction')

# Data reading convert a CSV file to a pandas data frame
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
churn_df = pd.read_csv(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))
df = churn_df.copy()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['TotalCharges'] = df['TotalCharges'].replace(np.nan, 0)
df['TotalCharges'].isnull().sum()
df = df.drop(['customerID'], axis=1 )

df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
   df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

y_train = df_train.churn.values
y_val = df_val.churn.values

del df_train['churn']
del df_val['churn']

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']

numerical = ['tenure', 'monthlycharges', 'totalcharges']

def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')

df_train_full[numerical].corrwith(df_train_full.churn).to_frame('correlation')
df_train_full.groupby(by='churn')[numerical].mean()

train_dict = df_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)
dv.get_feature_names_out()

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model.predict_proba(X_val)
y_pred = model.predict_proba(X_val)[:, 1]
churn = y_pred > 0.5
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

subset = ['contract', 'tenure', 'totalcharges']
train_dict_small = df_train[subset].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small)

X_small_train = dv_small.transform(train_dict_small)
dv_small.get_feature_names_out()

model_small = LogisticRegression(solver='liblinear', random_state=1)
model_small.fit(X_small_train, y_train)
dict(zip(dv_small.get_feature_names_out(), model_small.coef_[0].round(3)))

val_dict_small = df_val[subset].to_dict(orient='records')
X_small_val = dv_small.transform(val_dict_small)
y_pred_small = model_small.predict_proba(X_small_val)[:, 1]


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
        dcc.Input(id="MonthlyCharges", value = 79.85, type="number", placeholder="Indicate Monthly Charges"),
    ],
    className="mb-4",
)

TotalCharges = html.Div(
    [
        dbc.Label(["Expected Customer Total Charges"], className="fw-bold"),
        dcc.Input(id="TotalCharges", value = 3320.75 ,type="number", placeholder="Indicate Total Charges"),
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
        dbc.Col( dbc.Button('Submit', id='btn', n_clicks=0), xs=12, sm=6, md=2, lg=2, xl=2, className="my-6")
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
    
    if MonthlyCharges != '' and TotalCharges != '':
        
        customer = {
            # 'customerid': '8879-zkjof',
            'gender': gender,
            'seniorcitizen': SeniorCitizen,
            'partner': Partner,
            'dependents': Dependents,
            'tenure': tenure,
            'phoneservice': PhoneService,
            'multiplelines': MultipleLines,
            'internetservice': InternetService,
            'onlinesecurity': OnlineSecurity,
            'onlinebackup': OnlineBackup,
            'deviceprotection': 'yes',
            'techsupport': TechSupport,
            'streamingtv': 'yes',
            'streamingmovies': 'yes',
            'contract': Contract,
            'paperlessbilling': 'yes',
            'paymentmethod': PaymentMethod,
            'monthlycharges': MonthlyCharges,
            'totalcharges': TotalCharges,
            }
        
        X_test = dv.transform([customer])
        prediction = model.predict_proba(X_test)[0, 1] * 100
        
        if prediction > 50:
            return html.Div([
                html.Img(src=r'assets/images/furious.png', alt='client churn', className="churn-image"),
                html.H2(
                    "Client with {} % of probability to churn.".format((round(prediction, 2))),
                    className="text-danger"
                    )
                ], className="d-flex justify-content-center align-items-center m-auto mb-4")
        else:
            return html.Div([
                html.Img(src=r'assets/images/happy.png', alt='man', className="churn-image"),
                html.H2(
                    "Client with {} % of probability to churn.".format((round(prediction, 2))),
                    className="text-success"
                    )
                ], className="d-flex align-items-center m-auto mb-4")
            
    else:
        return html.Div([html.P("Complete all the fields")])
