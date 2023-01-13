import pathlib
import dash
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy import stats

dash.register_page(__name__, name='Exploration')

# Data reading convert a CSV file to a pandas data frame
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
churn_df = pd.read_csv(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))

# Data cleaning
df = churn_df.copy()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].replace(np.nan, 0)
df = df.drop(['customerID'], axis=1 )

tenure = df['tenure'].unique().tolist().sort(reverse=True)

# Customer Churn Indicators for Cards
churn_rate = (df['Churn'] == "Yes").sum()/df["Churn"].count()*100
return_rate = 100-churn_rate
recurring_revenue = df[df['Churn'] == "Yes"]['TotalCharges'].sum()/(df['TotalCharges'].sum())*100
MRR = df[df['Churn'] == 1]['TotalCharges'].sum()
risk_customers = len(df[(df['Churn'] == 0) & (df['tenure'] < 10) ])
income_risk_customers = df[(df['Churn'] == 0) & (df['tenure'] < 10) ]['MonthlyCharges'].sum()


header = html.H3(
    "Theme Explorer Sample App", className="text-primary p-2 mb-4 text-center"
)

checklist_gender = html.Div(
    [
        dbc.Label(["Select Gender"], className="fw-bold w-100"),
        dbc.Checklist(
            id="gender",
            options=[
              {'label': 'Male', 'value': 'Male'},
              {'label': 'Female', 'value': 'Female'},
              ],
            value=['Male', 'Female'],
            inline=True,
        ),
    ],
    className="mb-4",
)

checklist_senior = html.Div(
    [
        dbc.Label(["Select Seniority"], className="fw-bold w-100"),
        dbc.Checklist(
            id="senior",
            options=[
                {'label': 'Below 65', 'value': 0},
                {'label': 'Above 65', 'value': 1},
              ],
            value=[0,1],
            inline=True,
        ),
    ],
    className="mb-4",
)

dropdown = html.Div(
    [
        dbc.Label(["Select Partner"], className="fw-bold w-100"),
        dcc.Dropdown(
            id="partner",
            options=[
              {"label": 'Yes', "value": 'Yes'}, 
              {"label": 'No', "value": 'No'}, 
              ],
              multi=True,
              value=["Yes"],
            clearable=False,
        ),
    ],
    className="mb-4",
)

slider = html.Div(
    [
        dbc.Label(["Select Tenure"], className="fw-bold w-100"),
        dcc.RangeSlider(
            0,
            72,
            1,
            id="tenure",
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            value=[0, 72],
            className="p-0",
        ),
    ],
    className="mb-4",
)


controls = dbc.Card(
    [checklist_gender, checklist_senior, dropdown, slider],
    body=True,
)


pie_chart = dcc.Graph(id="churn_rate", config={"displayModeBar": False})
bar_chart1 = dcc.Graph(id="gender_share", config={"displayModeBar": False})
bar_chart2 = dcc.Graph(id="churn_contract", config={"displayModeBar": False})
bar_chart3 = dcc.Graph(id="churn_payment", config={"displayModeBar": False})
bar_chart4 = dcc.Graph(id="churn_billing", config={"displayModeBar": False})
bar_chart5 = dcc.Graph(id="churn_tenure", config={"displayModeBar": False})
bar_chart6 = dcc.Graph(id="churn_MonthlyCharges", config={"displayModeBar": False})
bar_chart7 = dcc.Graph(id="churn_streaming", config={"displayModeBar": False})

card_1 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Customer Churn Rate",
                          className="card_title text-secondary",
                      ),
                      html.P(
                        f"{churn_rate:,.2f}%",
                        id="card1",
                        className="card-text mb-0 fs-4 text",
                      ),
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )

card_2 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Customer Return Rate",
                          className="card_title text-secondary",
                      ),
                      html.P(
                        f"{return_rate:,.2f}%",
                        id="card1",
                        className="card-text mb-0 fs-4 text",
                      ),
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )

card_3 = dbc.Card(
              dbc.CardBody(
                  [
                      dbc.Label(
                          "Recurring Revenue",
                          className="card_title text-secondary",
                      ),
                      html.P(
                        f"{recurring_revenue:,.2f}%",
                        id="card1",
                        className="card-text mb-0 fs-4 text",
                      ),
                  ],
                  className="text-center text-info mb-1",
              ),
              className="card-wrapper",
          )


layout = html.Div(
  [
    header,
    html.Div(
    [
      dbc.Row([
        dbc.Col(html.Div(controls), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(pie_chart), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(bar_chart1), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(bar_chart2), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2")
      ],
        className="m-2"),
      
      dbc.Row([
        dbc.Col(html.Div(bar_chart3), xs=12, sm=12, md=6, lg=3, xl=3, className="mb-2"),
        dbc.Col(html.Div(bar_chart4), xs=12, sm=12, md=6, lg=3, xl=3, className="mb-2"),
        dbc.Col(html.Div(bar_chart5), xs=12, sm=12, md=12, lg=6, xl=6, className="mb-2")
      ],
        className="m-2"),
      
      dbc.Row([
        dbc.Col(html.Div(bar_chart6), xs=12, sm=12,md=12, lg=9, xl=9),
        dbc.Col([
          html.Div(card_1, className="mb-2"),
          html.Div(card_2, className="mb-2"),
          html.Div(card_3, className="mb-2"),
          ], xs=12, sm=12,md=12, lg=3, xl=3,
        ),
      ],
        className="m-2")
    ]),
  ],
    className="bg-light m-0 p-0"
)




@callback(
    Output("churn_rate", "figure"),
    [  
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_pie_chart(gen, sen, part):
  churn_df = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]
  churn_df = churn_df.groupby(['Churn'])['TotalCharges'].sum().reset_index()
  
  fig = px.pie(
            data_frame=churn_df, 
            values='TotalCharges',
            color="Churn",
            names='Churn',
            hole=.3, 
            color_discrete_map={'Yes':'lightcyan', 'No':'cyan',},
            title="Churn vs Not Churn",
            template='gridon',
            hover_data=['TotalCharges'], 
            labels={'Total Charges':'TotalCharges'}
          )
  fig.update_traces(textposition='inside', textinfo='percent+label')
  fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)

  return fig




@callback(
    Output("gender_share", "figure"),
    [  
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_1(gen, sen, part):  
  churn_gender = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]
  churn_gender = churn_gender.groupby(['gender','Churn'])['TotalCharges'].sum().reset_index()
  
  fig = px.histogram(
    data_frame=churn_gender, 
    x='gender', y='TotalCharges', histfunc='sum', facet_col="Churn",
    text_auto='.2s', orientation="v", barmode='relative', 
    title="Gender Share",
    labels={"TotalCharges":"Amount Charged until Q3", "gender":"Gender"},
    template='gridon',
    color_discrete_map={'Yes':'lightcyan', 'No':'cyan', 'Male':'red', 'Female':'green'},
    range_y=[0,5000000],
    hover_name='Churn',
    hover_data=['Churn']
    )
  fig.update_layout(bargap=0.1)
  fig.update_traces(textposition='inside', text='percent+label')
  fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
  
  return fig



@callback(
    Output("churn_contract", "figure"),
    [  
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_2(gen, sen, part):
  churn_contract = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]
  churn_contract = churn_contract.groupby(['Contract','Churn'])['TotalCharges'].sum().reset_index()
  
  fig = px.bar(
    data_frame=churn_contract, 
    x='Contract', y='TotalCharges', color="Churn",
    text_auto='.2s', orientation="v", barmode='relative', 
    title="Churn by Contract",
    labels={"TotalCharges":"Amount Charged until Q3", "Contract":"Contract Models"},
    template='gridon',
    color_discrete_map={'Yes':'lightcyan', 'No':'cyan'},
    text='Churn',
    hover_name='Churn',
    hover_data=['Churn']
    )
  fig.update_traces(textposition='inside', text='percent+label')
  fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', showlegend=False)
  
  return fig




@callback(
    Output("churn_payment", "figure"),
    [
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_3(gen, sen, part):
  churn_payment = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]
  churn_payment = churn_payment.groupby(['PaymentMethod','Churn'])['MonthlyCharges', 'TotalCharges'].sum().reset_index()
  
  fig = px.scatter(
    data_frame=churn_payment,
    x="Churn", y="MonthlyCharges",
	  size="TotalCharges", color="PaymentMethod",
    log_y=True, size_max=60,
    color_discrete_map={"Electronic check": "lightcyan" ,"Mailed check":"cyan", "Bank transfer": "darkcyan" ,"Credit card":"grey"},
    title="Churn by Payment Method",
    labels={"MonthlyCharges":"Monthly Amount Charged until Q3", "Churn":"Customer Churn or Not"},
    )
  
  fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-1.02,
    xanchor="right",
    x=1)
  )
  
  return fig




@callback(
    Output("churn_billing", "figure"),
    [
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_4(gen, sen, part):
  churn_billing = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]

  fig = px.sunburst(
    churn_billing, 
    path=['Churn', 'Contract'], 
    values='TotalCharges',
    color='Churn', hover_data=['TotalCharges'],
    color_discrete_map={"Electronic check": "lightcyan" ,"Mailed check":"cyan", "Bank transfer": "darkcyan" ,"Credit card":"grey"},
    title="Churn by Payment Method",
    labels={"MonthlyCharges":"Monthly Amount Charged until Q3", "Churn":"Customer Churn or Not"},
    )
  
  return fig




@callback(
    Output("churn_tenure", "figure"),
    [
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_5(gen, sen, part):
  churn_tenure = df[(df['gender'].isin(gen)) & (df['SeniorCitizen'].isin(sen)) & (df['Partner'].isin(part))]
  churn_tenure = churn_tenure.groupby(['tenure' ,'Churn'])["MonthlyCharges"].sum().reset_index()

  fig = px.scatter(
    data_frame=churn_tenure, 
    y='MonthlyCharges', x='tenure', color="Churn",
    title="Total Charges vs tenure",
    color_discrete_map={'Yes':'cyan', 'No':'darkcyan'},
    labels={"MonthlyCharges":"Total Monthly Amount Charged until Q3", "tenure":"Total Months Client until Churn"},
    )
  
  fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1)
  )

  fig.update_yaxes(range=[0, 6], type="log")

  return fig




@callback(
    Output("churn_MonthlyCharges", "figure"),
    [
      Input("gender", "value"),
      Input("senior", "value"),
      Input("partner", "value")
    ]
)
def make_bar_chart_6(gen, sen, part):
  churn_charges = df[(df['gender'].isin(gen)) & 
                     (df['SeniorCitizen'].isin(sen)) & 
                     (df['Partner'].isin(part))
                     ]
  q = churn_charges["TotalCharges"].quantile(0.99)
  churn_charges[churn_charges["TotalCharges"] < q]
  q_low = churn_charges["TotalCharges"].quantile(0.01)
  q_hi  = churn_charges["TotalCharges"].quantile(0.99)

  df_filtered = churn_charges[(churn_charges["TotalCharges"] < q_hi) & (churn_charges["TotalCharges"] > q_low)]
  
  churn_charges = churn_charges.groupby(['tenure','Churn'])['MonthlyCharges','TotalCharges'].sum().reset_index()
  
  fig = px.bar(
    data_frame=churn_charges, 
    y='MonthlyCharges', x='tenure', color="Churn",
    text_auto='.2s', orientation="v", barmode='relative',
    title="Churn by Tenure",
    color_discrete_map={'Yes':'cyan', 'No':'darkcyan'},
    text='Churn',
    hover_name='Churn',
    hover_data=['Churn']
    )
  fig.update_traces(textposition='inside', text='percent+label')
  fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    
  return fig



