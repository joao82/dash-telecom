import pathlib
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
import io


dash.register_page(__name__, name='Segmentation', prevent_initial_callbacks=True)

# Data reading convert a CSV file to a pandas data frame
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
churn_df = pd.read_csv(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))
df = churn_df.copy()


# DATA CLEANING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].replace(np.nan, 0)
df = df.drop(['customerID'], axis=1 )
df['Churn'].replace(['Yes', 'No'], [1, 0], inplace=True)

df_uniques = pd.DataFrame(
  [[i, len(df[i].unique())] for i in df.columns], 
  columns=['Variable', 'Unique Values']).set_index('Variable')

df['gender'].replace(['Male', 'Female'], [1, 0], inplace=True)

# Demographics INdicators
gender = df[df['Churn']==0]
gender = gender.groupby(['gender'])['Churn'].count().reset_index()
gender['%'] = gender['Churn']/gender['Churn'].sum()*100

partner = df[df['Churn']==0]
partner = partner.groupby(['Partner'])['Churn'].count().reset_index()
partner['%'] = partner['Churn']/partner['Churn'].sum()*100

dependents = df[df['Churn']==0]
dependents = dependents.groupby(['Dependents'])['Churn'].count().reset_index()
dependents['%'] = dependents['Churn']/dependents['Churn'].sum()*100

internet = df[df['Churn']==0]
internet = internet.groupby(['InternetService'])['Churn'].count().reset_index()
internet['%'] = internet['Churn']/internet['Churn'].sum()*100


header = html.H3(
    "Customer Persona", className="text-primary p-2 mb-4 text-center"
)

sub_header = html.H3(
    "Customer Segmentation - KNN Clustering", className="text-primary p-2 mb-4 text-center"
)

dropdown_cluster = html.Div(
    [
        dbc.Label(["Select Number of Clusters"], className="fw-bold bg-light w-100"),
        dcc.Dropdown(
            id="cluster",
            options=[1,2,3,4,5],
              multi=False,
              value=4,
            clearable=False,
        ),
    ],
    className="mb-4",
)



card1 = dbc.Col([
  dbc.Row([
    dbc.Col(html.P("Gender Analysis", className="card-title")),
    ]),
  dbc.Row([
    dbc.Col([
      html.Img(src=r'assets/images/man.png', alt='man', className="card-image"),
      html.H3(f"{gender['%'][1]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
    dbc.Col([
      html.Img(src=r'assets/images/woman.png', alt='woman', className="card-image"),
      html.H3(f"{gender['%'][0]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
  ], className="m-2 p-0")
], className="persona-card")


card2 = dbc.Col([
  dbc.Row([
    dbc.Col(html.P("Marital Status", className="card-title")),
    ]),
  dbc.Row([
    dbc.Col([
      html.Img(src=r'assets/images/single.png', alt='man', className="card-image"),
      html.H3(f"{partner['%'][1]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
    dbc.Col([
      html.Img(src=r'assets/images/couple.png', alt='woman', className="card-image"),
      html.H3(f"{partner['%'][0]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
  ], className="m-2 p-0")
], className="persona-card")


card3 = dbc.Col([
  dbc.Row([
    dbc.Col(html.P("Dependents Status", className="card-title")),
    ]),
  dbc.Row([
    dbc.Col([
      html.Img(src=r'assets/images/childrens.png', alt='man', className="card-image"),
      html.H3(f"{dependents['%'][1]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
    dbc.Col([
      html.Img(src=r'assets/images/Nochildren.png', alt='woman', className="card-image"),
      html.H3(f"{dependents['%'][0]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
  ], className="m-2 p-0")
], className="persona-card")


card4 = dbc.Col([
  dbc.Row([
    dbc.Col(html.P("Internet Service Active", className="card-title")),
    ]),
  dbc.Row([
    dbc.Col([
      html.Img(src=r'assets/images/internet.png', alt='man', className="card-image"),
      html.H3(f"{internet['%'][1]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
    dbc.Col([
      html.Img(src=r'assets/images/noInternet.png', alt='man', className="card-image"),
      html.H3(f"{internet['%'][0]:,.1f}%", className="card-text text-info"),
    ], width=6, className="card-info"),
  ], className="m-2 p-0")
], className="persona-card")


graph_1 = dcc.Graph(
            id="elbow",
            config={"displayModeBar": False}
          )

graph_2 = dcc.Graph(
            id="segmentation",
            config={"displayModeBar": False}
          )



layout = html.Div(
  [
    header,
    html.Div(
    [
      dbc.Row([
        dbc.Col(html.Div(card1), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(card2), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(card3), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2"),
        dbc.Col(html.Div(card4), xs=12, sm=12, md=6, lg=6, xl=3, className="mb-2")
      ],
        className="m-2"),
      
      sub_header,
      
      dbc.Row([
        dbc.Col(html.Div(dropdown_cluster), xs=12, sm=12, md=4, lg=4, xl=4, className="mb-2"),
      ],
        className="m-2"),
      
      dbc.Row([
        dbc.Col(html.Div(graph_1), xs=12, sm=12,md=12, lg=3, xl=3, className="mb-2"),
        dbc.Col(html.Div(graph_2), xs=12, sm=12,md=12, lg=9, xl=9, className="mb-2"),
      ],
        className="m-2"),
    ]),
  ],
    className="bg-light m-0 p-0"
)



@callback(
    Output("elbow", "figure"),
    Output("segmentation", "figure"),
    Input("cluster", "value")
)
def make_line_chart(Ncluster):
  X = df.iloc[:,[4,17]].values
  
  wcss = []
  for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init=5, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

  data = pd.DataFrame({'Ncluster' : range(1,11), 'wcss' : wcss}, columns=['Ncluster','wcss'])

  kmeans = KMeans(n_clusters=Ncluster, n_init=5, init='k-means++', random_state=0)
  
  Y = kmeans.fit_predict(X)
  
  fig_line = px.line(
    data_frame=data, 
    x='Ncluster', y='wcss',
    title="The Elbow Point Graph",
    labels={"Ncluster":"Number of Clusters", "wcss":"WCSS"},
    template='gridon',
    color_discrete_map={'Yes':'lightcyan', 'No':'cyan'},
    )
  
  fig_scatter = go.Figure()
  fig_scatter.add_trace(
    go.Scatter(
        x=X[Y==0,0], y=X[Y==0,1], mode = 'markers'
    ))
  
  fig_scatter.add_trace(
    go.Scatter(
        x=X[Y==1,0], y=X[Y==1,1], mode = 'markers', fillcolor='red', name="cluster 1"
    ))
  
  fig_scatter.add_trace(
    go.Scatter(
        x=X[Y==2,0], y=X[Y==2,1], mode = 'markers', fillcolor='yellow',name="cluster 2"
    ))
  
  fig_scatter.add_trace(
    go.Scatter(
        x=X[Y==3,0], y=X[Y==3,1], mode = 'markers', fillcolor='violet',name="cluster 3"
    ))
  
  fig_scatter.add_trace(
    go.Scatter(
        x=X[Y==4,0], y=X[Y==4,1], mode = 'markers', fillcolor='blue',name="cluster 4"
    ))

  fig_scatter.update_layout(
    title="Plot Title",
    xaxis_title="X Axis Title",
    yaxis_title="Y Axis Title",
    legend_title="Legend Title",
    legend=dict(orientation="h", yanchor="bottom", y=-1.02, xanchor="right", x=1),
    clickmode='event+select'
  )
  fig_scatter.update_traces(marker_size=5)
  
  return fig_line, fig_scatter