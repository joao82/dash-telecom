import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path='/', name='Home')

layout = html.Div(
  [
        html.Div([
          html.Div([
            html.H2("Customer Churn Analysis", className="title text-primary"),
            html.H3("Holiday Community App-Building Challenge", className="sub-title text-secondary"),
            html.P("Challenge to build the most impressive customer segmentation data app to better understand a telecom consumer behavior.", className="text-secondary"),
          ], className="wrapper-title"),
          
          dbc.Nav([
              dbc.NavLink([
                html.Div(
                  html.I(className="bi bi-clipboard-data-fill"), className="home-link"
                  )
                ], href="/dash2", active="exact", className="link"),
              
              dbc.NavLink([
                html.Div(
                  html.I(className="bi bi-person-bounding-box"), className="home-link"
                  )
                ], href="/dash3", active="exact", className="link"),
              
              dbc.NavLink([
                html.Div(
                    html.I(className="bi bi-activity"), className="home-link"
                    )
                ], href="/dash4", active="exact", className="link"),
              
              dbc.NavLink([
                html.Div(
                    html.I(className="bi bi-easel-fill"), className="home-link"
                    )
                ], href="/dash5", active="exact", className="link")
              
            ], pills=True, className="d-flex justify-content-between")
        ])
          
  ], className="menu"
)
