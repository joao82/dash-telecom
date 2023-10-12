import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
from dash import Input, Output, dcc, html, State

font_awesome = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
meta_tags = [{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
external_stylesheets = [meta_tags, font_awesome, dbc.themes.SPACELAB, dbc.icons.BOOTSTRAP]

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

url_theme1 = dbc.themes.SPACELAB
template_theme1 = "spacelab"
url_theme2 = dbc.themes.DARKLY
template_theme2 = "darkly"

app = dash.Dash(
    __name__,
    use_pages=True,
    assets_folder="assets",
    external_stylesheets=external_stylesheets,
    prevent_initial_callbacks=True,
)

server = app.server

theme_switch = ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2])

theme_colors = ["primary", "secondary", "success", "warning", "danger", "info", "light", "dark", "link"]


NavLinks = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div([theme_switch], className="d-flex align-items-center my-2"),
                        html.Div(
                            [
                                dbc.NavLink(
                                    [
                                        html.Div(page["name"], className="text-light ms-3"),
                                    ],
                                    href=page["path"],
                                    active="exact",
                                )
                                for page in dash.page_registry.values()
                            ],
                            className="d-flex justify-content-end me-2",
                        ),
                    ],
                    className="d-flex justify-content-end align-items-center",
                )
            ],
            className="d-flex justify-content-end mb-2",
        )
    ]
)


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Plotly", className="ms-2")),
                ],
                align="center",
                className="g-0",
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0, className="m-1"),
            dbc.Collapse(NavLinks, id="navbar-collapse", is_open=False, navbar=True),
        ]
    ),
    className="m-0 p-0",
    color="dark",
    dark=True,
)


content = html.Div(dash.page_container, id="page-content")


app.layout = dbc.Container([dbc.Col([navbar, content])], className="d-flex align-items-center m-0 p-0", fluid=True)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True, port=5000)
