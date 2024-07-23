# for the data visualizations of the machine learning model and more

from dash import Dash, dcc, html
from LOF_anomalies import updated_data, raw_data
import plotly.express as px

app = Dash(__name__)
server = app.server
test_dashboard = updated_data.head()


def load_test():
    fig = px.bar(data_frame=test_dashboard, x="Raw Names", y="Outlier Scores")
    return fig


app.layout = html.Div([
    # List all components of app here
    html.H1('My Dashboard!'),

    # to show a figure, we need to put inside a content divider.
    html.Div(
        dcc.Graph(id='callback_dashboard_1', figure=load_test())
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
