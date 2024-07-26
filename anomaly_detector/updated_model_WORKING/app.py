# for the data visualizations of the machine learning model and more
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from LOF_anomalies import updated_data, incidents
import plotly.express as px

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
nth_anomalies = updated_data.head()
updated_data = updated_data.astype(str)

# testing whether the 100th element of the list is proper
# subtract 2 to account for the difference in the csv to the proper location.
print(incidents[0].__str__())


def load_test():
    fig = px.bar(data_frame=nth_anomalies, x="Raw Names", y="Outlier Scores")
    return fig


def create_table():
    fig = dash_table.DataTable(
        id='table',
        data=updated_data
    )

    return fig


def find_keywords():
    keywords = set()
    for phrase in updated_data['No Stopwords']:
        words = phrase.split()
        for word in words: keywords.add(str(word))
    return keywords


app.layout = dbc.Container([
    dcc.Markdown(' # Visualizing MSFT Defender Metrics', style={'textAlign': 'center'}),

    dbc.Row([
        dbc.Col([
            dbc.Label("Show number of rows"),
            row_dropdown := dcc.Dropdown(value=10, options=[10, 20, 50, 100, 200],
                                         clearable=False),

        ]),

        dbc.Col([
            dbc.Label("Filter by keyword"),
            col_dropdown := dcc.Dropdown(options=[word for word in sorted(find_keywords())],
                                         multi=True, searchable=True),
        ]),

    ]),

    test_table := dash_table.DataTable(
        id='table',
        data=updated_data.to_dict('records'),
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
