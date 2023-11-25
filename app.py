import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash import Input, Output, dcc, html, dash_table, State

import pickle
import json
from datetime import datetime

from utils import *
#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
#from pyspark.sql import SparkSession

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
ajax_js = "https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"

original_data = pd.read_csv("data/company.csv", encoding='unicode_escape')
original_data = original_data.dropna()
original_data.CustomerID = original_data.CustomerID.astype(int)
original_data = original_data[original_data.Quantity > 0]

#session = SparkSession.builder.appName("Recommendation").getOrCreate()
#data_recommendation = make_recommendation(original_data)
#session.stop()

data_recommendation = pd.read_csv("data/recommendation_data.csv")
data_recommendation.columns = ["CustomerID", "StockCode", "Description"]

data_recommendation["CustomerID"] = data_recommendation["CustomerID"].astype(str)
data_recommendation["StockCode"] = data_recommendation["StockCode"].astype(str)
data_recommendation["Description"] = data_recommendation["Description"].astype(str)

prep_data = preprocess(original_data)
features = create_rfm_feature(prep_data)

classif_model = pickle.load(open("data/classif_model.pkl", 'rb'))
reg_model = pickle.load(open("data/reg_model.pkl", 'rb'))

pred_data = predict_values(features, reg_model, classif_model)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css], external_scripts=[ajax_js])
server = app.server

filters = dbc.Card(
    [
        html.H4("Filters", className="card-title"),
	dbc.Row([
		
		dbc.Col(html.Div(
		    [
		        dbc.Label("Likelihood of buying"),
			dcc.RangeSlider(id='prob', min=0., max=pred_data.ProbBuy.max(), step=0.01,
                                 value=[0., pred_data.ProbBuy.max()], tooltip={"placement": "bottom", "always_visible": True}, 
                                 marks={0.: '0.0', pred_data.ProbBuy.max(): str(pred_data.ProbBuy.max().round(2))}),
		    ]
		), md="6"),
		dbc.Col(html.Div(
		    [
		        dbc.Label("Value to be generated"),
			dcc.RangeSlider(id='pred-value', min=0.0, max=pred_data.PredictedCLV.max(),
                                 value=[0., pred_data.PredictedCLV.max()], tooltip={"placement": "bottom", "always_visible": True},
                                 marks={pred_data.PredictedCLV.min(): '$' + str(0.0), pred_data.PredictedCLV.max(): '$' + str(pred_data.PredictedCLV.max().round(2))}),
		    ]
		), md="6"),
	    ]),
    ],
    className="shadow rounded",
    body=True,
)

table = dash_table.DataTable(
    id="table",
    columns=[{"name": i, "id": i} for i in pred_data.drop("ExpectedValue", axis=1).columns],
    data=[],
    sort_action="native",
    sort_mode="multi",
    row_selectable="multi",
    selected_rows = [],
    filter_action='custom',
    filter_query='',
    page_action='none',
    fixed_rows={'headers': True},
    style_table={'maxHeight': '400px', 'overflowY': 'auto', 'overflowX': 'auto'},
    style_cell={
        'minWidth': 120,
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        },
    ],
    
)
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3
 
modal = html.Div(
    [
        html.Div(dbc.Button("Features Description", size="md", color="info", id="open-describe-feature"), className='mt-2 d-grid gap-2 d-md-flex justify-content-md-end'),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Feature Description"), close_button=True),
                dbc.ModalBody(
		
			html.Div([
				dbc.Row([
                                   dbc.Col(
                                         dbc.Card([
                                                  html.Ul([
							html.Li([html.B("CustomerID"), ": A unique identifier assigned to each customer."]),
							html.Li([html.B("Recency"), ": Measures the period of time, in days, since a customer's last transaction. The lower the value, the more recent the customer, i.e. -2 means the last purchase was made 2 days ago."]),
							html.Li([html.B("Frequency"), ": The total number of transactions made by a customer over a given period."]),
							html.Li([html.B("MonetarySum"), ": The total amount of money spent by a customer on all their transactions."]),
							html.Li([html.B("MonetaryMean"), ": The average amount spent by a customer per transaction."]),
							
							html.Li([html.B("ProbBuy (Probability of Purchase)"), ": The estimated probability that a customer will make a future purchase in 80 days."]),
						  ]),
                                         ],
                                         className="shadow rounded align-items-center justify-content-center p-2 mb-2"),
                                         md="6",
                                   ),
                                   dbc.Col(
                                         dbc.Card([
                                                  html.Ul([
							html.Li([html.B("InvoiceNo"), ": A unique identifier assigned to each transaction or invoice."]),
							html.Li([html.B("InvoiceDate"), ": The date on which a transaction was carried out."]),
							html.Li([html.B("StockCode"), ": A unique code assigned to each product or item in stock."]),
							html.Li([html.B("Description"), ": A brief description of the product or item in stock."]),
							html.Li([html.B("Quantity"), ": The number of units of a product sold in a specific transaction."]),
							html.Li([html.B("UnitPrice"), ": The price of one unit of the product sold."]),
							html.Li([html.B("PredictedCLV"), ": The estimated customer lifetime value, i.e. the prediction of the future monetary value that a customer will generate for the company in 80 days."]),
						  ]),
                                                  
                                         ],
                                         className="shadow rounded align-items-center justify-content-center p-2 mb-2"),
                                          md="6",
                                )],align="top", className="mb-1"),
                          
                        ],className="rounded align-items-top justify-content-center p-2 mt-1")
		),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-describe-feature",
                        className="ms-auto btn-info",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-describe-feature",
            size="xl",
            centered=True,
            is_open=False,
        ),
    ]
)

modal_marketing = html.Div(
    [
        html.Div(dbc.Button("Marketing Strategies", size="md", color="info", id="open-marketing"), className='mt-2 d-grid gap-2 d-md-flex justify-content-md-end'),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Marketing Strategies Recommendation"), close_button=True),
                dbc.ModalBody(
			html.Div([
				dbc.Row([
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4("Low probability and low value generation"),
						  html.Ul([
							html.Li(["Avoid spending valuable resources targeting these customers."]),
							html.Li(["Concentrate on more promising segments."]),
							html.Li(["Use email marketing to maintain minimal contact at low cost."]),
							html.Li(["Consider automation to manage these contacts without the need for constant human intervention."]),
						  ]),
                                         ],
                                         className="shadow rounded align-items-center justify-content-center bg-danger p-2 mb-2"),
                                         md="6",
                                   ),
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4("High probability, low value generation"),
						  html.Ul([
							html.Li(["Automate targeted marketing campaigns to minimize operational costs."]),
							html.Li(["Offer loyalty incentives, such as discounts on future purchases."]),
							html.Li(["Explore cross-selling strategies to increase the value of each transaction."]),
						  ]),
                                                  
                                         ],
                                         className="shadow rounded align-items-center justify-content-center bg-warning p-2 mb-2"),
                                          md="6",
                                )],align="top", className="mb-1"),

				dbc.Row([
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4("Low probability and high value generation"),
						  html.Ul([
							html.Li(["Use targeted social media advertising or paid ads to increase your brand's visibility to these customers."]),
							html.Li(["Create premium content, such as white papers or webinars, to grab their attention."]),
							html.Li(["Set up a loyalty programme to reward conversions, as the value generated by a high customer can offset acquisition costs."]),
						  ]),
                                         ],
                                         className="shadow rounded align-items-center justify-content-center bg-info p-2 mb-2"),
                                         md="6",
                                   ),
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4("High probability, high value generation"),
						  html.Ul([
							html.Li(["Invest in high-end advertising campaigns across channels such as TV, social media or events."]),
							html.Li(["Create high-end loyalty programmes with exclusive benefits for these customers."]),
							html.Li(["Personalise offers according to their preferences and behaviour to maximize the value at each stage of the customer journey."]),
						  ]),
                                                  
                                         ],
                                         className="shadow rounded align-items-center justify-content-center bg-success p-2 mb-2"),
                                          md="6",
                                )],align="top", className="mb-1"),
                          
                        ],className="shadow rounded align-items-center justify-content-center p-2 mt-1")

		),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-marketing",
                        className="ms-auto btn-info",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-marketing",
	    size="xl",
            centered=True,
            is_open=False,
        ),
    ]
)

details_table = dash_table.DataTable(
  			id="details-table",
    			columns=[{"name": i, "id": i} for i in original_data.columns],
    			data=[],
    			sort_action="native",
    			sort_mode="multi",
		    	filter_action='custom',
		    	filter_query='',
                        page_action='none',
	                fixed_rows={'headers': True},
                        style_table={'maxHeight': '400px', 'overflowY': 'auto', 'overflowX': 'auto'},
			style_cell={
			        'minWidth': 120,
			},
		    	style_data_conditional=[
				{
			    	'if': {'row_index': 'odd'},
			    	'backgroundColor': 'rgb(220, 220, 220)',
				},
		    	],)

recommendation_table = dash_table.DataTable(
  			id="recommendation-table",
    			columns=[{"name": i, "id": i} for i in data_recommendation.columns],
    			data=[],
    			sort_action="native",
    			sort_mode="multi",
		    	filter_action='custom',
		    	filter_query='',
                        page_action='none',
			fixed_rows={'headers': True},
                        style_table={'maxHeight': '400px', 'overflowY': 'auto', 'overflowX': 'auto'},
			style_cell={
			        'minWidth': 120,
			},
		    	style_data_conditional=[
				{
			    	'if': {'row_index': 'odd'},
			    	'backgroundColor': 'rgb(220, 220, 220)',
				},
		    	],)

selected_details_data = {}
selected_recommendation_data = {}

recommendation_modal = html.Div(
    [
        html.Button("Product Recommendation", className="btn btn-md btn-info mb-1",id="open-recommendation", n_clicks=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Product Recommendation Per Customer"), close_button=True),
                dbc.ModalBody([
			
			recommendation_table,
		]),
                dbc.ModalFooter([
                    html.Div([
				html.Button("Download Recommendation", className="btn btn-md btn-info mb-1", id="download-recommendation", n_clicks=0),
				dcc.Download(id="download-product-recommendation"),
                                dbc.Button("Close",id="close-recommendation",className="md-auto btn-info",n_clicks=0,)
				], className='ml-4 d-grid gap-2 d-flex justify-content-between align-items-center'),

		    
		    ]),
            ],
            id="modal-recommendation",
            size="xl",
            centered=True,
	    scrollable=True,
            is_open=False,
        ),
    ],
)


details_modal = html.Div(
    [
        html.Button("Display Details", className="btn btn-md btn-info mb-1",id="open-details", n_clicks=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Customer Purchase Details"), close_button=True),
                dbc.ModalBody([
			details_table,
		]),
                dbc.ModalFooter(
		    html.Div([
				recommendation_modal,
                                dbc.Button(
				        "Close",
				        id="close-details",
				        className="md-auto btn-info",
				        n_clicks=0,
				)
		    ], className='d-grid gap-2 d-flex justify-content-between align-items-center')),
            ],
            id="modal-details",
            size="xl",
            centered=True,
	    scrollable=True,
            is_open=False,
        ),
    ],
)


app.title = "Customer Lifetime Value Prediction"
app.layout = dbc.Container(
    [

        html.H1("Customer Lifetime Value Prediction", className="border-bottom"),
        dbc.Row(
            [
                dbc.Col(
			dbc.Container(
			[
			  dbc.Col(filters, className='mb-3',),
			  dbc.Col(dbc.Card([
                                      html.Div(
				      [
                                      	html.Div(
					[
                                          dbc.Checklist(options=[{"value": 1},], id="select-all-button"),
                                          html.Div(details_modal, id="display-details-button", style= {'display': 'none'}),
					], className='mx-2 d-grid gap-2 d-flex align-items-center'),
                                        html.Button("Export Data", className="btn btn-md btn-info mb-1", id="download-button", n_clicks=0), dcc.Download(id="download-clv-prediction"),
				      ], className='ml-4 d-grid gap-2 d-flex justify-content-between align-items-center'),
                                      html.Div(table, style={'Height': '10%', 'Width': '100%','overflow': 'auto'}), 
                                      modal], className="shadow rounded p-2")),
                          
			]),
		md=6),

                dbc.Col(
			dbc.Container(
            		[
                          dbc.Row([
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4(id="expected-value"),
                                                  html.H6("Total Expected Value"),
                                                  
                                         ],
                                         className="shadow rounded align-items-center justify-content-center p-2 mb-2"), md="6",),
                                   dbc.Col(
                                         dbc.Card([
                                                  html.H4(id="expected-pct"),
                                                  html.H6("Pct Expected Value"),
                                                  
                                         ],
                                         className="shadow rounded align-items-center justify-content-center p-2 mb-2"), md="6",)
				   ], align="center", className="mb-1 align-items-center"),

                          dbc.Col(dbc.Card([html.Div(dcc.Graph(id="cluster-graph", responsive="auto", style={'Width': '100%', 'Height': '100%'}), ), modal_marketing], className="shadow rounded p-2")),
                          dbc.Col(dbc.Card([
                                                  html.Span(["Developped by: ", html.B(html.A("Pierjos Francis COLERE MBOUKOU", href="https://www.linkedin.com/in/pierjos-colere/", target="_blank")), ". See his ", html.A("LinkedIn", href="https://www.linkedin.com/in/pierjos-colere/", target="_blank")]),
                                         ],
                                         className="shadow rounded align-items-center justify-content-center p-2 mt-1"))
			]),
		md=6),
            ],
            align="top",
        ),
        html.Div(id='selected-data'),
    ],
    fluid=True,
)


@app.callback(
    [Output("cluster-graph", "figure"), Output('table', 'data', allow_duplicate=True), Output('expected-value', 'children'), Output('expected-pct', 'children')],
    [
        Input("prob", "value"),
        Input("pred-value", "value"),
    ], prevent_initial_call=True
)
def make_graph(y, x):
    # minimal input validation, make sure there's at least one cluster
    if y[1] == pred_data.ProbBuy.max().round(2):
       y[1] = pred_data.ProbBuy.max()
    
    if x[1] == pred_data.PredictedCLV.max().round(2):
       x[1] = pred_data.PredictedCLV.max()

    try:
       plot_data = pred_data[(y[0] <= pred_data.ProbBuy) & (pred_data.ProbBuy <= y[1]) & (x[0] <= pred_data.PredictedCLV) & (pred_data.PredictedCLV <= x[1])]
    except:
       plot_data = pred_data.copy()

    layout = {"xaxis": {"title": "Value to be generated in the next 80 days"}, "yaxis": {"title": "Likelihood of buying in the next 80 days"}}
    fig = px.scatter(
	    plot_data,
            x=plot_data.loc[:, "PredictedCLV"],
            y=plot_data.loc[:, "ProbBuy"],
            color = plot_data.loc[:, "ExpectedValue"],
            color_continuous_scale='Inferno',
            hover_data=["CustomerID", "Recency", "Frequence", "MonetarySum", "MonetaryMean"]
        )

    fig.update_xaxes(title_text = "Value to be generate in the next 80 days")
    fig.update_yaxes(title_text = "Likelihood of buying in the next 80 days")
    fig.update_layout(clickmode='event+select',)
    
    total_expected_value = "$" + str(plot_data.ExpectedValue.sum().round(2))
    pct_expected_value = str(np.round(100 * plot_data.ExpectedValue.sum()/pred_data.ExpectedValue.sum(), 2)) + "%"
    plot_data = plot_data.drop("ExpectedValue", axis=1).round(2)
    
    plot_data.CustomerID = plot_data.CustomerID.astype(int)
    plot_data.CustomerID = plot_data.CustomerID.astype(str)
    return fig, plot_data.to_dict('records'), total_expected_value, pct_expected_value

@app.callback(
    [Output('pred-value', 'value'), Output('prob', 'value')],
    Input('cluster-graph', 'relayoutData'))
def display_relayout_data(relayoutData):

    isAuto = False
    if relayoutData != None:
       for key_ in list(relayoutData.keys()):
           if 'auto' in key_:
              isAuto = True
              break

    if(relayoutData != None and isAuto == False):
       x_values = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
       y_values = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
    else:
       x_values = [0., pred_data.PredictedCLV.max()]
       y_values = [0., pred_data.ProbBuy.max()]
    
    return x_values, y_values


@app.callback(
    Output('table', 'data'),
    [Input('table', 'filter_query')])
def update_table(filter_q):
    data = pred_data.copy()
    if len(data) > 0:
        data.CustomerID = data.CustomerID.astype(int)
        data.CustomerID = data.CustomerID.astype(str)
        data = data.drop("ExpectedValue", axis=1).round(2)

        filtering_expressions = filter_q.split(' && ')
        dff = data.copy()
        dff.CustomerID = dff.CustomerID.astype(int)
        dff.CustomerID = dff.CustomerID.astype(str)
        for filter_part in filtering_expressions:
          col_name, operator, filter_value = split_filter_part(filter_part)
          if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
          elif operator == 'contains':
            filter_value = str(int(filter_value))
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
          elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

        dff = dff.round(2)
        return dff.to_dict('records')
    else:
      data = pred_data.copy()
      data.CustomerID = data.CustomerID.astype(int)
      data.CustomerID = data.CustomerID.astype(str)
      data = data.drop("ExpectedValue", axis=1).round(2)

      return data.to_dict('records')

@app.callback(
    Output('details-table', 'data', allow_duplicate=True),
    [Input('details-table', 'filter_query'),], prevent_initial_call=True)
def update_details_table(filter_q):
    if selected_details_data != {}:
    	data = selected_details_data.copy()
    else:
        data = {}

    if len(data) > 0:
        data.CustomerID = data.CustomerID.astype(int)
        data.CustomerID = data.CustomerID.astype(str)

        filtering_expressions = filter_q.split(' && ')
        dff = data.copy()
        dff.CustomerID = dff.CustomerID.astype(int)
        dff.CustomerID = dff.CustomerID.astype(str)
        for filter_part in filtering_expressions:
          col_name, operator, filter_value = split_filter_part(filter_part)
          if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
          elif operator == 'contains':
            filter_value = str(int(filter_value))
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
          elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

        return dff.to_dict('records')
    else:
      data = selected_details_data.copy()
      return data


@app.callback(
    Output('recommendation-table', 'data', allow_duplicate=True),
    [Input('recommendation-table', 'filter_query'),], prevent_initial_call=True)
def update_recommendation_table(filter_q):
    if selected_recommendation_data != {}:
    	data = selected_recommendation_data.copy()
    else:
        data = {}

    if len(data) > 0:
        data.CustomerID = data.CustomerID.astype(str)

        filtering_expressions = filter_q.split(' && ')
        dff = data.copy()
        dff.CustomerID = dff.CustomerID.astype(str)
        for filter_part in filtering_expressions:
          col_name, operator, filter_value = split_filter_part(filter_part)
          if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
          elif operator == 'contains':
            filter_value = str(int(filter_value))
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
          elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

        return dff.to_dict('records')
    else:
      data = selected_recommendation_data.copy()
      return data



@app.callback(
    Output("modal-describe-feature", "is_open"),
    [Input("open-describe-feature", "n_clicks"), Input("close-describe-feature", "n_clicks")],
    [State("modal-describe-feature", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal-marketing", "is_open"),
    [Input("open-marketing", "n_clicks"), Input("close-marketing", "n_clicks")],
    [State("modal-marketing", "is_open")],
)
def toggle_modal_marketing(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal-details", "is_open"),
    [Input("open-details", "n_clicks"), Input("close-details", "n_clicks")],
    [State("modal-details", "is_open")],
)
def toggle_modal_details(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal-recommendation", "is_open"),
    [Input("open-recommendation", "n_clicks"), Input("close-recommendation", "n_clicks")],
    [State("modal-recommendation", "is_open")],
)
def toggle_modal_recommendation(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [Output('table', 'selected_rows'),],
    [Input('select-all-button', 'value'), Input('table', 'data'),],)
def select_deselect(btn, data):
    df = pd.DataFrame(data)

    if btn == [1]:
       selected_rows = df.index.to_list()
       
    else:
       selected_rows = []
    return [selected_rows]

@app.callback(
    Output("table", "style_data_conditional"),
    Input("table", "selected_rows"),
)
def style_selected_rows(selRows):

    if selRows is None:
        return dash.no_update, res_button
    
    return [{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)',}] + [
        {"if": {"row_index": i}, "backgroundColor": "#0dcaf0",}
        for i in selRows
    ]

@app.callback(
    [Output('display-details-button', 'style'), Output('details-table', 'data')],
    [Input('table', 'selected_rows'), Input('table', 'data')],)
def show_details(selected_rows, data):
    df = pd.DataFrame(data)

    if len(selected_rows) > 0:
      customer_ids = df.iloc[selected_rows]["CustomerID"].to_list()
      customer_ids = [int(id) for id in customer_ids]
      customer_details = original_data[original_data.CustomerID.isin(customer_ids)]
      customer_details.CustomerID = customer_details.CustomerID.astype(str)
      customer_details.loc[:, "InvoiceDate"] = pd.to_datetime(customer_details["InvoiceDate"], dayfirst=True)

      selected_details_data = customer_details.copy()
      return {'display': 'block'}, customer_details.to_dict('records')
    else:
      return {'display': 'none'}, {}

@app.callback(
    Output('recommendation-table', 'data'),
    [Input('table', 'selected_rows'), Input('table', 'data')],)
def show_recommendation(selected_rows, data):
    df = pd.DataFrame(data)
    if len(selected_rows) > 0:
      customer_ids = df.iloc[selected_rows]["CustomerID"].to_list()
      customer_ids = [str(id) for id in customer_ids]

      df = data_recommendation[data_recommendation['CustomerID'].isin(customer_ids)]
      selected_recommendation_data = df.copy()
      return df.to_dict('records')
    else:
      return {}

@app.callback(
    Output("download-clv-prediction", "data"),
    State("table","data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_prediction(data_json, n_clicks):
    df = pd.DataFrame.from_records(data_json)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    filename="CLV-Prediction-" + dt_string + ".xlsx"
    return dcc.send_data_frame(df.to_excel,  filename, index=False, sheet_name="CLV Prediction")


@app.callback(
    Output("download-product-recommendation", "data"),
    State("recommendation-table","data"),
    Input("download-recommendation", "n_clicks"),
    prevent_initial_call=True,
)
def download_recommendation(data_json, n_clicks):
    df = pd.DataFrame.from_records(data_json)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    filename="Product-Recommendation-" + dt_string + ".xlsx"
    return dcc.send_data_frame(df.to_excel,  filename, index=False, sheet_name="Product Recommendation")



if __name__ == "__main__":
    app.run_server(debug=True)
