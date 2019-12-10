import os
import pickle
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

from .visuals.custom_viz import CustomVisuals
from django_plotly_dash import DjangoDash

env = os.environ['env']
print(f'Running in {env} mode')

if os.environ['env'] == 'payoff':

    X_train = np.load('/code/models/X_train_demo.npy')
    X_test = np.load('/code/models/X_test_demo.npy')
    y_train = np.load('/code/models/y_train_demo.npy')
    y_test = np.load('/code/models/y_test_demo.npy')

    feature_names = ['interactionid', 'interactiondatetime', 'days_between_interaction',
                    'loannumber', 'loanpurposeid', 'loanamount', 'closingdt',
                    'closingcoststoloanamount', 'appraisaltypeid',
                    'appraisedvaluetoloanamount', 'piwflg', 'borrowerpoints',
                    'cashouttoloanamount', 'fico', 'interestrate', 'ltv', 'folderdtid',
                    'todaydtid', 'difffannie30yryield', 'difffannie15yryield',
                    'difffreddie30yryield', 'diffginniemae30yryield', 'diffswap10yryield',
                    'diffswap3yryield', 'diffdowjonesindex', 'diffoilprice',
                    'difftreasury2yryield', 'difftreasury10yryield',
                    'difftreasury30yryield'] 
    target_col = 'PayoffFlag'
    classes = ['not_payoff', 'payoff']
    model = pickle.load(open('/code/models/model_demo.pkl', 'rb'))

else:

    X_train = np.load('/code/models/X_train.npy')
    X_test = np.load('/code/models/X_test.npy')
    y_train = np.load('/code/models/y_train.npy')
    y_test = np.load('/code/models/y_test.npy')
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Class']
    target_col = 'SurvivedFlag'
    classes = ['not_survived', 'survived']
    model = pickle.load(open('/code/models/model.pkl', 'rb'))
    

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

Xdf = pd.DataFrame(X, columns=feature_names)
ydf = pd.DataFrame(y, columns=[target_col])
datadf = pd.concat([Xdf, ydf], axis=1)

cv = CustomVisuals(model=model, feature_names=feature_names, classes=classes, training_data=X_train)

cr_dash_name = 'classification_report'
cr_app = DjangoDash(name=cr_dash_name)
cr_app.layout = html.Div([
    dcc.Graph(
        id = 'classification_report',
       figure =  cv.generate_classification_report(X_test, y_test)
    )
])

cm_dash_name = 'confusion_matrix'
cm_app = DjangoDash(name=cm_dash_name)
cm_app.layout = html.Div([
    dcc.Graph(
        id = cm_dash_name,
       figure =  cv.generate_confusion_matrix(X_test, y_test)
    )
])

pca3d_dash_name = 'pca3d'
pca3d_app = DjangoDash(name=pca3d_dash_name)
pca3d_app.layout = html.Div([
    dcc.Graph(
        id = pca3d_dash_name,
       figure =  cv.generate_pca_3d(X, y), className='embed-responsive'
    )
])

pca2d_dash_name = 'pca2d'
pca2d_app = DjangoDash(name=pca2d_dash_name)
pca2d_app.layout = html.Div([
    dcc.Graph(
        id = pca2d_dash_name,
       figure =  cv.generate_pca_2d(X, y), className='embed-responsive'
    )
])

rank2d_dash_name = 'rank2d'
rank2d_app = DjangoDash(name=rank2d_dash_name)
rank2d_app.layout = html.Div([
    dcc.Graph(
        id = rank2d_dash_name,
       figure =  cv.generate_rank_2d(X), className='embed-responsive'
    )
])

rank1d_dash_name = 'rank1d'
rank1d_app = DjangoDash(name=rank1d_dash_name)
rank1d_app.layout = html.Div([
    dcc.Graph(
        id = rank1d_dash_name,
       figure =  cv.generate_rank_1d(X, y), className='embed-responsive'
    )
])

radviz_dash_name = 'radviz'
radviz_app = DjangoDash(name=radviz_dash_name)
radviz_app.layout = html.Div([
    dcc.Graph(
        id = radviz_dash_name,
       figure =  cv.generate_rad_viz(X, y), className='embed-responsive'
    )
])

rocauc_dash_name = 'rocauc'
rocauc_app = DjangoDash(name=rocauc_dash_name)
rocauc_app.layout = html.Div([
    dcc.Graph(
        id = radviz_dash_name,
        figure =  cv.generate_roc_auc(X_train, y_train,
                                     X_test, y_test), 
        className='embed-responsive'
    )
])

prob_dash_name = 'probability'
prob_app = DjangoDash(name=prob_dash_name)
prob_app.layout = html.Div([
    dcc.Graph(
        id = prob_dash_name,
        figure =  cv.generate_probability_chart(X_test[0]), 
        className='embed-responsive'
    )
])


feati_dash_name = 'feature_imp'
feati_app = DjangoDash(name=feati_dash_name)
feati_app.layout = html.Div([
    dcc.Graph(
        id = feati_dash_name,
        figure =  cv.local_feature_importance(data=X_test[0]), 
        className='embed-responsive'
    ),
])


featv_dash_name = 'feature_val'
featv_app = DjangoDash(name=featv_dash_name)
featv_app.layout = html.Div([
    dcc.Graph(
        id = featv_dash_name,
        figure =  cv.feature_values(X_test[0]), 
        className='embed-responsive'
    )
])

data_table_dash_name = 'data_table'
data_table_app = DjangoDash(name=data_table_dash_name)
data_table_app.layout = html.Div([
    dash_table.DataTable(
        id=data_table_dash_name,
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in datadf.columns
            # omit the id column
            if i != 'id'
        ],
        data = datadf.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_selectable='single',
        row_deletable=False,
        selected_rows=[],
        page_action='native',
        page_current= 0,
        page_size= 10,
        virtualization=True,
    ),
    html.Div(id='data_table-container')
])


def filter_data(selected_row_indices):
    dff = datadf.iloc[selected_row_indices]
    dff.columns = datadf.columns
    return dff

@data_table_app.callback(
    dash.dependencies.Output('data_table-container', 'children'),
    [dash.dependencies.Input(data_table_dash_name, 'selected_rows')])
def update_download_link(selected_row_indices):
    dff = filter_data(selected_row_indices)
    return html.Div([
        html.Div([
            html.Div([
                html.H3('Model Probability', className='title-2'),
                dcc.Graph(
                    id = prob_dash_name,
                    figure =  cv.generate_probability_chart(dff[feature_names].values[0]), 
                    className='embed-responsive'
                ),
            ], className='col-sm-12 col-md-6'),
            html.Br(),
            html.Div([
                html.H3('Local Feature Importance', className='title-2'),
                dcc.Graph(
                    id = 'feature_imp',
                    figure = cv.local_feature_importance(data=dff[feature_names].values[0]), 
                    className='embed-responsive'
                )
            ], className='col-sm-12 col-md-6')
            
        ], className='row')
    ], className='container-fluid')




box_plot_data_table_name = 'box_plot_data_table'
box_plot_data_table_app = DjangoDash(name=box_plot_data_table_name)
box_plot_data_table_app.layout = html.Div([
    dash_table.DataTable(
        id=box_plot_data_table_name,
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in datadf.columns
            # omit the id column
            if i != 'id'
        ],
        data = datadf.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_selectable='single',
        row_deletable=False,
        selected_columns=[],
        column_selectable='single',
        page_action='native',
        page_current= 0,
        page_size= 10,
        virtualization=True,
    ),
    html.Div(id='box_plot_data_table-container')
])

def test(data):
    return data

@box_plot_data_table_app.callback(
    dash.dependencies.Output('box_plot_data_table-container', 'children'),
    [dash.dependencies.Input(box_plot_data_table_name, 'active_cell')])
def update_active_cell(active_cell):
    datadfx = test(datadf)
    # active_cell returns in form {'row': 6, 'column': 4, 'column_id': 'Parch'}
    column_values = datadfx[active_cell['column_id']]
    point = datadfx.iloc[active_cell['row'], active_cell['column']] 
    
    # fig = html.Div([
    #     html.H3('Box Plot', className='title-2'),
    #     dcc.Graph(
    #         id = box_plot_data_table_name,
    #         figure = cv.box_plot(data=column_values, point=point), 
    #         className='embed-responsive'
    #     )
    # ], className='container-fluid')

    fig = html.Div([
        html.H3('Summary Statistics', className='title-2'),
        dcc.Graph(
            id='box_plot_data_table-container',
            figure=cv.summary_statistics(data=pd.DataFrame(column_values, 
                                         columns=[active_cell['column_id']]), 
                                         point=point),
            className='embed-responsive'
        )
    ])

    column_values, point = None, None
    return fig
