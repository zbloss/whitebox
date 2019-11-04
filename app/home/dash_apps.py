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

# Loading model attributes
print('Loading model...')
model = pickle.load(open('../models/model.pkl', 'rb'))
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Class']
classes = ['not_survived', 'survived']

# Loading train_test data
print('Loading data...')
X_train = np.load('../models/X_train.npy')
X_test = np.load('../models/X_test.npy')
y_train = np.load('../models/y_train.npy')
y_test = np.load('../models/y_test.npy')

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

Xdf = pd.DataFrame(X, columns=feature_names)
ydf = pd.DataFrame(y, columns=['survived'])
data = pd.concat([Xdf, ydf], axis=1)


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
    )
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
        id=data_table_dash_name,  #'datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in data.columns
            # omit the id column
            if i != 'id'
        ],
        data=data.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_selectable='multi',
        row_deletable=True,
        selected_rows=[],
        page_action='native',
        page_current= 0,
        page_size= 10,
    ),
    html.Div(id='data_table-container')
])


@data_table_app.callback(
    Output('data_table-container', 'children'),
    [Input('data_table', 'derived_virtual_row_ids'),
     Input('data_table', 'selected_row_ids'),
     Input('data_table', 'active_cell')])
def update_graphs(row_ids, selected_row_ids, active_cell):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dff = data
        # pandas Series works enough like a list for this to be OK
        row_ids = data['id']
    else:
        dff = data.loc[row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None

    colors = ['#FF69B4' if id == active_row_id
              else '#7FDBFF' if id in selected_id_set
              else '#0074D9'
              for id in row_ids]

    return [
        dcc.Graph(
            id=column + '--row-ids',
            figure={
                'data': [
                    {
                        'x': dff['country'],
                        'y': dff[column],
                        'type': 'bar',
                        'marker': {'color': colors},
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': column}
                    },
                    'height': 250,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ['pop', 'lifeExp', 'gdpPercap'] if column in dff
    ]