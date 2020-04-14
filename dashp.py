import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input,Output, State
import pickle

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app =dash.Dash(__name__,external_stylesheets=external_stylesheets)
house=pd.read_csv("kc_house_data.csv")
pd.set_option('display.max_columns', None)

#Colours
colors = {
    'background': 'white',
    'text': '#7FDBFF',
    'b':'black'
}
data=house

floor=[]
for i in house['floors']:
    floor.append(int(round(i,0)))
house['floor']=floor

def generate_table(dataframe,page_size=10):
    return dash_table.DataTable(
        id="dataTable",
        columns=[{
                "name":i,
                "id":i}
                for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action='native',
        page_current=0,
        page_size=page_size,
    )

def pie_chart(df,groupby,column):
    label=df[groupby].unique()
    values=[]
    for i in label:
        a=df[df[groupby]==i][column].mean()
        values.append(a)
    fig=go.Figure(data=[go.Pie(labels=label,values=values)])
    return(fig)

load_reg=pickle.load(open('King_lm_model.sav', 'rb'))
load_dtree=pickle.load(open('Kingtree_xgb_model.sav', 'rb'))


#copy paste this list from jupyter file
#defines which zipcodes are home to fancier houses with relatively more expensive prices
expensive_zip=[98004, 98005, 98006, 98007, 98008, 98024, 98027, 98029, 98033,
            98039, 98040, 98052, 98053, 98072, 98074, 98075, 98077, 98102,
            98103, 98105, 98107, 98109, 98112, 98115, 98116, 98117, 98119,
            98122, 98136, 98144, 98177, 98199]

from sklearn.preprocessing import StandardScaler
def predict_regression(df):
    #data =pd.DataFrame(data=[[sqft_living,waterfront,view,neigbour,living_ratio,age,renovated,floor,zipcode]],
    #columns=['sqft_living','waterfront','view','Bigger','living_ratio','age','renovated','floor','zipcode'])
    scaler=StandardScaler()
    a=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return load_reg.predict(a)


def predict_dtree(df):
    return load_dtree.predict(df)




app.layout = html.Div(style={'backgroundColor': colors['background']},children =[
    html.H1(children="Final Project : Kings County Housing Data",style={
        'textAlign': 'center',
        'color': colors['text']}),
    html.P("Created by Izzi",style={
        'textAlign': 'center',
        'color': colors['text']}),
    html.Div([
        dcc.Tabs(children =[   
            dcc.Tab(value='tab1',label='Data Frame',children = [
                html.Center(html.H1('Kings County Housing Data')),

                html.Div(children =[
                    html.Div([
                        html.P('Bedrooms  '),
                        dcc.Input(
                        id='Bedrooms',
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Bathrooms  '),
                        dcc.Input(
                            id='Bathrooms',
                            placeholder='Enter a number...',
                            type='number',
                            value=0)]
                    ,className='row col-3'),
                
                
                    html.Div([
                        html.P('Floor  '),
                        dcc.Input(
                        id='Floors',
                        placeholder='Enter a number...',
                        type='number',
                        value=-1)]
                    ,className='row col-3'),

                ],style={ 'display': 'flex','justify-content': 'space-between', 'horizontal-align': 'middle','word-spacing': '4px'}),

                html.Br(),
                html.Br(),
                html.Div(children =[
                        html.Button('search',id = 'filter')
                    ],className = 'col-md-4'),
                html.Br(),    
                html.Div(id = 'div-table', children =[generate_table(house)
            ])
            ]),



            dcc.Tab(value='tab2',label='Insights',children = [
                html.H2("Price vs _____", style={'textAlign': 'center', 'color': '#7FDBFF'}),
                html.Div(
                    dcc.Dropdown(id ='pie-dropdown',
                                options = [{'label': i, 'value': str(i)} for i in house.columns],
                                placeholder='choose a column...'
                                )   
                    ),
                html.Div([
                    dcc.Graph(id='pie-chart',
                            figure=pie_chart(house,'condition','price'))
                ])

            ]),
            dcc.Tab(value='tab3',label='Predictive Model',children=[
                html.H2("Prediction Using Multiple Linear Regression",style={'horizontal-align': 'middle','justify-content': 'center'}),
                html.Br(),
                html.P(' Living Ratio = ratio betewen square ft of house above ground and square ft of house in the basement'),
                html.P('Grade of the house on a scale of 1-13'),
                html.P('Condition is the quality of the house on a scale of 1-5'),
                html.Br(),
                html.Div([
                    html.Div([
                    html.Div([
                        html.P('Living Ratio'),
                        html.Br(),
                        dcc.Input(
                            id='living ratio',
                            style={'width': 50},
                            placeholder='Enter a number...',
                            type='number',
                            value=0)]
                        ,className='row col-3'),
                
                    html.Div([
                        html.P('Square Feet of Building'),
                        html.Br(),
                        dcc.Input(
                        id='Sqft_living',
                        style={'width': 60},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Square Feet Lot:'),
                        html.Br(),
                        dcc.Input(
                        id='sqft-lot',
                        style={'width': 60},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Age             :'),
                        html.Br(),
                        dcc.Input(
                            id='Age',
                            placeholder='Enter a number...',
                            type='number',
                            style={'width': 50},
                            value=0)]
                    ,className='row col-3'),
                ],className='row col-md-4',style={ 'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px',
                                    'display': 'flex','justify-content': 'space-between'}),
            html.Br(),

                html.Div([
                    html.Div([
                        html.P('View '),
                        dcc.Dropdown(value=1,id ='view',
                                options = [{'label': i, 'value': i} for i in range(0,5)]
                                )   
                    ],className='row col-3'),
                    html.Div([
                        html.P('Grade'),
                        dcc.Dropdown(id ='grade',
                                options = [{'label': i, 'value': i} for i in range(1,13)]
                                )   
                    ],className='row col-3'),
                    html.Div([
                        html.P("Condition"),
                        dcc.Dropdown(value=1,id ='condition',
                                options = [{'label': i, 'value': i} for i in range(1,6)]
                                )
                            ],className='row col-3'),
                    html.Div([
                        html.P('Floors             :'),
                        html.Br(),
                        dcc.Input(
                        id='floors',
                        style={'width': 50},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),
                ],className='row col-md-4',style={'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px',
                                    'display': 'flex','justify-content': 'space-between'}),
                html.Br(),
                html.Div([
                    html.Div([
                        html.P("Renovated?"),
                        dcc.Dropdown(value=1,id ='renovated',
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}]
                                )
                    ],className='row col-3'),

                    html.Div([
                        html.P("Waterfront?"),
                        dcc.Dropdown(value=1,id ='waterfront',
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}]
                                )
                    ],className='row col-3'),

                    html.Div([
                        html.P('Bigger ?'),
                        dcc.Dropdown(value=1,id ='neighbour',
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}]
                                )
                    ],className='row col-3'),
                    html.Div([
                    html.P("ZipCode"),
                    dcc.Dropdown(id ='zipcode',
                                options = [{'label': i, 'value': i} for i in house['zipcode'].unique()],
                                placeholder='enter zipcode...',
                                style={'width': 80},
                                )
                    ],className='row col-3')
                ],className='row col-md-4',style = { 'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px'
                                    ,'justify-content': 'space-between'}),
            html.Br(),
            html.Br(),
            html.Div([
            html.H2('Predicted House Price... $',style={'color': colors['text'],'font-size':'36'}),
            html.Div(id='output',children = [
            ],style={'color': colors['text'],'font-size':'36'})               
            ],className='row col-3',style={'horizontal-align': 'middle','justify-content': 'space-between','maxwidth':'1200px',
                                            'text-align': 'center','backgroundColor': colors['b'],'margin':'auto','display':'flex'})
        
            ],style={'horizontal-align': 'middle',
            'font-family': "Arial",
            'borderBottom': '1px solid #f2f2f2',
            'borderLeft': '1px solid #f2f2f2',
            'borderRight':'1px solid #f2f2f2',
            'padding' : '33px'})
        ],style={'background':'white'}),


        dcc.Tab(id='tab4',label='Decision Tree Predictor',children=[
                html.H2("Prediction Using Decision Tree Regressor"),
                html.Br(),
                html.P(' Living Ratio = ratio betewen square ft of house above ground and square ft of house in the basement'),
                html.P('Grade of the house on a scale of 1-13'),
                html.P('Condition is the quality of the house on a scale of 1-5'),
                html.Div([
                    html.Div([
                    html.Div([
                        html.P('Living Ratio'),
                        html.Br(),
                        dcc.Input(
                            id='living ratio-t',
                            style={'width': 50},
                            placeholder='Enter a number...',
                            type='number',
                            value=0)]
                        ,className='row col-3'),
                
                    html.Div([
                        html.P('Square Feet of Building'),
                        html.Br(),
                        dcc.Input(
                        id='Sqft_living-t',
                        style={'width': 60},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Square Feet Lot             :'),
                        html.Br(),
                        dcc.Input(
                        id='sqftlot-t',
                        style={'width': 70},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Age             :'),
                        html.Br(),
                        dcc.Input(
                            id='Age-t',
                            placeholder='Enter a number...',
                            type='number',
                            style={'width': 50},
                            value=0)]
                    ,className='row col-3'),
                ],className='row col-md-4',style={ 'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px',
                                    'display': 'flex','justify-content': 'space-between'}),
            html.Br(),
                html.Div([
                    html.Div([
                        html.P('Bedrooms'),
                        html.Br(),
                        dcc.Input(
                            id='bedroom-t',
                            style={'width': 50},
                            placeholder='Enter a number...',
                            type='number',
                            value=0)]
                        ,className='row col-3'),
                
                    html.Div([
                        html.P('Bathrooms'),
                        html.Br(),
                        dcc.Input(
                        id='bathroom-t',
                        style={'width': 60},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                        html.P('Floors             :'),
                        html.Br(),
                        dcc.Input(
                        id='Floors-t',
                        style={'width': 50},
                        placeholder='Enter a number...',
                        type='number',
                        value=0)]
                    ,className='row col-3'),

                    html.Div([
                    html.P("ZipCode"),
                    dcc.Dropdown(id ='zipcode-t',
                                style={'width': 80},
                                options = [{'label': i, 'value': i} for i in house['zipcode'].unique()],
                                placeholder='enter zipcode...'
                                )
                    ],className='row col-3'),

                ],className='row col-md-4',style={ 'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px',
                                    'display': 'flex','justify-content': 'space-between'}),

                html.Div([
                    html.Div([
                        html.P('View '),
                        dcc.Dropdown(value=1,id ='view-t',
                                options = [{'label': i, 'value': i} for i in range(0,5)]
                                )   
                    ],className='row col-3'),
                    html.Div([
                        html.P('Grade'),
                        dcc.Dropdown(id ='grade-t',
                                options = [{'label': i, 'value': i} for i in range(1,13)]
                                )   
                    ],className='row col-3'),
                    html.Div([
                        html.P("Condition"),
                        dcc.Dropdown(value=1,id ='condition-t',
                                options = [{'label': i, 'value': i} for i in range(1,6)]
                                )
                            ],className='row col-3'),
                    html.Div([
                        html.P("Waterfront?"),
                        dcc.Dropdown(value=1,id ='waterfront-t',
                                style={'width': 50},
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}],
                                placeholder='enter value...'
                                )
                    ],className='row col-3'),
                ],className='row col-md-4',style={'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px',
                                    'display': 'flex','justify-content': 'space-between'}),
                html.Br(),
                html.Div([
                    html.Div([
                        html.P("Renovated?"),
                        dcc.Dropdown(value=1,id ='renovated-t',
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}]
                                )
                    ],className='row col-3'),
                    html.Div([
                        html.P('Bigger ?'),
                        dcc.Dropdown(value=1,id ='neighbour-t',
                                style={'width': 80},
                                options = [{'label': 'yes', 'value': 1},
                                            {'label':'no','value':0}],
                                placeholder = 'enter value..'

                                )
                    ],className='row col-3'),
                ],className='row col-md-4',style = { 'maxwidth':'1200px',
                                    'margin':'auto','horizontal-align': 'middle','word-spacing': '4px'
                                    ,'justify-content': 'space-between'}),
            html.Br(),
            html.Br(),
            html.Div([
            html.H2('Predicted House Price... $',style={'color': colors['text'],'font-size':'36'}),
            html.Div(id='output-t',children = [
            ],style={'color': colors['text'],'font-size':'36'})               
            ],className='row col-3',style={'horizontal-align': 'middle','justify-content': 'space-between','maxwidth':'1200px',
                                            'text-align': 'center','backgroundColor': colors['b'],'margin':'auto','display':'flex'})
        
            ],style={'horizontal-align': 'middle',
            'font-family': "Arial",
            'borderBottom': '1px solid #f2f2f2',
            'borderLeft': '1px solid #f2f2f2',
            'borderRight':'1px solid #f2f2f2',
            'padding' : '33px'})
        ],style={'background':'white'})
    ])
])
])






@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'Bedrooms', component_property = 'value'),
    State(component_id = 'Bathrooms', component_property = 'value'),
    State(component_id = 'Floors', component_property = 'value'),]
)
def update_table(n_clicks, bedroom, bathroom,floor,):
    if bedroom==0 and bathroom==0 and floor==0:
        children = [generate_table(house, page_size = 10)]
    else:
        children = [generate_table(house[(house['bathrooms']==bathroom) &
        (house['bedrooms']==bedroom) & (house['floor']==floor)] , page_size = 10)]            
    return children

@app.callback(
    Output(component_id = 'pie-chart', component_property = 'figure'),
    [Input(component_id = 'pie-dropdown', component_property = 'value')],
)
def update_pie(column):
    try:
        children = pie_chart(house,column,'price')      
        return children
    except:
        children=pie_chart(house,'grade','price')
        return children

@app.callback(
    Output(component_id = 'output', component_property = 'children'),
    #[Input(component_id = 'predict', component_property = 'n_clicks')],
    [Input(component_id = 'Sqft_living', component_property = 'value'),
    Input(component_id = 'sqft-lot', component_property = 'value'),
    Input(component_id = 'living ratio', component_property = 'value'),
    Input(component_id = 'floors', component_property = 'value'),
    Input(component_id = 'Age', component_property = 'value'),
    Input(component_id = 'view', component_property = 'value'),
    Input(component_id = 'condition', component_property = 'value'),
    Input(component_id = 'grade', component_property = 'value'),
    Input(component_id = 'waterfront', component_property = 'value'),
    Input(component_id = 'renovated', component_property = 'value'),
    Input(component_id = 'neighbour', component_property = 'value'),
    Input(component_id = 'zipcode', component_property = 'value')]
)

def predict(sqft_living,sqft_lot,living_ratio,floor,age,view,condition,grade,waterfront,renovated,neighbour,zipcode):
    if zipcode in expensive_zip:
        zipc=1
    else:
        zipc=0
    if sqft_living==0 or pd.isna(living_ratio) or age==0 or pd.isna(zipcode):
        children = [0]
    elif pd.isna(sqft_living) or pd.isna(renovated) or pd.isna(waterfront) or pd.isna(condition) or pd.isna(grade):
        children = [0]
    else:
        score=condition + grade
        a=pd.DataFrame(data=[[sqft_living,sqft_lot,waterfront,view,neighbour,living_ratio,age,renovated,floor,score,zipc]],
        columns=['sqft_living','sqft_lot','waterfront','view','bigger','living_ratio','age','renovated','floor','score','wealthy neighbourhood'])
        children=[predict_regression(a)]
    return children[0]

@app.callback(
    Output(component_id = 'output-t', component_property = 'children'),
    #[Input(component_id = 'predict', component_property = 'n_clicks')],
    [Input(component_id = 'bedroom-t', component_property = 'value'),
    Input(component_id = 'bathroom-t', component_property = 'value'),
    Input(component_id = 'Sqft_living-t', component_property = 'value'),
    Input(component_id = 'sqftlot-t', component_property = 'value'),
    Input(component_id = 'waterfront-t', component_property = 'value'),
    Input(component_id = 'view-t', component_property = 'value'),
    Input(component_id = 'condition-t', component_property = 'value'),
    Input(component_id = 'grade-t', component_property = 'value'),
    Input(component_id = 'neighbour-t', component_property = 'value'),
    Input(component_id = 'living ratio-t', component_property = 'value'),
    Input(component_id = 'Age-t', component_property = 'value'),
    Input(component_id = 'renovated-t', component_property = 'value'),
    Input(component_id = 'Floors-t', component_property = 'value'),
    Input(component_id = 'zipcode-t', component_property = 'value')]
)

def predict(bedroomt,bath,sqftliv,sqftlot,waterfront,view,conditiont,gradet,biggert,livratio,aget,renovated,floort,zipcodet):
    head=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view', 'condition', 'grade', 'bigger', 
        'living ratio', 'age', 'renovated', 'floor', 'wealthy_neighbourhood']
    if zipcodet in expensive_zip:
        zipc=1
    else:
        zipc=0
    if sqftliv==0 or pd.isna(livratio) or aget==0 or pd.isna(zipcodet) or pd.isna(biggert):
        children = [0]
    elif pd.isna(sqftliv) or pd.isna(renovated) or pd.isna(waterfront) or pd.isna(conditiont) or pd.isna(gradet):
        children = [0]
    elif biggert==' ':
        children = [0]
    else:
        a=pd.DataFrame(data=[[bedroomt,bath,sqftliv,sqftlot,waterfront,view,conditiont,gradet,
        biggert,livratio,aget,renovated,floort,zipc]],
        columns=head,
        )
        children=[predict_dtree(a)]
    return children[0]



if __name__=='__main__':
    app.run_server(debug=True)