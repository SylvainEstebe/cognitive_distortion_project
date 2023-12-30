from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

df = pd.read_csv("/Users/sylvainestebe/Code/nlp_project/cognitive_distortion_project/data/corpus_kmean.csv")
df2 = pd.read_csv("/Users/sylvainestebe/Code/nlp_project/cognitive_distortion_project/data/corpus_aglom.csv")
df3 = pd.read_csv("/Users/sylvainestebe/Code/nlp_project/cognitive_distortion_project/data/corpus_hdbscan.csv")

# specify the model
model = 'all-mpnet-base-v2'
model_2 = 'all-MiniLM-L12-v2'
model_3 = 'All-Distilroberta-v1'
models = [model,model_2,model_3]

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("Embedding of Cognitive Distortion"),
        html.P("Select a model:"),
        dcc.RadioItems(
            id="selection",
            options=["K-mean 1","K-mean 2","K-mean 3","Agglomerative 1","Agglomerative 2","Agglomerative 3","HDBSCAN 1","HDBSCAN 2","HDBSCAN 3"],
            value="K-mean 1",
        ),
        dcc.Loading(dcc.Graph(id="graph"), type="cube"),
    ]
)

    
@app.callback(
    Output("graph", "figure"), Input("selection", "value")
)
def display_animated_graph(selection):

    animations = {
        "K-mean 1": px.scatter(
        df,
        x=df[f"{model} x"],
        y=df[f"{model} y"],
        color=  df[f"{model} k-mean"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model} model'
        ),
        "K-mean 2": px.scatter(
        df,
        x=df[f"{model_2} x"],
        y=df[f"{model_2} y"],
        color=  df[f"{model_2} k-mean"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_2} model'
        ),
        "K-mean 3": px.scatter(
        df,
        x=df[f"{model_3} x"],
        y=df[f"{model_3} y"],
        color=  df[f"{model_3} k-mean"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_3} model'
        ),
        "Agglomerative 1": px.scatter(
        df2,
        x=df2[f"{model} x"],
        y=df2[f"{model} y"],
        color=  df2[f"{model} aglo"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model} model'
        ),
        "Agglomerative 2": px.scatter(
        df2,
        x=df2[f"{model_2} x"],
        y=df2[f"{model_2} y"],
        color=  df2[f"{model_2} aglo"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_2} model'
        ),
        "Agglomerative 3": px.scatter(
        df2,
        x=df2[f"{model_3} x"],
        y=df2[f"{model_3} y"],
        color=  df2[f"{model_3} aglo"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_3} model'
        ),
        "HDBSCAN 1": px.scatter(
        df3,
        x=df3[f"{model} x"],
        y=df3[f"{model} y"],
        color=  df3[f"{model} hdbscan_manual"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model} model'
        ),
        "HDBSCAN 2": px.scatter(
        df3,
        x=df3[f"{model_2} x"],
        y=df3[f"{model_2} y"],
        color=  df3[f"{model_2} hdbscan_manual"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_2} model'
        ),
        "HDBSCAN 3": px.scatter(
        df3,
        x=df3[f"{model_3} x"],
        y=df3[f"{model_3} y"],
        color=  df3[f"{model_3} hdbscan_manual"].astype(str),
        hover_data= 'thought',
        title= f'Cluster of negative thought using {model_3} model'
        )
    }
    return animations[selection]


if __name__ == "__main__":
    app.run_server(debug=True)