import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
import plotly.graph_objects as go
import joblib

# Carregar o modelo salvo
with open("prophet.pkl", "rb") as f:
    model_prophet = joblib.load(f)

# Carregar seus dados (ajustar o caminho conforme necessário)
df_filtrado = pd.read_csv('df_filtrado.csv')
df_filtrado['data'] = pd.to_datetime(df_filtrado['data'])  # Certifique-se de que a coluna de data está no formato correto
df_filtrado_prophet = df_filtrado[['data', 'quantidade']].copy()
df_filtrado_prophet.columns = ['ds', 'y']

def plotar_serie(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Série Histórica'))
    fig.update_layout(title='Série Histórica', xaxis_title='Data', yaxis_title='Quantidade')
    return fig

def main():
    st.title('Análise de Série Temporal')
    fig = plotar_serie(df_filtrado_prophet)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
