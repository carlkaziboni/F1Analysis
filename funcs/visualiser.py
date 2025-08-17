import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from funcs.data_loader import fetch_car_data
import os

# Return fig of car data for a specific driver, lap and session
def plot_car_data(df, driver_number, lap_number, attribute):
    if df.empty:
        st.warning("No car data available for the selected driver and lap.")
        return
    
    # Filter data for the specific driver and lap
    df_filtered = df[(df['driver_number'] == driver_number)]
    df_filtered = df_filtered[['date', attribute]]

    if df_filtered.empty:
        st.warning(f"No data found for driver {driver_number} on lap {lap_number}.")
        return
    
    # Create a line plot for the specified attribute
    fig = px.line(df_filtered, x='date', y=attribute, title=f'{attribute} for Driver {driver_number} on Lap {lap_number}', labels={'date_start': 'Time', attribute: attribute})
    
    # Update layout for better readability
    fig.update_layout(xaxis_title='Time', yaxis_title=attribute, xaxis=dict(tickformat='%H:%M:%S'), height=600)

    return fig