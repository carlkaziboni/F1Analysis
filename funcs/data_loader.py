from math import isnan
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

BASE_URL = "https://api.openf1.org/v1/"

def fetch_data(endpoint, params=None):
    """
    Fetch data from the OpenF1 API.
    
    Args:
        endpoint (str): The API endpoint to fetch data from.
        params (dict, optional): Additional parameters for the request.
        
    Returns:
        pd.DataFrame or None: The fetched data as a DataFrame, or None if the request failed.
    """
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None
    
@st.cache_data
def fetch_meetings(year, country):
    df = fetch_data("meetings", params={"year": year, "country_name": country})
    if df is None:
        st.error("Failed to fetch meetings data.")
        return pd.DataFrame()
    df['label'] = df['meeting_name'] + " - " + df['location']
    df = df.sort_values(by="meeting_key", ascending=False)

    return df[['meeting_key', 'label', 'year']].drop_duplicates()

@st.cache_data
def fetch_sessions(meeting_key):
    df = fetch_data("sessions", params={"meeting_key": meeting_key})
    if df is None:
        st.error("Failed to fetch sessions data.")
        return pd.DataFrame()
    df['label'] = df['session_name'] + " - " + df['date_start']
    return df[['session_key', 'session_type','label']].drop_duplicates()

@st.cache_data
def fetch_laps(session_key):
    df = fetch_data("laps", params={"session_key": session_key})
    if df is None:
        st.error("Failed to fetch laps data.")
        return pd.DataFrame()
    return df

@st.cache_data
def fetch_drivers(session_key):
    df = fetch_data("drivers", params={"session_key": session_key})
    if df is None:
        st.error("Failed to fetch drivers data.")
        return pd.DataFrame()
    return df

@st.cache_data
def fetch_car_data(session_key, driver_number, lap_number):
    # Get lap timing information first
    df_laps = fetch_laps(session_key)
    if df_laps is None or df_laps.empty:
        st.error("Failed to fetch lap data.")
        return pd.DataFrame()
    
    lap_data = df_laps[(df_laps['driver_number'] == driver_number) & (df_laps['lap_number'] == lap_number)]
    if lap_data.empty:
        st.warning(f"No lap data found for driver {driver_number} on lap {lap_number}.")
        return pd.DataFrame()
    
    # Get lap start time and duration
    start_time_str = lap_data['date_start'].values[0]
    lap_duration = lap_data['lap_duration'].values[0]
    
    if start_time_str is None:
        st.warning(f"No start time available for driver {driver_number} on lap {lap_number}.")
        return pd.DataFrame()
    
    if isnan(float(lap_duration)) or lap_duration is None:
        st.warning(f"No lap duration available for driver {driver_number} on lap {lap_number}.")
        return pd.DataFrame()

    # Try to fetch car data with a time range that's more likely to work with the API
    # Convert to datetime objects for calculation
    start_time = pd.to_datetime(start_time_str)
    end_time = start_time + pd.Timedelta(seconds=lap_duration)
    
    # Format dates for API (try ISO format without microseconds)
    start_api = start_time.strftime('%Y-%m-%dT%H:%M:%S')
    end_api = end_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Try different API query approaches
    urls_to_try = [
        # Try with formatted datetime range
        f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}&date>={start_api}&date<={end_api}",
        # Try with just session and driver (fallback)
        f"https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}"
    ]
    
    for i, url in enumerate(urls_to_try):
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                car_data = pd.DataFrame(response.json())
                
                if car_data.empty:
                    continue  # Try next URL
                
                # If this was the fallback URL (no date filter), filter the data ourselves
                if i == 1:  # Second URL (no date filter)
                    try:
                        car_data['date'] = pd.to_datetime(car_data['date'], format='mixed')
                        car_data = car_data[(car_data['date'] >= start_time) & (car_data['date'] <= end_time)]
                    except Exception as e:
                        st.error(f"Error filtering car data by date: {e}")
                        return pd.DataFrame()
                
                return car_data
                
            elif response.status_code == 504:
                st.warning(f"API timeout (attempt {i+1}). Trying alternative approach...")
                continue
            else:
                st.warning(f"API error {response.status_code} (attempt {i+1})")
                continue
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timeout (attempt {i+1})")
            continue
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            continue
    
    st.error("Failed to fetch car data after trying multiple approaches.")
    return pd.DataFrame()
    