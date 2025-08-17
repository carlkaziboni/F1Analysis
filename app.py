import streamlit as st
from funcs.data_loader import fetch_car_data, fetch_laps, fetch_drivers, fetch_meetings, fetch_sessions, fetch_data
from funcs.visualiser import plot_car_data
from funcs.data_processor import process_lap_data

st.set_page_config(page_title="F1 Car Data Visualizer", layout="wide")

st.title("F1 Car Data Visualizer")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Select Year", options=range(2023, 2026), index=0)
    all_meetings = fetch_data("meetings", params={"year": year})

    if all_meetings.empty:
        st.error("No meetings found for the selected year.")
        st.stop()

    available_countries = sorted(all_meetings['country_name'].dropna().unique())
    selected_country = st.selectbox("Select Country", options=available_countries, index=0)

    filtered_meetings = all_meetings[all_meetings['country_name'] == selected_country].copy()
    filtered_meetings['label'] = filtered_meetings['meeting_name'] + " - " + filtered_meetings['location']
    filtered_meetings = filtered_meetings.sort_values(by="meeting_key", ascending=False)
    if filtered_meetings.empty:
        st.error("No meetings found for the selected country.")
        st.stop()
    
    selected_meeting = st.selectbox("Select Meeting", options=filtered_meetings['label'], index=0)
    selected_meeting_key = filtered_meetings['meeting_key'].loc[filtered_meetings['label'] == selected_meeting].values[0]
    sessions = fetch_sessions(selected_meeting_key)
    if sessions.empty:
        st.error("No sessions found for the selected meeting.")
        st.stop()
    selected_session = st.selectbox("Select Session", options=sessions['label'], index=0)
    selected_session_key = sessions.loc[sessions["label"] == selected_session, "session_key"].values[0]
    selected_session_type = sessions.loc[sessions["label"] == selected_session, "session_type"].values[0]

with col2:

    # Select drivers
    drivers = fetch_drivers(session_key=selected_session_key)
    if drivers.empty:
        st.error("No drivers found for the selected meeting.")
        st.stop()
    
    available_drivers = sorted(drivers['driver_number'].dropna().unique())
    driver_number = st.selectbox("Select Driver Number", options=available_drivers, index=0)

    laps = fetch_laps(selected_session_key)
    if laps.empty:
        st.error("No laps found for the selected session.")
        st.stop()
    
    available_lap_numbers = sorted(laps['lap_number'].dropna().unique())
    lap_number = st.selectbox("Select Lap Number", options=available_lap_numbers, index=0)

st.write(f"Selected Meeting: {selected_meeting} ({selected_session_type})")
st.write(f"Selected Lap Number: {lap_number}")
st.write(f"Selected Driver Number: {driver_number}")
st.write(f"Selected Year: {year}")

# Fetch car data
car_data = fetch_car_data(selected_session_key, driver_number, lap_number)
if car_data.empty:
    st.error("No car data found for the selected parameters.")
else:
    pass
    
# Process and visualize car data
processed_data = car_data.copy()
if processed_data.empty:
    st.error("No processed data found.")
else:
    st.write("Processed Data Retrieved Successfully!")
    fig = plot_car_data(processed_data, driver_number, lap_number, "speed")
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_car_data(processed_data, driver_number, lap_number, "throttle")
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_car_data(processed_data, driver_number, lap_number, "brake")
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_car_data(processed_data, driver_number, lap_number, "n_gear")
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_car_data(processed_data, driver_number, lap_number, "rpm")
    st.plotly_chart(fig, use_container_width=True)

