import utils
import streamlit as st
import visualization
import plotly.express as px

def app():
    # Import dataframe from csv
    file_path = '../../data/airline_delay_dataframe.csv'
    df = utils.import_data(file_path)

    # Assignment introduction
    st.markdown("""
        # Airline delay causes
        The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics (BTS) tracks the on-time performance of domestic flights operated by large air carriers. BTS began collecting details on the causes of flight delays in June 2003. Summary statistics and raw data are madeavailable to the public at the time the Air Travel Consumer Report is released, resulting in the following:
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Arrival Delay (min)", round(df['ArrDelay'].mean(), 2))
    col2.metric("Avg Flight Distance (km)", round(df['Distance'].mean(), 2))
    col3.metric("Avg Air Time (min)", round(df['AirTime'].mean(), 2))
    
    # Displaying the head of the dataframe
    st.markdown("""
        ### Initial look at the dataset
        The dataset consists of 1,247,486 different flights. Each flight consists of 30 columns. Due to the size of the data set, a stratified random sampling method was used to reduce this. The reduced dataset is used in this application to reduce loading time
    """)
    st.dataframe(df[:50])

    # Flight routes visualized in a map
    st.markdown("""
        ### Flight routes
        A diversity of flights can be found in the dataset. These flights mainly take place in and around America. To get a clear picture of this,a visualization of the flights has been made.
    """)
    display = ("Select a month", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    options = list(range(len(display)))
    month = st.selectbox('Month', options, format_func=lambda x: display[x])
    delayed = st.slider('Arrival delay (minutes)', 0, 100)
    file_path = '../data/flight_routes.csv'
    df_flight_routes = utils.import_data(file_path)
    fig1 = visualization.create_flight_map(df_flight_routes, month, delayed)
    st.write(fig1)
    
    # Number of delayed flights
    st.markdown("""
        ### Delayed flights
        Many flights are carried out during the months. The arrival delay of these flights has been recorded. Below is a bar chart showing the average arrival delay per month.
    """)
    type_of_delay = st.selectbox('Type of delay', ['ArrDelay', 'WeatherDelay', 'DepDelay', 'LateAircraftDelay', 'NASDelay', 'CarrierDelay'])
    fig2 = visualization.create_scatter_ArrDelay(df, type_of_delay)
    st.write(fig2)

    # Reliance of air carriers
    st.markdown("""
        ### Reliance of air carriers
        Looking at the different airlines, there is also a difference in delay. The table below shows the average arrival delay per airline. It seems that Southwest airlines, Frontier airlines & Continental airlines are three of the most reliant airlines in terms of flights arriving on time​.
    """)
    fig4 = visualization.unique_airline_hist(df)
    st.write(fig4)

    st.markdown("""
        ### Correlation between variables
        The graph below shows how different variables correlate with each other. A linear regression line is also drawn in the graph.
    """)

    y_axis = st.selectbox('Y axis', ['NASDelay', 'DepDelay', 'ActualElapsedTime', 'AirTime', 'Distance', 'CarrierDelay', 'WeatherDelay', 'SecurityDelay', 'LateAircraftDelay'])
    x_axis = st.selectbox('X axis', ['DepDelay', 'NASDelay', 'ActualElapsedTime', 'AirTime', 'Distance', 'CarrierDelay', 'WeatherDelay', 'SecurityDelay', 'LateAircraftDelay'])

    fig3 = px.scatter(
        x=df[x_axis], y=df[y_axis],
        trendline='ols', trendline_color_override='red',
        title='Linear regression analysis'
    )

    fig3.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
    )
    st.write(fig3)
