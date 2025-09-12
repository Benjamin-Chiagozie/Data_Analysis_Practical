import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import statsmodels.api as sm


# Set up the Streamlit app title and description
st.title("Global Climate Explorer")
st.write("Explore global climate data, including CO2 emissions and temperature trends.")

# Data Sourcing & Acquisition and Cleaning & Preprocessing
@st.cache_data
def load_data():
    co2_url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
    co2_df = pd.read_csv(co2_url)

    # Clean and preprocess data (from notebook cell 4SORNwZEvYzH)
    co2_df['population'] = co2_df.groupby('country')['population'].ffill()
    co2_df['gdp'] = co2_df.groupby('country')['gdp'].ffill()

    co2_emission_cols = [col for col in co2_df.columns if 'co2' in col or 'ghg' in col]
    for col in co2_emission_cols:
        co2_df[col] = co2_df[col].fillna(0)

    numerical_cols = co2_df.select_dtypes(include=np.number).columns.tolist()
    cols_to_ffill = [col for col in numerical_cols if col not in ['year', 'population', 'gdp'] + co2_emission_cols]

    for col in cols_to_ffill:
        co2_df[col] = co2_df.groupby('country')[col].ffill()
        co2_df[col] = co2_df.groupby('country')[col].bfill()

    co2_df['year'] = pd.to_datetime(co2_df['year'], format='%Y')

    # Data Integration (from notebook cell c5c44f91 - using dummy data for now)
    # Replace with actual temperature data loading and merging when available
    data = {'country': ['Afghanistan', 'Afghanistan', 'Albania', 'Albania'],
            'year': [datetime(1750, 1, 1), datetime(1751, 1, 1), datetime(1750, 1, 1), datetime(1751, 1, 1)],
            'average_temperature': [15.0, 15.2, 12.0, 12.5]}
    temperature_df = pd.DataFrame(data)
    merged_df = pd.merge(co2_df, temperature_df, on=['country', 'year'], how='outer')


    return merged_df

merged_df = load_data()


# Add interactive widgets to the sidebar
st.sidebar.header("Settings")

# Get a list of unique countries for the selectbox, sorting them and adding 'World' at the beginning
countries = sorted(merged_df['country'].unique().tolist())
countries.insert(0, 'World')
selected_country = st.sidebar.selectbox("Select a Country", countries)

# Get the minimum and maximum year from the dataset
min_year = merged_df['year'].min().year
max_year = merged_df['year'].max().year

# Create a slider for selecting the year range
selected_year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Filter data based on user input
filtered_df = merged_df[
    (merged_df['country'] == selected_country) &
    (merged_df['year'].dt.year >= selected_year_range[0]) &
    (merged_df['year'].dt.year <= selected_year_range[1])
]


st.header("Climate Data Analysis")

# Add storytelling for country-specific analysis
st.markdown(f"""
**Exploring Climate Trends for {selected_country}**

Use the widgets in the sidebar to select a country and a year range to see the climate data and analysis for that specific region.
""")

# Generate dynamic visualizations
if not filtered_df.empty:
    st.subheader(f"Analysis for {selected_country}")

    # Example: Line plot of CO2 emissions over time for the selected country
    st.markdown("""
    **CO2 Emissions Over Time**

    This line chart shows the trend of CO2 emissions for the selected country over the chosen year range. Observe how emissions have changed historically.
    """)
    fig_co2_time = px.line(filtered_df, x='year', y='co2',
                           title=f'CO2 Emissions Over Time for {selected_country}')
    st.plotly_chart(fig_co2_time, use_container_width=True)

    # Example: Scatter plot of CO2 vs GDP for the selected country
    # Filter out rows with NaN in 'co2' or 'gdp' for the scatter plot
    filtered_df_gdp = filtered_df.dropna(subset=['co2', 'gdp'])
    if not filtered_df_gdp.empty:
        st.markdown("""
        **CO2 Emissions vs. GDP**

        This scatter plot illustrates the relationship between CO2 emissions and Gross Domestic Product (GDP) for the selected country. Higher GDP is often associated with higher energy consumption and emissions, but this relationship can vary.
        """)
        fig_co2_gdp = px.scatter(filtered_df_gdp, x='gdp', y='co2',
                                 title=f'CO2 Emissions vs. GDP for {selected_country}')
        st.plotly_chart(fig_co2_gdp, use_container_width=True)

        # Advanced Analytics: Correlation and Regression Analysis
        st.subheader("Advanced Analytics")

        st.markdown("""
        **Correlation and Regression Analysis (CO2 vs. GDP)**

        Below are the results of statistical analyses examining the relationship between CO2 emissions and GDP for the selected country and year range. The correlation matrix shows the strength and direction of the linear relationship, while the regression analysis models how CO2 emissions change with GDP.
        """)

        # Correlation Analysis
        st.write("Correlation Matrix (CO2 vs. GDP):")
        correlation = filtered_df_gdp[['co2', 'gdp']].corr()
        st.write(correlation)

        # Regression Analysis
        st.write("Regression Analysis (CO2 vs. GDP):")
        X = filtered_df_gdp['gdp']
        y = filtered_df_gdp['co2']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.text(model.summary()) # Use st.text to display the summary

    else:
        st.write(f"No data available for CO2 vs. GDP for {selected_country} in the selected year range.")

    # Add more visualizations here based on available data and user selections
    # Example: Time Series Decomposition (only for 'World' as it was done in notebook)
    if selected_country == 'World':
        st.subheader("Time Series Decomposition (Global CO2 Emissions)")
        st.markdown("""
        **Time Series Decomposition of Global CO2 Emissions**

        This decomposition breaks down the global CO2 emissions time series into its underlying components: the overall trend, seasonality (if any), and the remaining residuals. This helps to understand the long-term patterns and any recurring cycles in global emissions.
        """)
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            global_co2_ts = filtered_df.set_index('year')['co2']
            decomposition = seasonal_decompose(global_co2_ts.dropna(), model='additive', period=1)

            fig = make_subplots(rows=4, cols=1, subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'])
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.observed, mode='lines', name='Original'), row=1, col=1)
            fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
            fig.update_layout(height=900, title='Time Series Decomposition of Global CO2 Emissions')
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.write("Statsmodels not installed. Please install it to see time series decomposition (`%pip install statsmodels`).")


else:
    st.write("No data available for the selected country and year range.")

# Example: Global visualizations (these don't depend on the selected country/year range filter above)
st.header("Global Overview")

st.markdown("""
**Global Climate Insights**

Explore visualizations that provide a global perspective on CO2 emissions, highlighting the distribution across countries and the top emitters in a recent year.
""")

# Analyze the distribution of CO2 emissions across countries in a recent year (using the full merged_df)
recent_year_full = merged_df['year'].max()
co2_recent_year_full = merged_df[(merged_df['year'] == recent_year_full) & (merged_df['country'] != 'World')]

if not co2_recent_year_full.empty:
    st.subheader(f"Distribution of CO2 Emissions Across Countries in {recent_year_full.year}")
    st.markdown("""
    This histogram shows how CO2 emissions are distributed among different countries in the most recent year available.
    """)
    fig_hist_global = px.histogram(co2_recent_year_full, x='co2', title=f'Distribution of CO2 Emissions Across Countries in {recent_year_full.year}')
    st.plotly_chart(fig_hist_global, use_container_width=True)

    # Interactive Bar Chart of CO2 Emissions by Country in a Recent Year (Top 20)
    st.subheader(f"Top 20 Countries by CO2 Emissions in {recent_year_full.year}")
    st.markdown("""
    This bar chart highlights the top 20 countries with the highest CO2 emissions in the most recent year.
    """)
    co2_recent_year_sorted_full = co2_recent_year_full.sort_values('co2', ascending=False)
    fig_bar_top20 = px.bar(co2_recent_year_sorted_full.head(20), x='country', y='co2',
                          title=f'Top 20 Countries by CO2 Emissions in {recent_year_full.year}')
    st.plotly_chart(fig_bar_top20, use_container_width=True)

    # Interactive Choropleth Map of CO2 Emissions by Country in a Recent Year
    st.subheader(f"Global CO2 Emissions Map in {recent_year_full.year}")
    st.markdown("""
    This interactive map visualizes CO2 emissions by country in the most recent year, allowing for a geographical comparison of emission levels.
    """)
    co2_recent_year_map_full = co2_recent_year_full.dropna(subset=['iso_code'])
    if not co2_recent_year_map_full.empty:
        fig_map_global = px.choropleth(co2_recent_year_map_full, locations="iso_code",
                                      color="co2",
                                      hover_name="country",
                                      title=f'CO2 Emissions by Country in {recent_year_full.year}',
                                      color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_map_global, use_container_width=True)
    else:
        st.write("No data available with ISO codes for mapping in the most recent year.")

else:
    st.write(f"No data available for the most recent year ({recent_year_full.year}) for global visualizations.")
