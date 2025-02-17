# Luis Sanchez / Weather DASH
# v1 - 11-02-25, First Dash - 3 plots (temp, wind, rad)
# v2 - 12-02-25, better format, IDAICE plots (JAN, JUN, AUG, OCT) 
# v3 - 13/02/25, new layout
# v4 - 13-02-25, ready for deploy - problems with deploy
# v5 - 14-02-25, line and box plot for temp
# v6 - 17/02/25, test buttom.  


## TO DO:
# - the moving Average - 4 days is missing for JAN

####################
#### Libraries #####
####################
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

######################
#### FILE IMPORT #####
######################

# Define root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Documents path
documents_path = os.path.join(ROOT_DIR, "Documents")
print(documents_path)
# Define file paths
scenarios_file = os.path.join(documents_path, "Scenarios_CH.csv")
stations_file = os.path.join(documents_path, "Stations_CH.csv")
# Define Weather_files path
weather_files_path = os.path.join(ROOT_DIR, "Weather_files")

# Load CSV files into DataFrames
Scenarios = pd.read_csv(scenarios_file) if os.path.exists(scenarios_file) else None
Stations = pd.read_csv(stations_file) if os.path.exists(stations_file) else None

# Check if DataFrames were loaded successfully
#if Scenarios is None:
    #print("Warning: Scenarios_CH.csv not found in Documents folder.")
#if Stations is None:
    #print("Warning: Stations_CH.csv not found in Documents folder.")

# Display first few rows of each DataFrame if loaded successfully
#if Scenarios is not None:
    #print("Scenarios DataFrame loaded successfully:")
    #print(Scenarios.head())
#if Stations is not None:
    #print("Stations DataFrame loaded successfully:")
    #print(Stations.head(5))

if Scenarios is not None and Stations is not None:
    found_files = []
    missing_files = []
    
    for station in Stations.iloc[:, 0]:
        for scenario in Scenarios.iloc[:, 0]:
            prn_filename = f"{station}_{scenario}.prn"
            prn_filepath = os.path.join(weather_files_path, prn_filename)
            if os.path.exists(prn_filepath):
                found_files.append(prn_filename)
            else:
                missing_files.append(prn_filename)
    
    print("Summary of .prn file search:")
    print(f"Found {len(found_files)} .prn files:")
    #for file in found_files:
        #print(f"  - {file}")
    print(f"Missing {len(missing_files)} .prn files:")
    #for file in missing_files:
        #print(f"  - {file}")

######################
#### Functions   #####
######################

def preprocess_df(df):
        df = df 
        to_drop = [0, #First row
            8761, 8762, 8763, 8764, 8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, #last day (12h)
            8773, 8774, 8775, 8776, 8777, 8778, 8779, 8780, 8781, 8782, 8783, 8784, #last day (+12h)
            8785] #Last row
        df.drop(to_drop, inplace = True)
        start = '2023-01-01 00:00:00'
        end = '2023-12-31 23:00:00'

        # Create the index as a timestamp range
        df.index = pd.date_range(start=start, end=end, freq='H')
        
        # Moving average temperature (4 days)
        df['moving_avg_temp'] = df['TAir'].rolling(window=96, min_periods=1).mean()
        df['Daily TAIR Average'] = df.groupby(df.index.date)['TAir'].transform('mean')
        #print(df)
        return df

# Function to calculate summary statistics
def generate_summary(df):
    if 'TAir' not in df.columns:
        return pd.DataFrame()
    df['Timestamp'] = pd.to_datetime(df.index, errors='coerce')
    df['Month'] = df['Timestamp'].dt.month
    yearly_summary = {
        'Max Value': df['TAir'].max(),
        'Max Time': df['Timestamp'][df['TAir'].idxmax()] if not df['TAir'].isnull().all() else None,
        'Min Value': df['TAir'].min(),
        'Min Time': df['Timestamp'][df['TAir'].idxmin()] if not df['TAir'].isnull().all() else None,
    }
    monthly_summary = df.groupby('Month')['TAir'].agg(['min', 'max', 'mean']).reset_index().round(2)
    return yearly_summary, monthly_summary

# Function to calculate SIA parameters
def generate_analysis(df):
    data = []

    ####-- January --#####
    january_data = df.loc[(df.index.month == 1)]
    # Find the minimum value for 'Moving Avg Temp' in January
    min_tavg_january = january_data['moving_avg_temp'].iloc[96:].min()
    # Find the corresponding timestamp
    min_tavg_timestamp = january_data.loc[january_data['moving_avg_temp'] == min_tavg_january].index[0]
    # Check if min_tavg_timestamp is after noon (hour >= 12)
    if min_tavg_timestamp.hour <= 12:
        min_tavg_timestamp -= timedelta(days=1)

    # Calculate the critical day, starting evaluation, and preparation dates
    starting_evaluation = min_tavg_timestamp.replace(hour=23, minute=59, second=59, microsecond=59) - timedelta(days=3)
    preparation = starting_evaluation.replace(hour=23, minute=59, second=59, microsecond=59) - timedelta(days=14)

    # Create a dictionary with the variable values for January
    result_january = {
        'Month': 'January',
        'Metric': '01 Minimum 96 hours-Moving Average (°C)',
        'Value': round(min_tavg_january, 2)
    }
    data.append(result_january)
    data.append({
        'Month': 'January',
        'Metric': '02 Critical Day',
        'Value': min_tavg_timestamp.strftime("%d-%m-%Y")
    })
    data.append({
        'Month': 'January',
        'Metric': '04 Starting Evaluation',
        'Value': starting_evaluation.strftime("%d-%m-%Y")
    })
    data.append({
        'Month': 'January',
        'Metric': '03 Preparation',
        'Value': preparation.strftime("%d-%m-%Y")
    })
    
    ####-- Jun/Aug/Oct --#####
    months_k = [6, 8, 10]
    for month in months_k:
        month_data = df.loc[(df.index.month == month)]
        # Find the maximum value for 'TAir' in the month
        max_month = month_data['Daily TAIR Average'].max()
        max_month_timestamp = month_data.loc[month_data['Daily TAIR Average'] == max_month].index[0]
        preparation = max_month_timestamp.replace(hour=23, minute=59, second=59, microsecond=59) - timedelta(days=14)

        data.append({
            'Month': pd.Timestamp(month=month, day=1, year=2021).strftime("%B"),
            'Metric': '01 Max TAIR Daily Average (°C)',
            'Value': round(max_month, 2)
        })
        data.append({
            'Month': pd.Timestamp(month=month, day=1, year=2021).strftime("%B"),
            'Metric': '02 Critical Day',
            'Value': max_month_timestamp.strftime("%d-%m-%Y")
        })
        data.append({
            'Month': pd.Timestamp(month=month, day=1, year=2021).strftime("%B"),
            'Metric': '03 Preparation',
            'Value': preparation.strftime("%d-%m-%Y")
        })
    
    # Create a DataFrame
    result_df = pd.DataFrame(data)

    # Define the correct order of months
    month_order = ['January', 'June', 'August', 'October']
    
    # Convert the 'Month' column to a categorical type with the specified order
    result_df['Month'] = pd.Categorical(result_df['Month'], categories=month_order, ordered=True)

    # Sort by Month (now correctly ordered) and Metric
    result_df_sorted = result_df.sort_values(by=['Month', 'Metric']).reset_index(drop=True)

    # Pivot table to get 'Month' and 'Metric' as rows, 'Value' as the final column
    result_df_pivot = result_df_sorted.pivot_table(index=['Month', 'Metric'], values='Value', aggfunc='first').reset_index()

    return result_df_pivot


###########################
#### PLOTS ################
###########################

# Function to create temperature plot
def plot_temperature(df, plot_type='line'):
    df['Month'] = df.index.month

    #print(plot_type)
    fig = go.Figure()

    rename_dict = {'TAir': 'TAir', 'moving_avg_temp': 'MAT'}
    
    # Loop through the columns and add traces
    for col, new_name in rename_dict.items():
        if col in df.columns:
            if plot_type == 'line':
                # Create a line plot
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=new_name))
            elif plot_type == 'box' and col == 'TAir':
                # Create a box plot
                
                for month in range(1, 13):  # Loop through each month
                    month_name = pd.to_datetime(f'2021-{month}-01').strftime('%B')  # Convert to month name
                    monthly_data = df[df['Month'] == month][col]  # Filter data for the month

                    if not monthly_data.empty:
                        fig.add_trace(go.Box(
                            y=monthly_data,
                            name=month_name,  # Use month name as label
                            boxmean=True,  # Show mean line
                            marker_color="#636efa",  # Set box color to light blue
                            fillcolor="#636efa",  # Optional: Fill color for transparency effect
                            line=dict(color="blue")  # Optional: Slightly darker blue outline
                        ))

    fig.update_layout(
        title="Temperature Trends" if plot_type == 'line' else "Temperature Distribution",
        xaxis_title="Time" if plot_type == 'line' else '',
        yaxis_title="Temperature (°C)",
        showlegend= True if plot_type == 'line' else False,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top")  # Moves legend below x-axis
    )
    
    return fig

# Function to create wind direction plot
def plot_wind(df):
    fig = go.Figure()

    if 'WindX' in df.columns and 'WindY' in df.columns:
        # Calculate wind direction (in degrees) and wind speed (magnitude)
        df['WindSpeed'] = (df['WindX']**2 + df['WindY']**2)**0.5
        df['WindDirection'] = (180 / 3.14159) * np.arctan2(df['WindY'], df['WindX'])

        # Create a windrose chart using Scatterpolar
        fig.add_trace(go.Scatterpolar(
            r=df['WindSpeed'],
            theta=df['WindDirection'],
            mode='markers',
            marker=dict(size=8, color=df['WindSpeed'], colorscale='Viridis', showscale=True),
            name='Wind Direction'
        ))

    fig.update_layout(
        title="Wind Direction and Speed",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(df['WindSpeed']) + 1]
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                rotation=90,  # Rotates 0° (N) to the top
                direction="counterclockwise"
            )
        ),
        showlegend=False
    )

    return fig

# Function to create radiation plot
def plot_radiation(df):
    fig = go.Figure()
    
    # Updated names for radiation types
    rename_dict = {'IDirNorm': 'Direct Radiation', 'IDiffHor': 'Diffuse Radiation'}
    
    for col, new_name in rename_dict.items():
        if col in df.columns:
            sorted_values = df[col].sort_values(ascending=False).reset_index(drop=True)
            x_values = list(range(1, len(sorted_values) + 1))   # Rank of each value
            
            fig.add_trace(go.Scatter(x=x_values, y=sorted_values, mode='lines', name=new_name))
    
    fig.update_layout(
        title="Radiation Duration Diagram",
        xaxis_title="Duration (Ranked Values)",
        yaxis_title="Radiation (W/m²)",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top")
    )
    
    return fig

# Function to create humidity plot
def plot_hum(df):
    fig = go.Figure()

    if 'RelHum' in df.columns:
        df['Month'] = df.index.month  # Extract month from the index

        for month in range(1, 13):  # Loop through each month
            month_name = pd.to_datetime(f'2021-{month}-01').strftime('%B')  # Convert to month name
            monthly_data = df[df['Month'] == month]['RelHum']  # Filter data for the month

            if not monthly_data.empty:
                fig.add_trace(go.Box(
                    y=monthly_data,
                    name=month_name,  # Use month name as label
                    boxmean=True,  # Show mean line
                    marker_color="lightblue",  # Set box color to light blue
                    fillcolor="lightblue",  # Optional: Fill color for transparency effect
                    line=dict(color="blue")  # Optional: Slightly darker blue outline
                ))

    fig.update_layout(
        title="Monthly Relative Humidity Distribution",
        xaxis_title="Month",
        yaxis_title="RH (%)",
        showlegend=False,
        xaxis=dict(categoryorder='array', categoryarray=[
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
    )

    return fig

def plot_monthly_TAir(df, summary):
    
    # Ensure the daily averages are unique and indexed by date
    daily_tair_avg = df[['Daily TAIR Average']].resample('D').mean()

    # Calculate the rolling average for the last 4 days
    daily_tair_avg['4-day Avg Daily TAIR'] = daily_tair_avg['Daily TAIR Average'].rolling(window=4, min_periods=1).mean()

    # Add a temporary date column to the original DataFrame
    df['date'] = df.index.floor('D')

    # Merge the 4-day average back into the original dataframe
    df = df.merge(daily_tair_avg[['4-day Avg Daily TAIR']], how='left', left_on='date', right_index=True)

    # Drop the temporary date column
    df.drop(columns=['date'], inplace=True)

    # The new column '4-day Avg Daily TAIR' is now in the original DataFrame
    #print(df.head())

    months_to_plot = ["January", "June", "August", "October"]
    figures = []

    for month in months_to_plot:
        # Extract relevant dates from the summary dataframe for Critical Day
        critical_day = pd.to_datetime(summary.loc[(summary['Month'] == month) & (summary['Metric'] == '02 Critical Day'), 'Value'].values[0], dayfirst=True)
        
        # Extract relevant start_evaluation date for the current year
        start_evaluation = pd.to_datetime(summary.loc[(summary['Month'] == month) & (summary['Metric'] == '03 Preparation'), 'Value'].values[0], dayfirst=True)
        
        # Check if start_evaluation is in December 2022 (we want to reuse December 2023 data)
        if start_evaluation.year == 2022 and start_evaluation.month == 12:
            # Extract December 2023 data
            df_december_2023 = df[(df['Timestamp'].dt.year == 2023) & (df['Timestamp'].dt.month == 12)]
            
            # Change the year of the December 2023 data to 2022
            df_december_2023['Timestamp'] = df_december_2023['Timestamp'].apply(lambda x: x.replace(year=2022))
            
            # Extract January 2023 data
            df_january_2023 = df[(df['Timestamp'].dt.year == 2023) & (df['Timestamp'].dt.month == 1)]
            
            # Concatenate December 2022 data (shifted) and January 2023 data
            df_combined = pd.concat([df_december_2023, df_january_2023])
            
            # Sort the combined data by the Timestamp to ensure proper chronological order
            df_combined = df_combined.sort_values(by='Timestamp')
            
            # Filter data for the desired time period (from start_evaluation to critical_day)
            df_filtered = df_combined[(df_combined['Timestamp'] >= start_evaluation) & (df_combined['Timestamp'] <= critical_day + pd.Timedelta(days=1))]
            
            # Log that we're using December 2023 data for December 2022
            print(f"Using December 2023 data for December 2022 for {month} starting on {start_evaluation}")
        else:
            # Regular case, filter data for the given period
            df_filtered = df[(df['Timestamp'] >= start_evaluation) & (df['Timestamp'] <= critical_day + pd.Timedelta(days=1))]

        # Reshape data for Plotly Express
        variables_to_plot = ["TAir", "moving_avg_temp", "Daily TAIR Average"]  # Default variables
        if month == "January" and "4-day Avg Daily TAIR" in df.columns:
            variables_to_plot.remove("Daily TAIR Average")
            variables_to_plot.append("4-day Avg Daily TAIR")  # Add '4daysavg' for January

        df_long = df_filtered.melt(id_vars=["Timestamp"], 
                                value_vars=variables_to_plot, 
                                var_name="Metric", 
                                value_name="Temperature (°C)")
        
        date_range_label = f"{start_evaluation.strftime('%d-%b')} to {critical_day.strftime('%d-%b')}"
        # Add "*" to the title if it's December
        title_suffix = " *" if start_evaluation.month == 12 else ""


        # Define legend labels
        label_map = {
            "TAir": "TAir", 
            "moving_avg_temp": "MAT", 
            "Daily TAIR Average": "DAT",
            "4-day Avg Daily TAIR": "4D-MAT"  # Add label for 4daysavg
        }

        df_long["Metric"] = df_long["Metric"].map(label_map)

        # Create the plot
        fig = px.line(
            df_long, 
            x="Timestamp", 
            y="Temperature (°C)", 
            color="Metric",  # Assign colors based on temperature type
            title=f'Period for {month}: {date_range_label}{title_suffix}',
            labels={"Timestamp": "Time", "Metric": "Temperature Type"}
        )

        # Improve layout
        fig.update_layout(
            xaxis_title=None,
            xaxis=dict(
                tickformat='%d-%b',  # Format x-axis ticks as (05-JAN)
                dtick = 'D',
                tickangle=310,  # Rotate labels for better readability if needed
            ),
            legend=dict(
                orientation="h",  # Move legend to the bottom
                yanchor="top",
                y=-0.2,  # Adjust position
                xanchor="center",
                x=0.5
            ),
            template="plotly_white"  # Clean background
        )

        figures.append(fig)
    
    return figures


    




######################
#### DASH SET UP   #####
######################

# Dash app setup
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    
    # General Title
    html.H1("Weather File Explorer", style={'textAlign': 'center', 'margin-bottom': '20px'}),
    html.H3(id='selected-file', style={'textAlign': 'center', 'margin-bottom': '20px'}),html.Br(),

    # Page Layout
    html.Div([
        
        # First Section / input and tables
        html.Div([
            html.H2("Weather File", style={'textAlign': 'center', 'margin-bottom': '10px'}),html.Hr(),
            
            # Dropdown menu
            html.Label("Station:"),
            dcc.Dropdown(
                id='station-selector',
                options=[{'label': row[1], 'value': row[0]} for _, row in Stations.iterrows()] if Stations is not None else [],
                value=Stations.iloc[0, 0] if Stations is not None else None
            ),
            #html.Hr(),
            html.Br(),

            # Check box / Radio items
            html.Label("Scenario:"),
            dcc.RadioItems(
                id='scenario-selector',
                options=[{'label': scenario, 'value': scenario} for scenario in Scenarios.iloc[:, 0]] if Scenarios is not None else [],
                value=Scenarios.iloc[0, 0] if Scenarios is not None else None,
                 style={
                        'display': 'flex',
                        'flexDirection': 'column',  # Stacks radio items vertically
                        'gap': '10px'  # Adds space between each radio item
                    }
            ),
            html.Br(),#html.Hr(),html.Hr(),html.Br(),
            html.Button(
                'Update', 
                id='update-button', 
                n_clicks=0, 
                style={
                    'margin-top': '10px',
                    'backgroundColor': '#007BFF',  # Blue color
                    'color': 'white',  # White text
                    'border': 'none',  # No border
                    'padding': '10px 20px',  # Padding for size
                    'borderRadius': '5px',  # Rounded corners
                    'fontSize': '16px',  # Font size
                    'cursor': 'pointer',  # Pointer cursor on hover
                    'transition': 'background-color 0.3s ease',  # Smooth transition
                },
                # Hover effect added using CSS in the Dash app's external stylesheet or inline
            ),
            dcc.Store(id='weather-data-store'),html.Br(),
            html.Br(),
            #html.H3(id='selected-file'),html.Br(),
            html.Hr(),

            html.Div(id='yearly-summary', 
                     style={
                         'whiteSpace': 'pre-line',
                         'fontSize': '20px',
                         'border': '1px solid #ccc',
                         'borderRadius': '5px',
                         'lineHeight': '2',
                         'textAlign': 'center',
                         'backgroundColor': '#b8d3e0',
                         }),
            html.Hr(),

            html.H3("Monthly temperatures:"),
            dash_table.DataTable(
                id='monthly-summary-table',
                style_cell={
                            #'backgroundColor': '#fff',
                            #'color': 'black',
                            'textAlign': 'left',
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'fontSize': '14px',
                            'borderRadius': '5px',
                        },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#f2f2f2',  # Light background for the header
                },
                style_table={'overflowX': 'auto',  'maxWidth': '800px'}),
            html.Hr(),

            html.H3("Simulation Parameters:"),
            dash_table.DataTable(
                id='analysis-table', 
                style_table = {'overflowX': 'auto', 'maxWidth': '800px'},
                style_cell={
                    #'backgroundColor': '#fff',
                    #'color': 'black',
                    'textAlign': 'left',
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'fontSize': '14px',
                    'borderRadius': '5px',
                    'height': 'auto',
                    'whiteSpace': 'normal',
                    'minWidth': '90px',
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': '#f2f2f2',  # Light background for the header
                },
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{Month} = "January"',
                        },
                        'backgroundColor': '#bccde0',  # Gold background for January
                        'color': 'black',
                    },
                    {
                        'if': {
                            'filter_query': '{Month} = "June"',
                        },
                        'backgroundColor': '#9fa8d1',  # Light Blue background for June
                        'color': 'black',
                    },
                    {
                        'if': {
                            'filter_query': '{Month} = "August"',
                        },
                        'backgroundColor': '#bccde0',  # Light Green background for August
                        'color': 'black',
                    },
                    {
                        'if': {
                            'filter_query': '{Month} = "October"',
                        },
                        'backgroundColor': '#9fa8d1',  # Light Gray background for October
                        'color': 'black',
                    },
                    {
                        'if': {
                            'column_id': 'Value' , # Targets the last column
                        },
                        'minWidth': '100px',  # Set minimum width for the last column
                    },
                ],
            ),

        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 
                'border': '2px solid #ccc', 'padding': '15px', 'border-radius': '10px',
                'margin-bottom': '20px', 'background-color': '#f9f9f9', 'margin': '0 auto', }),

        # Second Section / diagrmas 1
        html.Div([
            html.H2("Summary", style={'textAlign': 'center', 'margin-bottom': '10px'}),
            html.Hr(),

            dcc.RadioItems(
                id='temperature-plot-type',
                options=[
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Boxplot', 'value': 'box'}
                ],
                value='line',  # Default selection
                inline=True,  # Display horizontally
                style={'textAlign': 'left', 'margin-bottom': '10px'}
            ),
            dcc.Graph(id='temperature-plot', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            
            dcc.Graph(id='graph-2', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            dcc.Graph(id='radiation-plot', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            dcc.Graph(id='wind-plot', style={'maxWidth': '900px', 'margin': '0 auto'}),
        ], style={'width': '36%', 'display': 'inline-block', 'border': '2px solid #ccc', 'padding': '15px', 'border-radius': '10px',
        'margin-bottom': '20px', 'background-color': '#f9f9f9', 'margin': '0 auto', }),

        # Third Section / diagrmas 2
        html.Div([
            #html.H4("Additional Graphs"),
            html.H2("Periods for Evaluation", style={'textAlign': 'center', 'margin-bottom': '10px'}),
            html.Hr(),html.Br(),html.Br(),
            dcc.Graph(id='graph-3', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            dcc.Graph(id='graph-4', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            dcc.Graph(id='graph-5', style={'maxWidth': '900px', 'margin': '0 auto'}),html.Br(),html.Hr(),
            dcc.Graph(id='graph-6', style={'maxWidth': '900px', 'margin': '0 auto'}),
        ], style={'width': '38%','verticalAlign': 'top',  'display': 'inline-block', 'border': '2px solid #ccc', 'padding': '15px', 'border-radius': '10px',
        'margin-bottom': '20px', 'background-color': '#f9f9f9','margin': '0 auto', 
        }),

        html.Div([
            # Left Section: Contact Information and References
            html.Div([
                #html.P("Footnotes:", style={'marginTop': '10px', 'fontSize': '14px', 'fontWeight': 'bold'}),
                html.P([
                    "Scenarios according to the: ", html.Br(),
                    "Federal Office of Meteorology and Climatology MeteoSwiss", html.Br(),
                    "-----------------------------------", html.Br(),
                    "TAir = Outdoor temperature", html.Br(),
                    "MAT = Moving Average temperature (96h)", html.Br(),
                    "DAT = Daily Average temperature", html.Br(),
                    "4D-MAT = Moving Average temperature (4D)", html.Br(),
                    "-----------------------------------", html.Br(),
                    " * Using 2023 values for December"
                ], style={'fontSize': '14px'})
            ], style={'width': '30%', 'textAlign': 'left'}),

            # Center Section: Empty Space
            html.Div([], style={'width': '40%'}),  # Adjust width as needed

            # Right Section: Footnotes
            
            html.Div([
                
                html.P("Version: 1.0", style={'margin': '5px', 'fontSize': '14px'}),
                html.P("Last Updated: February 2025", style={'margin': '5px', 'fontSize': '14px'}),
                
                html.P("Luis Enrique Sanchez Vazquez", 
                           style={
                            'margin': '5px', 
                            'fontSize': '20px', 
                            'fontFamily': 'Montserrat, sans-serif', 
                            'fontWeight': '600', 
                            'letterSpacing': '1px', 
                            'color': '#333',  # Dark gray for a softer look
                            'textAlign': 'right',
                        }
                    ),
                html.P("Contact: your.email@example.com", style={'margin': '5px', 'fontSize': '14px'}),
                html.P("References:", style={'marginTop': '10px', 'fontSize': '14px', 'fontWeight': 'bold'}),
                html.A(
                    "Federal Office of Meteorology and Climatology MeteoSwiss", 
                    href="https://s.geo.admin.ch/rcykvbphrfkv", 
                    target="_blank", 
                    style={'fontSize': '14px', 'color': '#1E90FF', 'textDecoration': 'none'}
                )
            ], style={'width': '30%', 'textAlign': 'right'}),

        ], style={
            'display': 'flex', 
            'justifyContent': 'space-between', 
            'alignItems': 'center',
            'marginTop': '30px', 
            'padding': '10px',
            'borderTop': '2px solid #ccc', 
            'backgroundColor': '#f9f9f9',
            'fontFamily': 'Arial, sans-serif'
        })

    ])
    
    
])


@app.callback(
    [
        Output('selected-file', 'children'),
        Output('temperature-plot', 'figure'),
        Output('wind-plot', 'figure'),
        Output('radiation-plot', 'figure'),
        Output('yearly-summary', 'children'),
        Output('monthly-summary-table', 'data'),
        Output('monthly-summary-table', 'columns'),
        Output('analysis-table', 'data'),
        Output('analysis-table', 'columns'),
        Output('graph-2', 'figure'),
        Output('graph-3', 'figure'),
        Output('graph-4', 'figure'),
        Output('graph-5', 'figure'),
        Output('graph-6', 'figure'),
       # Output('selected-file-title')
    ],
    Input('update-button', 'n_clicks'),  # Button triggers update
    [
        State('scenario-selector', 'value'),
        State('station-selector', 'value'),
        State('temperature-plot-type', 'value')
    ]
)

######################
#### Functions for   #####
######################  
def update_weather_callback(n_clicks, scenario, station, plot_type):
    if n_clicks == 0:
        #raise dash.exceptions.PreventUpdate  # Prevent update until button is pressed
        return (
            "Please select a Station and Scenario to proceed.", 
            px.scatter(title="Click 'Update' to view the data."),
            px.scatter(title="Click 'Update' to view the data."),
            px.scatter(title="Click 'Update' to view the data."),
            "After selecting a Station and Scenario, click 'Update' to view the data.", [], [], [], [],
            px.scatter(title="Click 'Update' to view the data."), 
            px.scatter(title="Click 'Update' to view the data."),
            px.scatter(title="Click 'Update' to view the data."),
            px.scatter(title="Click 'Update' to view the data."), 
            px.scatter(title="Click 'Update' to view the data.")
        )
    return update_weather_file(scenario, station, plot_type)  

def update_weather_file(scenario, station, plot_type):
    if not scenario or not station:
        return (
            "No valid selection", 
            px.scatter(title="Invalid Input"),
            px.scatter(title="Invalid Input"),
            px.scatter(title="Invalid Input"),
            "", [], [], [], [],
            px.scatter(title="Invalid Input"), 
            px.scatter(title="Invalid Input"),
            px.scatter(title="Invalid Input"),
            px.scatter(title="Invalid Input"), 
            px.scatter(title="Invalid Input")
        )
    
    prn_filename = f"{station}_{scenario}.prn"
    prn_filepath = os.path.join(weather_files_path, prn_filename)
    
    if os.path.exists(prn_filepath):
        try:
            df = pd.read_csv(prn_filepath, na_values=[''], sep = '\t')
            df = preprocess_df(df)



            # Generate plots
            temp_fig = plot_temperature(df, plot_type)
            wind_fig = plot_wind(df)
            rad_fig = plot_radiation(df)
            hum_fig = plot_hum(df) 

            # Generate summaries
            yearly_summary, monthly_summary = generate_summary(df)
            analysis_tab = generate_analysis(df)
            analysis_tab.columns = analysis_tab.columns.astype(str)

            # Monthly temperature plots
            fig_jan, fig_jun, fig_aug, fig_oct = plot_monthly_TAir(df, analysis_tab)

            # Summary and analysis text
            selected_file_text = f"Selected Weather File: \n{station} - {scenario}"
            summary_text = f"Max Temp: {yearly_summary['Max Value']}°C at {yearly_summary['Max Time']}\nMin Temp: {yearly_summary['Min Value']}°C at {yearly_summary['Min Time']}"

            # Convert summaries and analysis to a dictionary format
            columns = [{'name': col, 'id': col} for col in monthly_summary.columns]
            columns_a = [{'name': col, 'id': col} for col in analysis_tab.columns]

            return (
                selected_file_text, 
                temp_fig, 
                wind_fig, 
                rad_fig, 
                summary_text, 
                monthly_summary.to_dict('records'), 
                columns, 
                analysis_tab.to_dict('records'), 
                columns_a, 
                hum_fig, 
                fig_jan, 
                fig_jun, 
                fig_aug, 
                fig_oct
            )

        except Exception as e:
            # If there is any error while reading or processing, return an error message
            return (
                f"Error processing {prn_filename}: {str(e)}", 
                px.scatter(title="Error"),
                px.scatter(title="Error"),
                px.scatter(title="Error"),
                "", [], [], [], [],
                px.scatter(title="Error"), 
                px.scatter(title="Error"),
                px.scatter(title="Error"),
                px.scatter(title="Error"), 
                px.scatter(title="Error")
            )
    else:
        return (
            f"Weather file {prn_filename} not found", 
            px.scatter(title="File Not Found"),
            px.scatter(title="File Not Found"),
            px.scatter(title="File Not Found"),
            "", [], [], [], [],
            px.scatter(title="File Not Found"), 
            px.scatter(title="File Not Found"),
            px.scatter(title="File Not Found"),
            px.scatter(title="File Not Found"), 
            px.scatter(title="File Not Found")
        )


if __name__ == '__main__':
        #app.run_server(debug=True)
        port = int(os.environ.get("PORT", 8050))  # Default to 8050 for local testing
        app.run_server(host="0.0.0.0", port=port)

    