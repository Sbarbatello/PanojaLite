# lib_streamlit.py
from config import config
import streamlit as st
import altair as alt
import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime, timedelta

from libraries import lib_db, lib_api

####################################################################################################

def graph_matrix_sizing(fix_rows, fix_cols, charts, preview_mode=False, num_charts=0):
    """
    Determines the graph matrix layout and displays the provided charts.

    Args:
        fix_rows (int): Number of fixed rows (0 for default).
        fix_cols (int): Number of fixed columns (0 for default).
        charts (list of tuples): A list of tuples where each tuple contains:
            - The title of the chart (str).
            - A callable function that creates and renders the chart.
        preview_mode (bool): If True, only calculates and returns the layout size.
        num_charts (int): The number of charts to consider when in preview mode.

    If neither fix_rows nor fix_cols is set, calculates the layout to fit all charts.
    """

    if preview_mode:  # If in preview mode
        if num_charts == 0:  # If no charts are passed, return
            return None  # Return None to avoid warning message if no charts are available yet
        num_charts_preview = num_charts  # Use num_charts to calculate layout in preview mode

    num_charts = len(charts)  # Total number of charts (use this in display mode)
    warning_message = None  # Initialize the warning message

    # Determine the layout size
    if fix_rows == 0 and fix_cols == 0:  # If both are not set
        # Default to a square-ish layout to fit all charts
        num_cols = int(num_charts**0.5) or 1
        num_rows = (num_charts + num_cols - 1) // num_cols
        if preview_mode:  # Use num_charts_preview in preview mode
            warning_message = f"Graph matrix size not set. Automatically calculated to {num_rows}x{num_cols} based on {num_charts_preview} charts."
        else:
            warning_message = f"Graph matrix size not set. Automatically calculated to {num_rows}x{num_cols}."
    elif fix_rows == 0:  # If rows are not set, calculate based on fixed columns
        num_cols = fix_cols
        num_rows = (num_charts + num_cols - 1) // num_cols
    elif fix_cols == 0:  # If columns are not set, calculate based on fixed rows
        num_rows = fix_rows
        num_cols = (num_charts + num_rows - 1) // num_rows
    else:  # If both are set, use the provided values
        num_rows = fix_rows
        num_cols = fix_cols

    if preview_mode:
        return warning_message  # Return the warning message (or None)

    # Create the layout (only in display mode)
    rows = []
    for _ in range(num_rows):
        row = st.columns(num_cols)
        rows.append(row)

    # Display the charts in the layout (only in display mode)
    for idx, (title, chart_fn) in enumerate(charts):  # Unpack title and chart function
        row_idx = idx // num_cols  # Determine the row index
        col_idx = idx % num_cols  # Determine the column index
        if row_idx < len(rows):  # Ensure we don't exceed the number of rows
            with rows[row_idx][col_idx]:
                st.subheader(title)  # Display the chart's title
                chart_fn()  # Call the chart function to render the chart

    # Return the warning message if needed (in display mode)
    return warning_message

####################################################################################################

####################################################################################################

def render_stacked_area_chart(df, title, height=400, asset_order=None):
    """Creates and renders a stacked area chart with controlled stacking order."""

    # Ensure valuation_date is datetime
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])

    df_full = fill_missing_asset_values(df)

    if asset_order is not None:
        # 1. Convert asset_id to categorical with specified order
        df_full['asset_id'] = pd.Categorical(df_full['asset_id'], categories=asset_order, ordered=True)

        # 2. Sort the DataFrame by the ordered categorical asset_id
        df_full = df_full.sort_values('asset_id')

    # Create asset_id_name for display (legend, tooltips) - after sorting
    df_full['asset_id_name'] = df_full['asset_id'].astype(str) + ". " + df_full['asset_name']

    # 3. Create a rank column to explicitly control stacking order
    if asset_order is not None:
        df_full['rank'] = df_full['asset_id'].cat.codes # Assign numerical ranks based on categorical order

    chart = alt.Chart(df_full).mark_area().encode(
        x=alt.X('valuation_date:T', title='Date'),
        y=alt.Y('sum(asset_value):Q', title='Asset Value', stack='zero'), # Essential for correct stacking
        color=alt.Color('asset_id_name:N', title='Asset ID & Name', sort=None, scale=alt.Scale(scheme='tableau20')),
        order=alt.Order('rank:Q'),  # Use the rank to control stacking
        tooltip=[
            'asset_id:N',
            'asset_name:N',
            'valuation_date:T',
            alt.Tooltip('sum(asset_value):Q', title='Asset Value', format='.2f')
        ]
    ).properties(
        title=title,
        height=height
    ).configure_view(
        stroke=None
    )

    st.altair_chart(chart, theme=None, use_container_width=True)

####################################################################################################

def fill_missing_asset_values(df):
    """
    Fills missing asset values by creating a full date range for each asset
    and forward-filling missing values based on the most recent valuation date.
    
    Args:
        df (pd.DataFrame): The input DataFrame with 'asset_id', 'valuation_date', and asset values.
    
    Returns:
        pd.DataFrame: A DataFrame with filled asset values, sorted by 'valuation_date' and 'asset_id'.
    """
    ########## 1 ##########
    # Sort the original DataFrame by asset_id and valuation_date
    df_sorted = df.sort_values(by=['valuation_date', 'asset_id'])
    # print('\n', 10*'#','1',10*'#')
    # print(df)

    ########## 2 ##########
    # Generate a full date range from the minimum to the maximum valuation_date
    full_date_range = pd.date_range(df['valuation_date'].min(), df['valuation_date'].max())
    # print('\n', 10*'#','2',10*'#')
    # print(full_date_range)
    
    ########## 3 ##########
    # Step 3: Generate all asset-date combinations
    all_assets = df['asset_id'].unique()
    df_full = pd.DataFrame([(asset, date) for asset in all_assets for date in full_date_range],
                           columns=['asset_id', 'valuation_date'])
    # print('\n', 10*'#','3',10*'#')
    # print(all_assets)

    ########## 4 ##########
    # Step 4: Merge with the original DataFrame
    df_full = df_full.merge(df_sorted, on=['asset_id', 'valuation_date'], how='left') 
    # print('\n', 10*'#','4',10*'#')
    # print(df_full)

    ########## 5 ##########
    # Step 5: Build the dictionary of minimum valuation dates
    dict_min_valuation_dates = df.groupby('asset_id')['valuation_date'].min().to_dict()
    # print('\n', 10*'#','5',10*'#')
    # print(dict_min_valuation_dates)

    ########## 6 ##########
    # Step 6: Filter out rows earlier than the minimum valuation date for each asset
    df_full = df_full[
        df_full.apply(lambda row: row['valuation_date'] >= dict_min_valuation_dates[row['asset_id']], axis=1)
    ]
    # print('\n', 10*'#','6',10*'#')
    # print(df_full)

    ########## 7 ##########
    # Step 7: Sort the DataFrame
    df_full = df_full.sort_values(by=['asset_id', 'valuation_date'])
    # print('\n', 10*'#','7',10*'#')
    # print(df_full)

    ########## 8 ##########
    # Step 8: Forward-fill missing rows for each asset
    def forward_fill_rows(group):
        # Forward-fill all columns except 'asset_id' and 'valuation_date'
        return group.ffill()

    df_full = df_full.groupby('asset_id', group_keys=False).apply(forward_fill_rows)
    # print('\n', 10*'#','8',10*'#')
    # print(df_full)

    return df_full

####################################################################################################

def render_monthly_change_heatmap_old(df, title, height=400):
    """Creates and renders a heatmap of monthly percentage change in asset balance."""

    df_full = normalise_asset_df(df)
    df_monthly = create_monthly_summary(df_full)

    # Calculate monthly changes *after* filling missing values
    df_monthly = df_monthly.sort_values(['asset_id', 'valuation_date'])  # Sort for correct calculations
    df_monthly['monthly_change'] = df_monthly.groupby('asset_id')['asset_value'].pct_change() * 100

    # Handle potential infinite values (e.g., first month)
    df_monthly.replace([float('inf'), -float('inf')], None, inplace=True)  # In-place replacement
    df_monthly.fillna(0, inplace=True)

    # Define a custom diverging color scale
    background_color = st.get_option("theme.backgroundColor")

    # 1. Create a separate column for clipped values (for the chart)
    df_monthly['clipped_change'] = np.clip(df_monthly['monthly_change'], -200, 200)

    # 2. Define custom sequential color scale
    color_scale = alt.Scale(
        domain=[-200, 200],  # Single domain for sequential scale
        range=['#be2e33', '#21e90b']  # Red to Green
    )

    heatmap = alt.Chart(df_monthly).mark_rect().encode(
        x=alt.X('yearmonth(valuation_date):T', title='Month-Year'),
        y=alt.Y('asset_id:N', title='asset_id'),
        color=alt.condition(
            alt.datum.clipped_change == 0, # Handle 0 values separately
            alt.value(background_color),
            alt.Color('clipped_change:Q', title='Monthly Change (%)', scale=color_scale)
        ),
        tooltip=[  # Use original values for tooltip
            alt.Tooltip('yearmonth(valuation_date):T', title='Month-Year', format='%Y-%m'),
            alt.Tooltip('asset_id:N', title='asset_id'),
            alt.Tooltip('monthly_change:Q', title='Monthly Change (%)', format='.2f')
        ]
    ).properties(
        title=title,
        height=height
    )

    st.altair_chart(heatmap, theme=None, use_container_width=True)
    
    # print(tabulate(df_monthly, headers='keys', tablefmt='psql'))

####################################################################################################

def render_monthly_change_heatmap(df, title, height=400):
    """Creates and renders a heatmap of monthly percentage change using Seaborn."""

    df_full = normalise_asset_df(df)
    df_monthly = create_monthly_summary(df_full)

    # Calculate monthly changes *after* filling missing values
    df_monthly = df_monthly.sort_values(['asset_id', 'valuation_date'])
    df_monthly['monthly_change'] = df_monthly.groupby('asset_id')['asset_value'].pct_change() * 100

    # Handle potential infinite values (e.g., first month)
    df_monthly.replace([float('inf'), -float('inf')], None, inplace=True)
    df_monthly.fillna(0, inplace=True)

    # Convert 'valuation_date' to datetime
    df_monthly['valuation_date'] = pd.to_datetime(df_monthly['valuation_date'])

    # Create a 'month' column as strings (YYYY-MM)
    df_monthly['month'] = df_monthly['valuation_date'].dt.strftime('%Y-%m')

    # Aggregate before pivoting (crucial!)
    df_grouped = df_monthly.groupby(['asset_id', 'month'])['monthly_change'].mean().reset_index()

    # Pivot the aggregated data
    df_pivot = df_grouped.pivot(index='asset_id', columns='month', values='monthly_change')
    # Clip the data directly in df_pivot
    df_pivot_clipped = df_pivot.clip(lower=-100, upper=100)

    num_rows = df_pivot.shape[0]
    fig_height = max(height/300, num_rows * 0.2)  # Adjust 0.2 as needed

    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Create masks for clipped values
    mask_over_100 = df_pivot > 100
    mask_under_minus_100 = df_pivot < -100

    min_change = df_pivot_clipped.min().min()
    max_change = df_pivot_clipped.max().max()

    vmin = min_change  # Red scale goes from min to 0
    vmax = max_change  # Green scale goes from 0 to max

    cmap1 = plt.get_cmap('Greens').copy()
    cmap1.set_under('none')

    cmap2 = plt.get_cmap('Reds_r').copy() #Inverted
    cmap2.set_over('none')

    # Create a mask where True means "mask this value"
    mask = df_pivot_clipped == 0

    sns.heatmap(df_pivot_clipped, vmin=0, vmax=vmax, cmap=cmap1, cbar_kws={'pad': -0.02}, ax=ax, mask=mask)
    sns.heatmap(df_pivot_clipped, vmin=vmin, vmax=0, cmap=cmap2, cbar_kws={'pad': 0.02}, ax=ax, mask=mask)

    # Plot clipped values with a different color
    sns.heatmap(df_pivot, mask=~mask_over_100, vmax=100.1, vmin=100, cmap=colors.ListedColormap(["#66ff00"]), cbar=False, ax=ax)  # Bright Green
    sns.heatmap(df_pivot, mask=~mask_under_minus_100, vmin=-100.1, vmax=-100, cmap=colors.ListedColormap(["#FF000C"]), cbar=False, ax=ax)  # Bright Red

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Asset ID")

    # Rotate y-axis labels to be horizontal
    plt.yticks(rotation=0)  # or ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    st.pyplot(fig)  # If using Streamlit

####################################################################################################

def render_chart_with_df(df, title, height=400):
    """
    Renders a chart and displays the DataFrame underneath.
    """
    # Render the chart
    render_monthly_change_heatmap(df, title, height)
    
    # Display the DataFrame
    st.write(f"Data for {title}:")
    st.dataframe(df)  # Use st.dataframe to render a scrollable table

####################################################################################################

def style_preview_dataframe(df, max_valuation_date):
    """Styles the preview DataFrame based on valuation_date."""
    if df is None or df.empty:
        st.warning("No data to preview.")
        return

    try:
        # Convert valuation_date to datetime64
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        # Sort the DataFrame by valuation_date in descending order
        df = df.sort_values(by='valuation_date', ascending=False)
        # Convert max_valuation_date to pandas.Timestamp
        max_valuation_date = pd.Timestamp(max_valuation_date)

        # Apply styling based on the comparison
        def color_rows(row):
            if pd.isna(row['valuation_date']):
                return ['background-color: gray'] * len(row)  # Handle null values
            elif pd.to_datetime(row['valuation_date']) <= max_valuation_date:
                return ['background-color: #e37575'] * len(row)
            else:
                return ['background-color: #38A169'] * len(row)

        styled_df = df.style.apply(color_rows, axis=1)
        styled_df.format({"current_value": "{:,.2f}"})
        st.dataframe(styled_df)

    except Exception as e:
        st.error(f"Error styling preview: {e}")
        st.dataframe(df) #show the df with no styling in case of error.

####################################################################################################

def render_plotly_stacked_area_chart(df, title, height=400, asset_order=None):
    """Creates and renders a stacked area chart using Plotly with correct colors (using example's approach)."""

    # Ensure valuation_date is datetime
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])

    df_full = fill_missing_asset_values(df)

    if asset_order is not None:
        # 1. Convert asset_id to categorical with specified order
        df_full['asset_id'] = pd.Categorical(df_full['asset_id'], categories=asset_order, ordered=True)

        # 2. Sort the DataFrame by the ordered categorical asset_id
        df_full = df_full.sort_values('asset_id')

    # Create display_name for legend and tooltips
    df_full['display_name'] = df_full['asset_label']

    fig = go.Figure()

    # Get a list of distinct colors (e.g., from Plotly's default color sequence)
    set3_colors = px.colors.qualitative.Set3
    additional_colors = [
    'rgb(230, 190, 255)',  # Light purple
    'rgb(166, 206, 227)',  # Light blue
    'rgb(253, 220, 150)',  # Light orange
    'rgb(227, 251, 170)',  # Light green
    'rgb(255, 215, 240)',  # Light pink
    'rgb(220, 220, 220)',  # Light gray
    'rgb(210, 170, 220)',  # Light violet
    'rgb(210, 240, 210)'   # Light mint
    ]
    extended_colors = set3_colors + additional_colors

    for i, display_name in enumerate(df_full['display_name'].unique()):
        asset_data = df_full[df_full['display_name'] == display_name]
        color = extended_colors[i % len(extended_colors)]

        fig.add_trace(go.Scatter(
            x=asset_data['valuation_date'],
            y=asset_data['asset_value'],
            hoverinfo='x+text',
            mode='lines',
            line=dict(width=0.5, color=color),
            stackgroup='one',
            name=display_name,
            text=[f"Asset: {row['asset_label']}<br>Value: {row['asset_value']:.2f}" for _, row in asset_data.iterrows()],
        ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title='Date',
        yaxis_title='Asset Value',
        hovermode='x unified',
        legend_title='Asset',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0), # Change Legend location
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

####################################################################################################

def render_monthly_net_worth_waterfall(df, title, height=600):
    """
    Plots a Waterfall Chart for Monthly Changes in Net Worth using Plotly.

    Args:
        df (pd.DataFrame): DataFrame with 'valuation_date' and 'asset_value' columns.
        title (str): Title of the chart.
        height (int): Height of the chart.
    """

    # Ensure valuation_date is datetime
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])

    # Group by month and sum asset values
    df['month'] = df['valuation_date'].dt.to_period('M')
    monthly_net_worth = df.groupby('month')['asset_value'].sum().reset_index()

    # Calculate monthly changes
    monthly_net_worth['previous_net_worth'] = monthly_net_worth['asset_value'].shift(1, fill_value=monthly_net_worth['asset_value'].iloc[0])
    monthly_net_worth['change'] = monthly_net_worth['asset_value'] - monthly_net_worth['previous_net_worth']

    # Create Waterfall Chart
    fig = go.Figure(go.Waterfall(
        name="Net Worth Change",
        orientation="v",
        x=monthly_net_worth['month'].astype(str),
        y=monthly_net_worth['change'],
        textposition="outside",
        text=[f"{change:.2f}" for change in monthly_net_worth['change']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgb(0, 150, 0)"}},
        decreasing={"marker": {"color": "rgb(255, 0, 0)"}},
        totals={"marker": {"color": "rgb(63, 63, 63)"}}
    ))

    fig.update_layout(
        title=title,
        height=height,
        yaxis_title="Net Worth Change",
        xaxis_title="Month",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

####################################################################################################

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

def normalise_asset_df(df):
    """Normalises the DataFrame to have one entry per asset per day,
       forward-filling missing values.
    """
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])  # Ensure datetime

    # 1. Create Full Date Range:  Use ALL days between min and max date.
    min_date = df['valuation_date'].min().floor('D') #Floor to the start of the day
    max_date = df['valuation_date'].max().ceil('D') #Ceil to the end of the day
    full_date_range = pd.date_range(min_date, max_date, freq='D')

    # 2. Create All Asset-Date Combinations:
    all_assets = df['asset_id'].unique()
    df_normalised = pd.DataFrame([(asset, date) for asset in all_assets for date in full_date_range],
                                columns=['asset_id', 'valuation_date'])

    # 3. Merge with Original Data:
    df_normalised = df_normalised.merge(df, on=['asset_id', 'valuation_date'], how='left')

    # 4. Forward Fill:
    df_normalised = df_normalised.sort_values(['asset_id', 'valuation_date'])
    df_normalised['asset_name'] = df_normalised.groupby('asset_id')['asset_name'].ffill()
    df_normalised['asset_type'] = df_normalised.groupby('asset_id')['asset_type'].ffill()
    df_normalised['asset_owner'] = df_normalised.groupby('asset_id')['asset_owner'].ffill()
    df_normalised['asset_description'] = df_normalised.groupby('asset_id')['asset_description'].ffill()
    df_normalised['asset_currency'] = df_normalised.groupby('asset_id')['asset_currency'].ffill()
    df_normalised['asset_value'] = df_normalised.groupby('asset_id')['asset_value'].ffill()

    return df_normalised

####################################################################################################

def create_monthly_summary(df):
    """Creates a DataFrame with one entry per asset per month, 
       using the value from the *first* day of the month.
    """
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])
    df = df.sort_values(['asset_id', 'valuation_date'])

    # 1. Create a 'month' column:
    df['month'] = df['valuation_date'].dt.to_period('M')

    # 2. Group by asset and month:
    grouped = df.groupby(['asset_id', 'month'])

    # 3. Get the *first* date and value for each group:
    df_monthly = grouped.agg({'valuation_date': 'min', 'asset_value': 'first'}).reset_index()

    # 4. Convert 'month' back to datetime (first day of the month):
    df_monthly['valuation_date'] = df_monthly['month'].dt.to_timestamp()
    df_monthly = df_monthly.drop(columns=['month']) # Drop the extra 'month' column

    return df_monthly

####################################################################################################
####################################################################################################
# Tests Section
####################################################################################################

def run_tests(df, df_full):
    print('\n', 10 * '#', 'Test no.1', 10 * '#')

    # Test 1: Create a DataFrame with the required columns
    min_dates_original = df.groupby('asset_id')['valuation_date'].min()
    min_dates_filled = df_full.groupby('asset_id')['valuation_date'].min()

    # Debugging outputs for min_dates
    print(min_dates_original)
    print(min_dates_filled)
    
    # Initialize lists to store asset values
    asset_values_original = []
    asset_values_filled = []

    # For each asset_id in min_dates_original, get the corresponding asset_value
    for asset_id, min_date in min_dates_original.items():
        # Get the asset_value from df for the original valuation date
        asset_value_original = df.loc[(df['asset_id'] == asset_id) & (df['valuation_date'] == min_date), 'asset_value'].values
        # Get the asset_value from df_full for the original valuation date
        asset_value_filled = df_full.loc[(df_full['asset_id'] == asset_id) & (df_full['valuation_date'] == min_date), 'asset_value'].values
        
        # If no value is found, fill with NaN
        asset_values_original.append(asset_value_original[0] if len(asset_value_original) > 0 else None)
        asset_values_filled.append(asset_value_filled[0] if len(asset_value_filled) > 0 else None)

    # Create the DataFrame with the required columns
    test_1_df = pd.DataFrame({
        'asset_id': min_dates_original.index,
        'min_date (original)': min_dates_original.values,
        'min_date (after fill)': min_dates_filled.values,
        'asset_value (original)': asset_values_original,
        'asset_value (after fill)': asset_values_filled
    }).reset_index(drop=True)

    # Output the result
    print(test_1_df)

    print('\n', 10 * '#', 'Test no.2', 10 * '#')

    # Test 2: Filter for date = '2022-01-03' and get required columns
    test_date = pd.Timestamp('2022-01-03')
    original_values = df[df['valuation_date'] == test_date].set_index('asset_id')['asset_value']
    filled_values = df_full[df_full['valuation_date'] == test_date].set_index('asset_id')['asset_value']

    test_2_df = pd.DataFrame({
        'asset_id': df['asset_id'].unique(),
        'date': test_date,
        'asset_value (original)': original_values.reindex(df['asset_id'].unique()).values,
        'asset_value (after fill)': filled_values.reindex(df['asset_id'].unique()).values
    }).reset_index(drop=True)

    print(test_2_df)

####################################################################################################
def calculate_summary_metrics(df):
    """
    Calculates summary metrics from asset position data.

    Args:
        df (pd.DataFrame): DataFrame with asset position data (asset_id, valuation_date, current_value, asset_name).

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    # Ensure valuation_date is datetime in the input df BEFORE passing to fill_missing_asset_values
    df['valuation_date'] = pd.to_datetime(df['valuation_date'])

    # Step 1: Use fill_missing_asset_values to get a daily, forward-filled DataFrame
    df_daily_filled = fill_missing_asset_values(df.copy())

    # Step 2: Aggregate the daily filled data to a monthly level, taking the last value of the month
    df_daily_filled['month_period'] = df_daily_filled['valuation_date'].dt.to_period('M')

    df_monthly_grouped = df_daily_filled.sort_values(by=['asset_id', 'valuation_date']).groupby(
        ['asset_id', 'asset_name', 'month_period']
    )['asset_value'].last().reset_index()

    df_monthly_grouped.rename(columns={'month_period': 'month'}, inplace=True)

    df_grouped = df_monthly_grouped

    # Calculate total asset value for the current month
    current_month = df_grouped['month'].max()
    total_asset_value = df_grouped[df_grouped['month'] == current_month]['asset_value'].sum()

    # Calculate monthly change
    previous_month = current_month - 1
    current_month_value = df_grouped[df_grouped['month'] == current_month]['asset_value'].sum()
    previous_month_value = df_grouped[df_grouped['month'] == previous_month]['asset_value'].sum()

    if previous_month_value > 0:
        monthly_change_euro = current_month_value - previous_month_value
        monthly_change_percent = (monthly_change_euro / previous_month_value) * 100
    else:
        monthly_change_euro = 0
        monthly_change_percent = 0

    # Calculate annual change
    annual_month = current_month - 12
    annual_month_value = df_grouped[df_grouped['month'] == annual_month]['asset_value'].sum()

    if annual_month_value > 0:
        annual_change_euro = current_month_value - annual_month_value
        annual_change_percent = (annual_change_euro / annual_month_value) * 100
    else:
        annual_change_euro = 0
        annual_change_percent = 0

    # Calculate monthly asset performance
    monthly_performance = df_grouped.copy()
    monthly_performance['previous_month'] = monthly_performance.groupby('asset_id')['asset_value'].shift(1)

    # Handle division by zero or NaN previous_month values for change_percent
    monthly_performance['change_percent'] = np.nan # Initialize with NaN
    # Only calculate change_percent where previous_month is not zero and not NaN
    valid_previous_month_for_calc = (monthly_performance['previous_month'] != 0) & (monthly_performance['previous_month'].notna())
    monthly_performance.loc[valid_previous_month_for_calc, 'change_percent'] = (
        (monthly_performance['asset_value'] - monthly_performance['previous_month']) / monthly_performance['previous_month']
    ) * 100

    monthly_performance = monthly_performance[monthly_performance['month'] == current_month]  # Filter for current month

    # Sort with na_position='last' to ensure NaN values are at the end
    top_3_monthly = monthly_performance.sort_values('change_percent', ascending=False, na_position='last').head(3)
    bottom_3_monthly = monthly_performance.sort_values('change_percent', ascending=True, na_position='last').head(3)

    # Calculate 12-Month asset performance
    annual_pivot = df_grouped.pivot_table(index=['asset_id', 'asset_name'], columns='month', values='asset_value').reset_index()

    annual_pivot['current_value'] = annual_pivot[current_month]
    annual_pivot['previous_year_value'] = annual_pivot[annual_month]

    # Handle division by zero or NaN previous_year_value for change_percent
    annual_pivot['change_percent'] = np.nan # Initialize with NaN
    # Only calculate change_percent where previous_year_value is not zero and not NaN
    valid_previous_year_for_calc = (annual_pivot['previous_year_value'] != 0) & (annual_pivot['previous_year_value'].notna())
    annual_pivot.loc[valid_previous_year_for_calc, 'change_percent'] = (
        (annual_pivot['current_value'] - annual_pivot['previous_year_value']) / annual_pivot['previous_year_value']
    ) * 100

    # DO NOT use annual_pivot.fillna(0) for change_percent here.
    # Rely on na_position in sort_values.

    top_3_annual = annual_pivot.sort_values('change_percent', ascending=False, na_position='last').head(3)
    bottom_3_annual = annual_pivot.sort_values('change_percent', ascending=True, na_position='last').head(3)

    return {
        'total_asset_value': total_asset_value,
        'monthly_change_euro': monthly_change_euro,
        'monthly_change_percent': monthly_change_percent,
        'annual_change_euro': annual_change_euro,
        'annual_change_percent': annual_change_percent,
        'top_3_monthly': top_3_monthly,
        'bottom_3_monthly': bottom_3_monthly,
        'top_3_annual': top_3_annual,
        'bottom_3_annual': bottom_3_annual,
        'current_month_for_calc': current_month
    }
####################################################################################################

def prepare_asset_data_mortgage(df_raw, include_mortgage_flag):
    """
    Prepares the asset data for analysis based on the include_mortgage_flag.
    If include_mortgage_flag is True, calculates house equity as a pseudo-asset.
    Otherwise, filters out the mortgage asset.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame from get_asset_history,
                                containing all assets including house and mortgage.
        include_mortgage_flag (bool): Flag from Streamlit sidebar to determine
                                    if mortgage should be included in equity calculation.

    Returns:
        pd.DataFrame: DataFrame with mortgage/house equity handled.
                    Returns None if df_raw is None or empty.
    """
    if df_raw is None or df_raw.empty:
        return None

    # Ensure valuation_date is datetime for consistency
    df_raw['valuation_date'] = pd.to_datetime(df_raw['valuation_date'])

    # Define asset IDs for house and mortgage
    HOUSE_ASSET_ID = 14
    MORTGAGE_ASSET_ID = 19
    PSEUDO_ASSET_ID = 99
    PSEUDO_ASSET_NAME = 'azcona_43_house_owned'
    PSEUDO_ASSET_TYPE = 'housing'

    if include_mortgage_flag:
        # 1. Filter out house and mortgage from the main DataFrame
        df_other_assets = df_raw[
            (df_raw['asset_id'] != HOUSE_ASSET_ID) &
            (df_raw['asset_id'] != MORTGAGE_ASSET_ID)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # 2. Isolate house and mortgage data
        df_house = df_raw[df_raw['asset_id'] == HOUSE_ASSET_ID].copy()
        df_mortgage = df_raw[df_raw['asset_id'] == MORTGAGE_ASSET_ID].copy()

        # 3. Ensure both house and mortgage data are filled for all dates
        #    This is crucial for accurate subtraction.
        #    fill_missing_asset_values needs asset_name, asset_type, etc.
        #    Let's add dummy values if they are missing for the purpose of fill_missing_asset_values
        #    or ensure df_raw already has them. df_raw from get_asset_history should have them.
        
        # We need to pass the full columns for fill_missing_asset_values
        # Create a combined df for house and mortgage to fill missing values together
        df_combined_house_mortgage = pd.concat([df_house, df_mortgage])
        
        # Apply fill_missing_asset_values to ensure continuous data for both
        df_filled_combined = fill_missing_asset_values(df_combined_house_mortgage)

        # Separate them again after filling
        df_house_filled = df_filled_combined[df_filled_combined['asset_id'] == HOUSE_ASSET_ID]
        df_mortgage_filled = df_filled_combined[df_filled_combined['asset_id'] == MORTGAGE_ASSET_ID]

        # 4. Merge house and mortgage data on valuation_date
        #    Use an outer merge to keep all dates from both, then fill NaNs if any
        df_merged = pd.merge(
            df_house_filled[['valuation_date', 'asset_value']],
            df_mortgage_filled[['valuation_date', 'asset_value']],
            on='valuation_date',
            how='outer',
            suffixes=('_house', '_mortgage')
        )

        # Fill any remaining NaNs after merge (e.g., if one started earlier than the other)
        # This should ideally be handled by fill_missing_asset_values, but as a safeguard
        df_merged['asset_value_house'] = df_merged['asset_value_house'].ffill().bfill()
        df_merged['asset_value_mortgage'] = df_merged['asset_value_mortgage'].ffill().bfill()
        
        # Handle cases where there might be no data for one of them at all (e.g., new mortgage)
        df_merged['asset_value_house'] = df_merged['asset_value_house'].fillna(0)
        df_merged['asset_value_mortgage'] = df_merged['asset_value_mortgage'].fillna(0)

        # 5. Calculate the pseudo-asset value (House - Mortgage)
        df_merged['asset_value'] = df_merged['asset_value_house'] - df_merged['asset_value_mortgage']

        # 6. Create the pseudo-asset DataFrame
        df_pseudo_asset = pd.DataFrame({
            'asset_id': PSEUDO_ASSET_ID,
            'asset_name': PSEUDO_ASSET_NAME,
            'asset_type': PSEUDO_ASSET_TYPE,
            'asset_owner': 'Common',
            'asset_description': 'Net equity in Azcona 43 (House Value - Mortgage Balance)',
            'asset_currency': 'EUR',
            'valuation_date': df_merged['valuation_date'],
            'asset_value': df_merged['asset_value']
        })

        # 7. Concatenate with other assets
        df_processed = pd.concat([df_other_assets, df_pseudo_asset], ignore_index=True)

    else:
        # If mortgage is not included, simply filter out the mortgage asset
        df_processed = df_raw[df_raw['asset_id'] != MORTGAGE_ASSET_ID].copy()

    # Ensure final DataFrame has correct dtypes and is sorted
    df_processed['valuation_date'] = pd.to_datetime(df_processed['valuation_date'])
    df_processed = df_processed.sort_values(by=['valuation_date', 'asset_id'])

    return df_processed

####################################################################################################

def prepare_asset_data_owner(df_raw, owner_selection):
    """
    Filters and adjusts asset values based on the selected owner.

    Args:
        df_raw (pd.DataFrame): The DataFrame of assets, potentially already processed for mortgage equity.
        owner_selection (str): The selected owner ('All', 'Carlos', 'Sara', 'Common').

    Returns:
        pd.DataFrame: The DataFrame filtered and adjusted for the selected owner.
                    Returns None if df_raw is None or empty.
    """
    if df_raw is None or df_raw.empty:
        return None

    df_processed = df_raw.copy() # Work on a copy to avoid modifying the original DataFrame

    if owner_selection == "All":
        # No changes needed, return the full DataFrame
        pass
    elif owner_selection == "Carlos":
        # Select assets owned by Carlos
        df_carlos_assets = df_processed[df_processed['asset_owner'] == 'Carlos'].copy()
        
        # Select assets owned by Common and divide their value by 2
        df_common_assets = df_processed[df_processed['asset_owner'] == 'Common'].copy()
        df_common_assets['asset_value'] = df_common_assets['asset_value'] / 2
        
        # Concatenate the two sets of assets
        df_processed = pd.concat([df_carlos_assets, df_common_assets], ignore_index=True)

    elif owner_selection == "Sara":
        # Select assets owned by Sara
        df_sara_assets = df_processed[df_processed['asset_owner'] == 'Sara'].copy()
        
        # Select assets owned by Common and divide their value by 2
        df_common_assets = df_processed[df_processed['asset_owner'] == 'Common'].copy()
        df_common_assets['asset_value'] = df_common_assets['asset_value'] / 2
        
        # Concatenate the two sets of assets
        df_processed = pd.concat([df_sara_assets, df_common_assets], ignore_index=True)

    elif owner_selection == "Common":
        # Select only assets owned by Common
        df_processed = df_processed[df_processed['asset_owner'] == 'Common'].copy()

    # Ensure the DataFrame is sorted after processing
    df_processed = df_processed.sort_values(by=['valuation_date', 'asset_id'])

    return df_processed

####################################################################################################

def prepare_asset_data_type_focus(df_raw, asset_type_focus):
    """
    Filters and adjusts asset values based on the selected asset type (cash vs non-cash).

    Args:
        df_raw (pd.DataFrame): The DataFrame of assets, potentially already processed for mortgage equity and owner.
        asset_type_focus (str): The selected owner ("All Assets", "Non-Cash Assets", "Cash Accounts Only").

    Returns:
        pd.DataFrame: The DataFrame filtered and adjusted for the selected asset type.
                    Returns None if df_raw is None or empty.
    """
    if df_raw is None or df_raw.empty:
        return None

    df_processed = df_raw.copy() # Work on a copy to avoid modifying the original DataFrame

    if asset_type_focus == "All Assets":
        # No changes needed, return the full DataFrame
        pass

    elif asset_type_focus == "Non-Cash Assets":
        # Select account assets only
        df_processed = df_processed[df_processed['asset_type'] != 'account_balance'].copy()

    elif asset_type_focus == "Cash Accounts Only":
        # Select assets owned by Sara
        df_processed = df_processed[df_processed['asset_type'] == 'account_balance'].copy()

    # Ensure the DataFrame is sorted after processing
    df_processed = df_processed.sort_values(by=['valuation_date', 'asset_id'])

    return df_processed

####################################################################################################

def convert_value(row, exchange_rates, target_currency):
    # Map the database currency name to its API code
    original_currency_api_code = config.currency_map.get(row['asset_currency'], row['asset_currency'])
    original_value = row['asset_value']

    # If original currency (mapped to API code) is already the target currency, no conversion needed
    if original_currency_api_code == target_currency:
        return original_value

    # All rates from ExchangeRate-API.com (when base_currency="EUR") are relative to EUR.
    # So, to convert from ORIGINAL_CURRENCY to TARGET_CURRENCY:
    # 1. Convert ORIGINAL_CURRENCY to EUR: value_in_eur = original_value / exchange_rates[original_currency]
    # 2. Convert EUR to TARGET_CURRENCY: final_value = value_in_eur * exchange_rates[target_currency]
    # Combined: final_value = original_value * (exchange_rates[target_currency] / exchange_rates[original_currency])

    rate_original_to_eur = exchange_rates.get(original_currency_api_code)
    rate_eur_to_target = exchange_rates.get(target_currency)

    if rate_original_to_eur is None or rate_eur_to_target is None:
        # This warning should ideally be caught earlier or handled more gracefully.
        # For now, return original value if rate is missing.
        # Consider logging this or displaying a more prominent error in the UI if rates are critical.
        st.warning(f"Exchange rate not found for {original_currency_api_code} or {target_currency}. Value not converted.")
        return original_value 

    # Perform the conversion
    converted_value = original_value * (rate_eur_to_target / rate_original_to_eur)

    return converted_value
    
####################################################################################################

def prepare_asset_data_currency_conversion(df_input, target_currency, secrets_file_path):
    """
    Converts all asset values in the DataFrame to the target currency.

    Args:
        df_input (pd.DataFrame): The DataFrame of assets with original currencies.
        target_currency (str): The currency to convert to (e.g., "EUR", "GBP", "USD").
        secrets_file_path (str): Path to the secrets.json file for API key.

    Returns:
        pd.DataFrame: The DataFrame with all asset values converted to the target currency.
                    Returns None if df_input is None or empty, or if exchange rates cannot be fetched.
    """
    if df_input is None or df_input.empty:
        return None

    # Fetch exchange rates (always relative to EUR from ExchangeRate-API.com)
    # We pass "EUR" as the base_currency to get rates relative to EUR.
    exchange_rates = lib_api.get_exchange_rates(secrets_file_path, base_currency="EUR")

    if exchange_rates is None:
        st.error("Could not fetch exchange rates for currency conversion.")
        return None

    df_processed = df_input.copy()

    # Apply conversion
    # Iterate through each row and convert based on its original currency
    df_processed['asset_value'] = df_processed.apply(
    lambda row: convert_value(row, exchange_rates, target_currency),
    axis=1
    )

    df_processed['asset_currency'] = target_currency # Update currency column to the new target currency

    return df_processed

####################################################################################################

def prepare_asset_data_for_analysis(df_raw, include_mortgage_flag, owner_selection, asset_type_focus, target_currency, secrets_file_path):
    """
    Orchestrates the data preparation pipeline for asset analysis.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame from get_asset_history.
        include_mortgage_flag (bool): Flag to include/exclude mortgage equity.
        owner_selection (str): Selected owner for filtering.
        asset_type_focus (str): Selected asset type focus for filtering.
        target_currency (str): The currency to convert all values to.
        secrets_file_path (str): Path to the secrets.json file for API key.

    Returns:
        pd.DataFrame: The fully processed DataFrame ready for metrics and charting.
                    Returns None if the initial df_raw is None/empty or if any
                    processing step results in an empty DataFrame.
    """
    # Step 1: Handle mortgage/house equity
    df_mortgage = prepare_asset_data_mortgage(df_raw, include_mortgage_flag)
    if df_mortgage is None: return None

    # Step 2: Handle owner-based filtering and value adjustment
    df_owner = prepare_asset_data_owner(df_mortgage, owner_selection)
    if df_owner is None: return None

    # Step 3: Handle asset type focus filtering
    df_type_focus = prepare_asset_data_type_focus(df_owner, asset_type_focus)
    if df_type_focus is None: return None

    # Step 4: Handle currency conversion
    df_converted = prepare_asset_data_currency_conversion(df_type_focus, target_currency, secrets_file_path)
    if df_converted is None: return None

    # Step 5: Add asset_label for display
    df_final = df_converted.copy() # Work on a copy
    df_final['asset_label'] = df_final['asset_name'].apply(apply_labelling_assets)

    return df_final

####################################################################################################

def apply_labelling_assets(asset_name):
    """
    Returns the user-friendly label for an asset_name from the ASSET_LABELS map.
    If no label is found, returns the original asset_name.

    Args:
        asset_name (str): The internal asset_name (e.g., "acc_bk_sm_5081").

    Returns:
        str: The user-friendly label (e.g., "Bankinter Savings (S&M)") or the original asset_name.
    """
    return config.ASSET_LABELS.get(asset_name, asset_name)

####################################################################################################

####################################################################################################

####################################################################################################
####################################################################################################
# Cached functions
####################################################################################################
# Function to get asset order from the database (cached)
@st.cache_data
def get_asset_order(_engine):
    """
    Gets a sorted list of asset IDs based on their first appearance in the positions table.
    This is used to maintain a consistent stacking order in charts.

    Args:
        _engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        list: A list of asset IDs, or None if an error occurs.
    """
    asset_order_query = """
    SELECT 
        asset_id
    FROM 
        positions
    GROUP BY 
        asset_id
    ORDER BY 
        MIN(valuation_date) ASC;
    """
    try:
        # Read directly from the database using the engine.
        asset_order_df = pd.read_sql_query(asset_order_query, _engine)
        
        if asset_order_df is not None:
            return asset_order_df['asset_id'].tolist()
        
        return None # Return None if the DataFrame is None

    except Exception as e:
        st.error(f"Error loading asset order: {e}")
        return None
####################################################################################################
# Query to get asset value evolution (cached)
@st.cache_data
def get_asset_history(_engine):
    """
    Fetches the complete, joined asset and position history from the database.

    Args:
        _engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        pd.DataFrame: A DataFrame with the complete asset history.
    """
    query = """
    SELECT
        p.asset_id,
        da.asset_name,
        da.asset_type,
        da.asset_owner,
        da.asset_description,
        da.asset_currency,
        p.valuation_date,
        p.current_value AS asset_value
    FROM 
        positions p
    LEFT JOIN 
        dim_assets da
    ON 
        p.asset_id = da.asset_id;
    """
    try:
        # Read directly from the database using the engine. No more helper functions needed.
        df = pd.read_sql_query(query, _engine)
        print("Asset history loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading asset history: {e}")
        return None
####################################################################################################
####################################################################################################
# Function to get asset data from the database (cached)
@st.cache_data  # Cache this function as well
def get_recent_asset_data(_engine):
    """
    Fetches the most recent valuation date and value for each asset from the database.
    This is used to populate the data ingestion sidebar and forms.

    Args:
        _engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        pd.DataFrame: A DataFrame with the most recent data for each asset, or None on error.
    """
    query = """
    SELECT
        da.asset_id,
        da.asset_name,
        da.asset_description,
        da.asset_owner,
        p.valuation_date AS latest_valuation_date,
        p.current_value AS latest_asset_value
    FROM
        dim_assets da
    LEFT JOIN
        positions p ON da.asset_id = p.asset_id
    WHERE
        p.valuation_date = (
            SELECT
                MAX(valuation_date)
            FROM
                positions
            WHERE
                asset_id = da.asset_id
        )
        OR p.valuation_date IS NULL;
    """
    asset_data_df = pd.read_sql_query(query, _engine)
    # Convert latest_valuation_date to date objects immediately after retrieval
    if asset_data_df is not None:
        # Check for non-NaT values before using .dt
        valid_dates = pd.to_datetime(asset_data_df['latest_valuation_date'], errors='coerce')
        asset_data_df['latest_valuation_date'] = valid_dates
        asset_data_df['latest_valuation_date_date'] = valid_dates.dt.date

    return asset_data_df

####################################################################################################
####################################################################################################
