# Core data tools
import pandas as pd
import numpy as np
import geopandas as gpd
import math

# Visualization
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # For custom map legends
import seaborn as sns

from shapely.geometry import Point
from scipy.spatial import cKDTree

from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.patches import Patch

counties_gdf = gpd.read_file('../data/raw/CA_Counties.shp').to_crs(epsg=4326)
       
def interpolate_idw(
    gdf_points: gpd.GeoDataFrame,
    value_column: str,
    firename: str = None,
    k: int = 5,
    grid_spacing: float = 0.05,
    buffer: float = 0.1,
    crop: bool = False,
    title: str = "",
    plot: bool = True
) -> gpd.GeoDataFrame:
    """
    Interpolate values spatially using Inverse Distance Weighting (IDW),
    clipped to county boundaries, with fire star markers, continuous severity colormap,
    and a labeled colorbar.

    Parameters:
        gdf_points (GeoDataFrame): Input GeoDataFrame with point geometries and a value column.
        value_column (str): Column with values to interpolate (e.g., 'Target').
        county_gdf (GeoDataFrame): Polygon GeoDataFrame for boundary clipping.
        firename (str): Optional fire name ('Pal' or 'Dixie') for custom axis limits and markers.
        k (int): Number of neighbors used in IDW.
        grid_spacing (float): Grid resolution in degrees.
        buffer (float): Padding around point extent.
        plot (bool): Whether to display the plot.

    Returns:
        GeoDataFrame: Interpolated values clipped to county boundaries.
    """
    # Extract coordinates and values
    known_coords = np.array([(geom.x, geom.y) for geom in gdf_points.geometry])
    known_values = gdf_points[value_column].values

    # Define interpolation grid bounds
    minx, miny, maxx, maxy = gdf_points.total_bounds
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer

    x_coords = np.arange(minx, maxx, grid_spacing)
    y_coords = np.arange(miny, maxy, grid_spacing)
    grid_points = [Point(x, y) for x in x_coords for y in y_coords]
    grid_coords = np.array([(pt.x, pt.y) for pt in grid_points])

    # IDW interpolation
    tree = cKDTree(known_coords)
    distances, indices = tree.query(grid_coords, k=k)
    weights = 1 / (distances + 1e-12)
    interpolated_values = np.sum(weights * known_values[indices], axis=1) / np.sum(weights, axis=1)

    # Interpolated GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({'Interpolated': interpolated_values}, geometry=grid_points, crs=gdf_points.crs)
    grid_clipped = gpd.overlay(grid_gdf, counties_gdf, how='intersection')
    
    xlim = ylim = None
    
    # Fire locations
    fire_points = []
    if (firename == 'Pal'):
        fire_points = [
            {'name': 'Palisades Fire', 'lat': 34.07022, 'lon': -118.54453, 'color': 'gold'},
            {'name': 'Eaton Fire',     'lat': 34.203483, 'lon': -118.069155, 'color': 'blue'}
        ]
        if crop:
            xlim = (-119.5, -117)
            ylim = (32.5, 34.5)
            
    elif (firename == 'Dixie'):
        fire_points = [{'name': 'Dixie Fire', 'lat': 39.871306, 'lon': -121.389439, 'color': 'red'}]
        if crop:
            xlim = (-125, -119)
            ylim = (39, 42.2)
    else:
        xlim = ylim = None

    # Plotting
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Custom continuous colormap for severity
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_heat', ['lightgray', 'orange', 'red']
        )

        # Plot with continuous severity map (0–2)
        grid_clipped.plot(
            column='Interpolated',
            cmap=custom_cmap,
            vmin=0,
            vmax=2,
            legend=False,
            ax=ax
        )

        # Add labeled colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=2))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Wildfire Severity', fontsize=10)
        cbar.ax.set_yticks([0, 1, 2])
        cbar.ax.set_yticklabels(['Low', 'Moderate', 'High'])

        # County boundaries
        counties_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)

        # Fire markers
        legend_handles = []
        for fire in fire_points:
            ax.scatter(fire['lon'], fire['lat'], color=fire['color'], edgecolor='black',
                       marker='*', s=200, zorder=5)
            legend_handles.append(Patch(facecolor=fire['color'], edgecolor='black', label=fire['name']))

        # Add fire legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', title='Fire Locations',
                      fontsize=9, title_fontsize=10)

        # Apply axis limits if applicable
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_title(f'Southern California Interpolated {value_column} 01/07/2025')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        plt.savefig("../plots/Interpolated.png", dpi=300)

    return grid_clipped

def correlation_map(df, title):
    
    # Compute the correlation matrix
    corr = df.corr()

    # Create a mask to show only the lower triangle (for cleaner layout)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    sns.set(style="white")

    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .8},
        vmin=-1, vmax=1,
        annot_kws={"size": 10}
    )

    # Title and formatting
    plt.title(title, fontsize=14, pad=12)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
def individual_plot_map(gdf, gdf_column, firename):
    """Plots wildfire severity predictions on a California map with fire location markers and categorical styling."""
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot base map: counties
    counties_gdf.plot(ax=ax, color='whitesmoke', edgecolor='gray', linewidth=0.5)

    # Define fire locations and labels
    if firename == 'Pal':
        fire_points = [
            {'name': 'Palisades Fire', 'lat': 34.07022, 'lon': -118.54453},
            {'name': 'Eaton Fire',     'lat': 34.203483, 'lon': -118.069155}
        ]
    else:  # Dixie Fire
        fire_points = [{'name': 'Dixie Fire', 'lat': 39.871306, 'lon': -121.389439}]

    # Plot fire locations
    for fire in fire_points:
        ax.scatter(fire['lon'], fire['lat'], s=500, c='red', marker='*', edgecolor='black', alpha=0.8, zorder=3)
        ax.text(fire['lon'] + 0.1, fire['lat'] - 0.03, fire['name'], fontsize=11, color='darkred', fontweight='bold')

    # Color palette for severity predictions
    prediction_colors = {
        'Low': '#4575b4',        # blue
        'Moderate': '#91bfdb',   # light blue
        'High': '#f46d43',       # orange-red
        'Extreme': '#d73027'     # red
    }

    # Plot severity predictions
    for _, row in gdf.iterrows():
        ax.scatter(row['Longitude'], row['Latitude'],
                   color=prediction_colors.get(row[gdf_column], 'gray'),
                   s=200, edgecolor='black', alpha=0.7, zorder=2)

    # Custom legend
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', label=label,
                      markersize=10, markerfacecolor=color, markeredgecolor='black')
        for label, color in prediction_colors.items()
    ]
    ax.legend(handles=legend_handles, title="Predicted Severity", title_fontsize=13,
              fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Set axis limits
    if firename == 'Pal':
        ax.set_xlim(-119.5, -117)
        ax.set_ylim(32.5, 34.5)
    else:
        ax.set_xlim(-123.25, -120.75)
        ax.set_ylim(39.5, 41.5)

    ax.set_title(f"{gdf_column} - {firename} Fire", fontsize=18, pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    
def plot_map(gdf, gdf_column, firename, ax=None):
    
    """Plots wildfire severity predictions on a California map with fire location markers and categorical styling."""

    # Use passed axis or create one if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Base map
    counties_gdf.plot(ax=ax, color='whitesmoke', edgecolor='gray', linewidth=0.5)

    # Fire location(s)
    if firename == 'Pal' or firename == 'Pal Cali':
        fire_points = [
            {'name': 'Palisades Fire', 'lat': 34.07022, 'lon': -118.54453},
            {'name': 'Eaton Fire',     'lat': 34.203483, 'lon': -118.069155}
        ]
    else:
        fire_points = [{'name': 'Dixie Fire', 'lat': 39.871306, 'lon': -121.389439}]

    for fire in fire_points:
        ax.scatter(fire['lon'], fire['lat'], s=500, c='red', marker='*',
                   edgecolor='black', alpha=0.8, zorder=3)
        ax.text(fire['lon'] + 0.1, fire['lat'] - 0.03, fire['name'],
                fontsize=11, color='darkred', fontweight='bold')

    # Color palette
    prediction_colors = {
        'Low': '#4575b4',
        'Moderate': '#f46d43',
        'High': '#d73027'
    }

    # Plot predictions
    for _, row in gdf.iterrows():
        ax.scatter(row['Longitude'], row['Latitude'],
                   color=prediction_colors.get(row[gdf_column], 'gray'),
                   s=200, edgecolor='black', alpha=0.7, zorder=2)

    # Custom legend (only for first axis if sharing)
    if ax.get_subplotspec().is_first_col():
        legend_handles = [
            mlines.Line2D([], [], marker='o', color='w', label=label,
                          markersize=10, markerfacecolor=color, markeredgecolor='black')
            for label, color in prediction_colors.items()
        ]
        ax.legend(handles=legend_handles, title="Predicted Severity", title_fontsize=13,
                  fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Axis limits
    if firename == 'Pal':
        ax.set_xlim(-119.5, -117)
        ax.set_ylim(32.5, 34.5)
    elif firename == 'Dixie':
        ax.set_xlim(-125, -119)
        ax.set_ylim(39, 42.2)

    ax.set_title(f"{gdf_column} - {firename} Fire", fontsize=14, pad=10)
    ax.set_axis_off()

    return ax  # return the axis for optional external control

def grid_kde(df):
    # Configuration
    plots_per_col = 2  # 2 plots vertically stacked per column
    num_plots = len(df.columns)
    cols = math.ceil(num_plots / plots_per_col)
    rows = plots_per_col

    # Create subplots with wider width (for multiple columns)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    sns.set(style="whitegrid")

    for i, column in enumerate(df.columns):
        sns.kdeplot(
            data=df[column].dropna(),
            ax=axes[i],
            fill=True,
            color='skyblue',
            linewidth=1.5
        )
        axes[i].set_title(column, fontsize=15, weight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Title below plots
    #plt.figtext(0.5, -0.02, 'KDE Distributions of Damage Measures', 
    #            fontsize=20, weight='bold', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    
def grid_box(df):
    # Configuration
    max_cols = 5
    num_columns = len(df.columns)
    cols = min(max_cols, num_columns)
    rows = math.ceil(num_columns / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 8 * rows))
    axes = axes.flatten()

    # Style
    sns.set(style="whitegrid")

    # Plot each boxplot vertically
    for i, column in enumerate(df.columns):
        sns.boxplot(
            y=df[column],
            ax=axes[i],
            color='skyblue',
            fliersize=2,
            linewidth=1
        )
        axes[i].set_title(column, fontsize=15, weight='bold')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Enlarged title
    plt.suptitle('Damage Measures', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

        
def build_index(category):
    if category == 'Extreme':
        return 3
    if category == 'High':
        return 2
    if category == 'Moderate':
        return 1
    else:
        return 0  # Default case for 'Low' or other unspecified categories