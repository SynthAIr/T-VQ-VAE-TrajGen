import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import altair as alt
import cartopy.crs as ccrs
import cartopy.feature
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cartopy.crs import EuroPP, PlateCarree
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from traffic.core import Traffic
from traffic.data import airports


def extract_geographic_info(
    trajectories: Traffic,
    lon_padding: float = 1,
    lat_padding: float = 1,
) -> Tuple[float, float, float, float, float, float]:

    # Determine the geographic bounds for plotting
    lon_min = trajectories.data["longitude"].min()
    lon_max = trajectories.data["longitude"].max()
    lat_min = trajectories.data["latitude"].min()
    lat_max = trajectories.data["latitude"].max()

    geographic_extent = [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding,
    ]

    return geographic_extent


def extract_airport_coordinates(
    training_data_path: str,
) -> Tuple[float, float]:

    training_data = Traffic.from_file(training_data_path)

    if len(training_data.data["ADES"].unique()) > 1:
        raise ValueError("There are multiple destination airports in the training data")
    if len(training_data.data["ADES"].unique()) == 0:
        raise ValueError("There are no destination airports in the training data")

    ADES_code = training_data.data["ADES"].value_counts().idxmax()
    ADES_lat = airports[ADES_code].latitude
    ADES_lon = airports[ADES_code].longitude
    return ADES_lat, ADES_lon


def plot_trajectories(
    trajectories: Traffic,
    geographic_extent: List[float],
    save_path: str,
    ADEP_code: str = None,
    ADES_code: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": EuroPP()})

    trajectories.plot(ax, alpha=0.2, color="darkblue", linewidth=1)

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax.set_extent(geographic_extent)

    # Plot the origin and destination airports
    if ADEP_code is not None:
        airports[ADEP_code].point.plot(
            ax, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
        )
    if ADES_code is not None:
        airports[ADES_code].point.plot(
            ax, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
        )

    # plt.title(f"Trajectories from {ADEP_code} to {ADES_code}")
    plt.legend(loc="upper right")

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    # tight layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_trajectories_with_class_colors(
    trajectories: Traffic,
    geographic_extent: List[float],
    save_path: str,
    ADEP_code: str = None,
    ADES_code: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": EuroPP()})

    # Get unique clusters and assign colors
    n_clusters = 1 + trajectories.data.cluster.max()
    colors = sns.color_palette("husl", n_clusters)

    # Plot each trajectory with its corresponding cluster color
    for cluster in range(n_clusters):
        trajectories.query(f"cluster == {cluster}").plot(
            ax, alpha=0.2, color=colors[cluster], linewidth=1
        )

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax.set_extent(geographic_extent)

    # Plot the origin and destination airports
    if ADEP_code is not None:
        airports[ADEP_code].point.plot(
            ax, color="red", label=f"Origin: {ADEP_code}", s=300, zorder=5
        )
    if ADES_code is not None:
        airports[ADES_code].point.plot(
            ax, color="green", label=f"Destination: {ADES_code}", s=300, zorder=5
        )

    plt.legend(loc="best")

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    # tight layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def get_landing_trajectories(trajectories: Traffic, landing_time: int) -> Traffic:
    landing_trajectories = trajectories.last(minutes=landing_time).eval()
    return landing_trajectories


def get_takeoff_trajectories(trajectories: Traffic, takeoff_time: int) -> Traffic:
    takeoff_trajectories = trajectories.first(minutes=takeoff_time).eval()
    return takeoff_trajectories

def plot_clustering(
    trajectories: Traffic,
    geographic_extent: List[float],
    save_path: str,
    ADEP_code: str = None,
    ADES_code: str = None,
) -> None:
    """
    Class-conditional generation of flight trajectories
    """

    # t_gmm = Traffic.from_file(trajectories_path)

    n_clusters = 1 + trajectories.data.cluster.max()

    colors = sns.color_palette("husl", n_clusters)

    nb_cols = n_clusters
    nb_lines = 1

    fig, axs = plt.subplots(
        nb_lines, nb_cols, figsize=(30, 8), subplot_kw={"projection": EuroPP()}
    )

    for cluster in range(n_clusters):
        ax = axs[cluster]

        trajectories.query(f"cluster == {cluster}").plot(
            ax, alpha=0.2, color=colors[cluster], linewidth=1
        )
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
        ax.set_extent(geographic_extent)

        # Plot the origin and destination airports
        if ADEP_code is not None:
            airports[ADEP_code].point.plot(
                ax, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
            )
        if ADES_code is not None:
            airports[ADES_code].point.plot(
                ax, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
            )

        # # plt.title(f"Training data of flight trajectories from {ADEP_code} to {ADES_code}")
        # # plt.legend(loc="upper right")

        # Add gridlines
        gridlines = ax.gridlines(
            draw_labels=True, color="gray", alpha=0.5, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False

        # title
        ax.set_title(f"Class {cluster}")

    # if the save_path contains "synthetic" (regarless of capitals or not), then we are plotting the synthetic data, else we are plotting the real data
    if "synthetic" in save_path.lower():
        fig.suptitle(
            f"Class-conditional generation of flight trajectories from {ADEP_code} to {ADES_code}",
            fontsize=20,
        )
    else:
        fig.suptitle(
            f"Training data of flight trajectories from {ADEP_code} to {ADES_code}",
            fontsize=20,
        )

    plt.tight_layout()
    # fig.subplots_adjust(top=0.1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# def plot_clustering(
#     trajectories: Traffic,
#     geographic_extent: List[float],
#     save_path: str,
#     ADEP_code: str = "EHAM",
#     ADES_code: str = "LIMC",
# ) -> Figure:
#     """
#     Class-conditional generation of flight trajectories
#     """

#     # t_gmm = Traffic.from_file(trajectories_path)

#     n_clusters = 1 + trajectories.data.cluster.max()

#     colors = sns.color_palette("husl", n_clusters)

#     nb_cols = n_clusters
#     nb_lines = 1

#     fig, axs = plt.subplots(
#         nb_lines, nb_cols, figsize=(30, 8), subplot_kw={"projection": EuroPP()}
#     )

#     for cluster in range(n_clusters):
#         ax = axs[cluster]

#         trajectories.query(f"cluster == {cluster}").plot(
#             ax, alpha=0.2, color=colors[cluster], linewidth=1
#         )
#         ax.coastlines()
#         ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
#         ax.set_extent(geographic_extent)

#         # # Plot the origin and destination airports
#         airports[ADEP_code].point.plot(
#             ax, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
#         )
#         airports[ADES_code].point.plot(
#             ax, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
#         )

#         # # plt.title(f"Training data of flight trajectories from {ADEP_code} to {ADES_code}")
#         # # plt.legend(loc="upper right")

#         # Add gridlines
#         gridlines = ax.gridlines(
#             draw_labels=True, color="gray", alpha=0.5, linestyle="--"
#         )
#         gridlines.top_labels = False
#         gridlines.right_labels = False

#         # title
#         ax.set_title(f"Class {cluster}")

#     # if the save_path contains "synthetic" (regarless of capitals or not), then we are plotting the synthetic data, else we are plotting the real data
#     if "synthetic" in save_path.lower():
#         fig.suptitle(
#             f"Class-conditional generation of flight trajectories from {ADEP_code} to {ADES_code}",
#             fontsize=20,
#         )
#     else:
#         fig.suptitle(
#             f"Training data of flight trajectories from {ADEP_code} to {ADES_code}",
#             fontsize=20,
#         )

#     plt.tight_layout()
#     fig.subplots_adjust(top=0.1)
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.show()
#     # return fig


def plot_altitude(
    trajectories: Traffic,
    geographic_extent: List[float],
    save_path: str,
    ADEP_code: str = "EHAM",
    ADES_code: str = "LIMC",
) -> None:

 
    if any(keyword in save_path.lower() for keyword in ["opensky", "landing", "simulated"]):
        if "opensky" in save_path.lower():
            trajectories = trajectories.sample(1000)
        edgecolor = None
    else:
        edgecolor = "black"



    df = trajectories.data

    # Check and handle duplicate indices
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)


    fig, ax = plt.subplots(figsize=(9, 8))
    m = Basemap(
        projection="merc",
        llcrnrlat=geographic_extent[2],
        urcrnrlat=geographic_extent[3],
        llcrnrlon=geographic_extent[0],
        urcrnrlon=geographic_extent[1],
        lat_ts=20,
        resolution="i",
    )

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    # Convert latitude and longitude to x and y coordinates
    x, y = m(df["longitude"].values, df["latitude"].values)

    # Connect points with a line
    plt.plot(x, y, color="black", alpha=0.2, zorder=1)

    # Plot the points with altitude as hue and size
    sns.scatterplot(
        x=x,
        y=y,
        hue=df["altitude"],
        palette="viridis",
        size=df["altitude"],
        sizes=(20, 200),
        legend="brief",
        ax=ax,
        # edgecolor="black",
        edgecolor=edgecolor,
    )

    # Add color bar for altitude
    norm = plt.Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(
        []
    )  # This is necessary because of a Matplotlib bug when using scatter with norm.
    cbar = plt.colorbar(sm, ax=ax, aspect=30)
    cbar.set_label("Altitude (feet)")
    # set legend upper right
    plt.legend(loc="upper right")

    # Add title and labels
    # plt.title(f"Trajectories from {ADEP_code} to {ADES_code}")

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
