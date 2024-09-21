import logging
import os
from math import atan2, cos, radians, sin, sqrt
import random
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from traffic.core import Traffic

from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# class TrajectoryDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y

#         self._len = self.X.shape[0]

#     def __len__(self):
#         return self._len

#     def __getitem__(self, idx):
#         x, y = self.X[idx], self.Y[idx]
#         # x = x[None, :] # adds a channel dimension
#         return x, y


# def get_data(dataset_file: str, features: list, batch_size: int):
#     traffic = Traffic.from_file(dataset_file)
#     data = np.stack(list(f.data[features].values.ravel() for f in traffic))
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(data)
#     data = scaler.transform(data)
#     # check that each flight has a unique cluster
#     for flight in traffic:
#         if flight.data.cluster.nunique() != 1:
#             raise ValueError("Each flight should have a unique cluster")
#     labels = np.array([f.data.cluster.iloc[0] for f in traffic])
#     le = LabelEncoder()
#     labels = le.fit_transform(labels.ravel())[:, None]
#     data = torch.FloatTensor(data)
#     data = data.view(data.size(0), -1, len(features))
#     data = torch.transpose(data, 1, 2)
#     split_idx = int(0.9 * len(data))
#     X_train, X_test = data[:split_idx], data[split_idx:]
#     Y_train, Y_test = labels[:split_idx], labels[split_idx:]

#     train_data_loader = DataLoader(
#         TrajectoryDataset(X_train, Y_train),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#     )
#     test_data_loader = DataLoader(
#         TrajectoryDataset(X_test, Y_test),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#     )

#     return train_data_loader, test_data_loader, scaler



class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._len = self.X.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        return x, y

def get_data(dataset_file: str, features: list, batch_size: int, train_ratio: float = 0.9, random_seed: int = 42):
    # Load the data
    traffic = Traffic.from_file(dataset_file)
    data = np.stack([f.data[features].values.ravel() for f in traffic])
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    # Ensure each flight has a unique cluster
    for flight in traffic:
        if flight.data.cluster.nunique() != 1:
            raise ValueError("Each flight should have a unique cluster")
    
    # Create labels
    labels = np.array([f.data.cluster.iloc[0] for f in traffic])
    le = LabelEncoder()
    labels = le.fit_transform(labels.ravel())[:, None]

    # Convert data and labels to PyTorch tensors
    data = torch.FloatTensor(data)
    data = data.view(data.size(0), -1, len(features))
    data = torch.transpose(data, 1, 2)
    labels = torch.LongTensor(labels)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate shuffled indices
    indices = np.random.permutation(len(data))

    # Calculate the split index
    split_idx = int(train_ratio * len(data))

    # Split data into training and test sets
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train, X_test = data[train_indices], data[test_indices]
    Y_train, Y_test = labels[train_indices], labels[test_indices]

    # Create DataLoader for train and test sets
    train_data_loader = DataLoader(
        TrajectoryDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_data_loader = DataLoader(
        TrajectoryDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_data_loader, test_data_loader, scaler




def parse_runtime_env(filename):
    with open(file=filename, mode="r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def export_environment_variables(runtime_env_filename):
    runtime_env = parse_runtime_env(runtime_env_filename)
    for key, value in runtime_env["env_vars"].items():
        os.environ[key] = value


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


# def calculate_consecutive_distances(df, distance_threshold):
#     """Calculates distances between consecutive points and flags flights with any excessive distance."""
#     # Calculate distances for each point to the next within each flight
#     df = df.sort_values(["flight_id", "timestamp"])
#     df["next_latitude"] = df.groupby("flight_id")["latitude"].shift(-1)
#     df["next_longitude"] = df.groupby("flight_id")["longitude"].shift(-1)

#     # Apply the Haversine formula
#     df["segment_distance"] = df.apply(
#         lambda row: (
#             haversine(
#                 row["latitude"],
#                 row["longitude"],
#                 row["next_latitude"],
#                 row["next_longitude"],
#             )
#             if not pd.isna(row["next_latitude"])
#             else 0
#         ),
#         axis=1,
#     )

#     # Find flights with any segment exceeding the threshold
#     outlier_flights = df[df["segment_distance"] > distance_threshold][
#         "flight_id"
#     ].unique()
#     return outlier_flights

# import numpy as np

def calculate_consecutive_distances(df, distance_threshold):
    """Calculates distances between consecutive points and flags flights with any excessive distance."""

    df = df.sort_values(["flight_id", "timestamp"])

    # Shift coordinates to get next point's coordinates
    df["next_latitude"] = df.groupby("flight_id")["latitude"].shift(-1)
    df["next_longitude"] = df.groupby("flight_id")["longitude"].shift(-1)

    # Convert to radians using NumPy
    lat1 = np.radians(df["latitude"])
    lon1 = np.radians(df["longitude"])
    lat2 = np.radians(df["next_latitude"])
    lon2 = np.radians(df["next_longitude"])

    # Vectorized Haversine calculation
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["segment_distance"] = 6371 * c  # Earth's radius in km

    # Find outlier flights
    outlier_flights = df[df["segment_distance"] > distance_threshold][
        "flight_id"
    ].unique()
    
    return outlier_flights

def calculate_initial_distance(df, origin_lat_lon, distance_threshold):
    """Calculates distances between the first point in each flight and the origin airport."""
    # Calculate distances from the origin airport to the first point of each flight

    # first point of each flight
    first_points = df.groupby("flight_id").first()
    # Calculate distances from the origin airport to the first point of each flight
    first_points["initial_distance"] = [
        haversine(lat, lon, origin_lat_lon[0], origin_lat_lon[1])
        for lat, lon in zip(first_points["latitude"], first_points["longitude"])
    ]

    # Find flights with the first point exceeding the threshold
    outlier_flights = first_points[
        first_points["initial_distance"] > distance_threshold
    ].index
    return outlier_flights


def calculate_final_distance(df, destination_lat_lon, distance_threshold):
    """Calculates distances between the last point in each flight and the destination airport."""
    # Calculate distances from the destination airport to the last point of each flight

    # last point of each flight
    last_points = df.groupby("flight_id").last()
    # Calculate distances from the destination airport to the last point of each flight
    last_points["final_distance"] = [
        haversine(lat, lon, destination_lat_lon[0], destination_lat_lon[1])
        for lat, lon in zip(last_points["latitude"], last_points["longitude"])
    ]

    # Find flights with the last point exceeding the threshold
    outlier_flights = last_points[
        last_points["final_distance"] > distance_threshold
    ].index
    return outlier_flights




