import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class AutomatedEDA:
    def __init__(self):
        self.data = None
        self.column_types = {}  # Dictionary to store column types

    def select_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        return file_path

    def load_data(self):
        file_path = self.select_file()
        file_type = file_path.split(".")[-1]

        if file_type == 'csv':
            self.data = pd.read_csv(file_path)
        elif file_type == 'xlsx':
            self.data = pd.read_excel(file_path)
        elif file_type == 'sql':
            # Implement code to connect to SQL database and fetch data
            pass

    def preprocess_data(self):
        # Handle missing values
        self.data.fillna(method='ffill', inplace=True)  # Forward fill missing values

        # Identify column types
        for column in self.data.columns:
            dtype = str(self.data[column].dtype)
            if dtype == 'object':
                self.column_types[column] = 'categorical'
                # Encode categorical features
                self.data[column] = LabelEncoder().fit_transform(self.data[column])
            elif dtype in ['int64', 'float64']:
                self.column_types[column] = 'numerical'
                # Scale numerical features
                self.data[column] = StandardScaler().fit_transform(self.data[column].values.reshape(-1, 1))

    def generate_visualization(self):
        for column, dtype in self.column_types.items():
            if dtype == 'categorical':
                self.generate_categorical_visualizations(column)
            elif dtype == 'numerical':
                self.generate_numerical_visualizations(column)

    def generate_categorical_visualizations(self, column):
        counts = self.data[column].value_counts()
        labels = counts.index
        values = counts.values

        # Bar Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=values)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'{column} Distribution')
        plt.show()

        # Pie Chart
        fig = px.pie(self.data, names=column, title=f'{column} Distribution')
        fig.show()

    def generate_numerical_visualizations(self, column):
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'{column} Distribution')
        plt.show()

        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(self.data[column])
        plt.xlabel(column)
        plt.title(f'{column} Distribution')
        plt.show()

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.generate_visualization()


# Run the AutomatedEDA tool
eda_tool = AutomatedEDA()
eda_tool.run()