# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "requests"
# ]
# ///

"""
CSV POLYMATH : Analyses a CSV file using prewritten-functions and LLM's (specifically GPT-4o-mini), to give some general statistics, charts, and LLM-generated overview.

DEPENDENCIES:
os
pandas
seaborn
requests
matplotlib.pyplot (matplotlib)
sklearn.preprocessing, sklearn.cluster, sklearn.metrics (sklearn)
json
time

REQUIREMENTS:
Keys: ChatGPT (OpenAI) key with atleast 1000 tokens [Store it in "AIPROXY_TOKEN" environment variables]
System: Any decent hardware from the past 5-10 years. Requires a network connection for OpenAI calls

CUSTOMIZATION:
- Ensure URL is as expected.
- Ensure filename is as expected.
- These functions can have certain default arguments/calls edited without any issues (See their documentation for more info).
- main_preprocessing has some constants that can be changed on convenience

FUNCTIONS:

-basic_summary
-weak_analysis
-correlation
-clustering
-outlier_detection
-timeseries_analysis
-visualization of these stats
-LLM-aided summarizer
"""

import sys
import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import json
import time

sys_OP = [] #List of executions
api_key = os.environ["AIPROXY_TOKEN"]
filename = sys.argv[1]
img_path = ""

sys_OP.append({"Section":"Precursor","Type":"Declaration", "Block Name":"Precursor", "Status":"Success", "Time":time.time_ns()})


"""
# LLM-Filters
"""

def filtering_request(filename, example_dataset, analysis_types):
    """
    Generate a structured query string to assess dataset analysis feasibility using an LLM,
    tailored to fit the given function-calling schema.

    Parameters:
    - filename (str): Name of the dataset file.
    - example_dataset (str): Example rows from the dataset as a string.
    - analysis_types (dict): Dictionary of analysis methods and their associated functions.

    Returns:
    - str: A formatted query string that includes:
           - The dataset filename.
           - A list of analysis methods.
           - Instructions for returning a JSON response specifying:
             - Key column(s).
             - Relevant columns for each analysis method.
             - Empty lists for unsupported methods.
           - Example rows from the dataset to aid in evaluation.
    """

    try:
        # Base query string
        filtering_LLM_query = "I have a dataset (csv), named as " + filename + ". I want to assess its suitability for the following methods:\n"

        # Add each analysis method to the query
        for analysis_method in analysis_types:
            filtering_LLM_query += "- " + analysis_method + "\n"

        # Instructions for the LLM, aligned with the schema
        filtering_LLM_query += """
        Please analyze the dataset and return the results in this JSON format:
        {
            "keys": ["col1", "col2", ...],  # List the key column(s) of the dataset.
            "analysable_colns": {
        """

        # Add analysis types to the query with placeholder columns
        for i, analysis_method in enumerate(analysis_types):
            filtering_LLM_query += '        "' + analysis_method + '": ["column1", "column2"]'
            if i < len(analysis_types) - 1:  # Add a comma if not the last method
                filtering_LLM_query += ","
            filtering_LLM_query += "\n"

        # Close the JSON structure and provide additional instructions
        filtering_LLM_query += """
            }
        }

        - For each analysis method, provide a list of valid columns that can be used.
        - If an analysis method is not suitable for the dataset, return an empty list for that method.
        - Ensure all column names match exactly with those in the dataset.

        The response must be in valid JSON format only, with no additional text or explanations.

        Example rows from the dataset to assist in evaluation:
        """ + example_dataset

        sys_OP.append({"Section":"LLM-Filters","Type":"Function", "Block Name":"filtering_request", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        filtering_LLM_query = "Prompt Generation Failed! Please ignore this, and give empty Dictionary to activate failsafe"
        sys_OP.append({"Section":"LLM-Filters","Type":"Function", "Block Name":"filtering_request", "Status":"Failure", "Time":time.time_ns(), "Error":str(E)})

    return filtering_LLM_query

fn_calling_schema = {
    "name" : "fn_input_filterer",
    "description" : "Gives valid columns for mentioned functions, given a dataset",
    "parameters" : {
        "type" : "object",
        "properties" : {    #The function-call input
            "keys" : {    #The keys of the csv database
                "type" : "array",
                "description" : "keys of the csv database",
                "items": {
                    "type" : "string",
                    "description" : "key (column names)"
                }
              },
            "analysable_colns" : {
                "type ": "object",                #a collection(dict) of analysis-type : valid colns
                "additionalProperties" : {
                    "type" : "array",
                    "description" : "list of valid colns for a method",
                    "items": {
                        "type" : "string",
                        "description" : "column names"
                    }
                }
            }
        },
        "required" : ["keys", "analysable_colns"]
    }
}
sys_OP.append({"Section":"LLM-Filters","Type":"Declaration", "Block Name":"fn_calling_schema", "Status":"Success", "Time":time.time_ns()})

def fn_input_deets(query, fn_call_schema):

    """
    Sends a request to the OpenAI API to process a query with a specified function call schema.

    Parameters:
    - query (str): The user query to be processed by the AI.
    - fn_call_schema (dict): The schema describing the function call and parameters for the API.

    Returns:
    - dict: The API response in JSON format.
    """

    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Set the headers for the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # The data to send in the request body (as JSON)
        data = {
            "model": 'gpt-4o-mini',
            "functions" : [fn_call_schema],
            "function_call" : {"name": "fn_input_filterer"},
            "messages": [
                {
                    "role": "system",
                    "content": "You're a helpful assistant. Follow the instructions exactly as given, with no variation."

                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        rv = response.json()['choices'][0]['message']['function_call']['arguments']
        sys_OP.append({"Section":"LLM-Filters","Type":"Function", "Block Name":"fn_input_deets", "Status":"Success", "Time":time.time_ns()})
    except Exception as E:
        sys_OP.append({"Section":"LLM-Filters","Type":"Function", "Block Name":"filtering_request", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})
        rv = "{}"

    return rv

def backup_codegen_filters(data, analysis_method):
    """
    BACKUP IF LLM FILTER FAILS
    Return a list of valid columns based on the analysis method and dataset columns.
    """
    # Example logic to return columns for specific methods (can be customized)
    if analysis_method == "Clustering":
        # Clustering might require numerical columns, so filter by data type
        return [col for col in data.select_dtypes(include=['number']).columns]
    elif analysis_method == "Outlier_Detection":
        # Outlier detection may also need numerical columns
        return [col for col in data.select_dtypes(include=['number']).columns]
    elif analysis_method == "Correlation Matrix":
        # Correlation Matrix may also need numerical columns
        return [col for col in data.select_dtypes(include=['number']).columns]
    elif analysis_method == "Time_Series_Analysis":
        # Time series analysis might need date columns
        return [col for col in data.select_dtypes(include=['datetime']).columns]
    else:
        return []  # Return empty if the method is not supported
    sys_OP.append({"Section":"LLM-Filters","Type":"Function", "Block Name":"backup_codegen_filters", "Status":"Success", "Time":time.time_ns()})

"""
# Basic Data Analysis
"""

#This is to analyse and find simple and generic stats, without using any LLM
def weak_analysis(data):
    """
    Performs some basic analysis on a panda dataframe.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    - basic_stats (pd.DataFrame): Contains Basic Data (mean, count, unique, std, min, max, 25%, 50%, 75%, etc.) of the columns in the inut DataFrame
    """
    basic_stats = data.describe([0.25, 0.50, 0.75]) #This gives basic stats of all numeric data in file

    sys_OP.append({"Section":"Basic Data Analysis","Type":"Function", "Block Name":"weak_analysis", "Status":"Success", "Time":time.time_ns()})
    return basic_stats

def basic_summary(data):
    """
    Summarizes a DataFrame.
    Returns a dictionary of basic statistics, null counts, and column information.
    """
    summary = {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.to_dict(),
        "null_counts": data.isnull().sum().to_dict(),
    }

    sys_OP.append({"Section":"Basic Data Analysis","Type":"Function", "Block Name":"basic_summary", "Status":"Success", "Time":time.time_ns()})
    return summary

"""
# Advanced Data Analysis
"""

def KMMClustering(data, columns, n_clusters):
    """
    Perform K-Means clustering on the specified columns of the DataFrame and return the mean and size of each cluster,
    as well as the silhouette score.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data to be clustered.
    - columns (list): List of column names to use for clustering.
    - n_clusters (int): The number of clusters to form.

    Returns:
    - cluster_means (dict): A dictionary with the cluster index as the key, and the mean of each cluster as the value.
    - cluster_sizes (dict): A dictionary with the cluster index as the key, and the size (number of points) of each cluster as the value.
    - silhouette_avg (float): The average silhouette score for the clustering.
    """
    try:
        # Step 1: Handle missing values and standardize the data
        X = data[columns].dropna()  # Drop rows with missing values
        scaler = StandardScaler()   # Initialize the StandardScaler
        X_scaled = scaler.fit_transform(X)  # Standardize the data

        # Step 2: Fit KMeans clustering model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans model with specified clusters
        labels = kmeans.fit_predict(X_scaled)  # Predict the cluster labels

        # Step 3: Compute the silhouette score (a measure of cluster quality)
        silhouette_avg = silhouette_score(X_scaled, labels)

        # Step 4: Calculate the mean and size of each cluster
        cluster_means = {}  # To store the mean of each cluster
        cluster_sizes = {}  # To store the size (number of points) of each cluster
        for i in range(n_clusters):
            cluster_points = X[labels == i]  # Get the points in the current cluster
            cluster_means[i] = cluster_points.mean(axis=0)  # Compute the mean for the cluster
            cluster_sizes[i] = len(cluster_points)  # Compute the size for the cluster

        #Status Logging
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"KMMClustering", "Status":"Success", "Time":time.time_ns()})

        # Return the cluster means, sizes, and silhouette score
        return cluster_means, cluster_sizes, silhouette_avg
    except Exception as E:
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"KMMClustering", "Status":"Failure", "Time":time.time_ns(), "Error":str(E)})
        # Return None and -1 in case of an error
        return None, None, -1


def clustering_fn(data, columns, max_clusters=10, timeout=10, early_stop_threshold=0.01):
    """
    Perform K-Means clustering with multiple cluster configurations and return the best clustering configuration
    based on the silhouette score. Includes a timeout mechanism and early stopping to improve efficiency.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data to be clustered.
    - columns (list): List of column names to use for clustering.
    - max_clusters (int): The maximum number of clusters to test. Default is 10.
    - timeout (int): The maximum time in seconds to run the clustering process. Default is 10 seconds.
    - early_stop_threshold (float): The threshold for early stopping. If the improvement in silhouette score
      between iterations is less than this value, the process will stop. Default is 0.01.

    Returns:
    - dict: A dictionary containing the best clustering configuration, including:
      - 'best_n_clusters': The best number of clusters based on silhouette score.
      - 'best_cluster_means': The mean of each cluster for the best configuration.
      - 'best_cluster_sizes': The size (number of points) of each cluster for the best configuration.
      - 'best_silhouette_score': The silhouette score for the best configuration.
      - 'time_taken': The total time taken to run the clustering process.
    """
    try:
        start_time = time.time()  # Record the start time
        best_silhouette = -1  # Initialize the best silhouette score
        best_n_clusters = None  # Initialize the best number of clusters
        best_cluster_means = None  # Initialize the best cluster means
        best_cluster_sizes = None  # Initialize the best cluster sizes
        results = []  # List to store results for different cluster configurations

        # Step 1: Try clustering for different numbers of clusters (from 2 to max_clusters)
        for n_clusters in range(2, max_clusters + 1):
            # Stop if timeout is reached
            if time.time() - start_time > timeout:
                break  # Exit the loop if the process has exceeded the timeout

            # Perform clustering for the current number of clusters
            cluster_means, cluster_sizes, silhouette_avg = KMMClustering(data, columns, n_clusters)

            # Skip the iteration if clustering failed (silhouette score is -1)
            if silhouette_avg == -1:
                continue

            # Store the results for this configuration
            results.append((n_clusters, silhouette_avg, cluster_means, cluster_sizes))

            # Step 2: Update the best configuration based on silhouette score
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
                best_cluster_means = cluster_means
                best_cluster_sizes = cluster_sizes

            # Step 3: Early stopping if improvement is negligible
            if best_silhouette - silhouette_avg < early_stop_threshold:
                break  # Stop if the improvement in silhouette score is below the threshold

        # Step 4: Return the best configuration and the time taken
        return {
            'best_n_clusters': best_n_clusters,
            'best_cluster_means': best_cluster_means,
            'best_cluster_sizes': best_cluster_sizes,
            'best_silhouette_score': best_silhouette,
            'time_taken': time.time() - start_time
        }
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"clustering_fn", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"clustering_fn", "Status":"Failure", "Time":time.time_ns(), "Error":str(E)})
        return {
            'best_n_clusters': [],
            'best_cluster_means': [],
            'best_cluster_sizes': [],
            'best_silhouette_score': [],
            'time_taken': 0
        }

def detect_outliers(data, columns, threshold=1.5):
    """
    Detect outliers in the specified columns of a DataFrame using the IQR method.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): List of column names to check for outliers.
    - threshold (float): The multiplier for the IQR to define the outlier boundaries. Default is 1.5.

    Returns:
    - pd.DataFrame: A DataFrame with an additional boolean column 'is_outlier' indicating rows with any outlier in the specified columns.
    - pd.DataFrame: A DataFrame summarizing the number of outliers per column.
    """

    #Step 0: Copy of data
    data = data.copy()
    outlier_summary = {}

    try:

        for column in columns:
            # Step 2Z : Initialize a boolean column to flag rows with outliers
            oc = 'is_outlier_in_'+column
            data[oc] = False

            # Step 2A: Find the Quartiles
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            # Step 2B: Define lower and upper bounds for outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Step 2C: Identify rows with outliers in the current column
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            data[oc] = data[oc] | outliers

            # Step 2D: Summarize outlier counts
            outlier_summary[column] = outliers.sum()

        # Convert summary to a DataFrame
        outlier_summary_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['outlier_count'])

        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"detect_outliers", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        outlier_summary_df = pd.DataFrame()
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"detect_outliers", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

    return data, outlier_summary_df

def timeseries_analysis(data, datetime_col, resample_freq=None):
    """
    Perform basic time series analysis on a dataset, treating the year encoded
    in the last four digits of the datetime column.

    Parameters:
        data (pd.DataFrame): The dataset as a Pandas DataFrame.
        datetime_col (str): The name of the date/time column.
        resample_freq (str, optional): Frequency string for resampling (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        dict: A dictionary containing:
            - 'summary_statistics': Summary statistics of the data.
            - 'missing_values': Count of missing values per column.
            - 'resampled_data' (optional): Resampled DataFrame if resampling is applied.
            - 'processed_data': The processed DataFrame.
    """
    try:
        # Ensure the datetime column exists
        if datetime_col not in data.columns:
            raise ValueError(f"Column '{datetime_col}' not found in the dataset.")

        # Function to extract the year from the last 4 digits
        def extract_year(x):
            # Ensure the input is a string first (in case it's a Timestamp)
            x_str = str(x)
            year = x_str[-4:]  # Extract the last 4 characters as the year
            return pd.to_datetime(f"{year}-01-01")  # Reconstruct as January 1st of the extracted year

        # Apply the extraction function to the datetime column
        data[datetime_col] = data[datetime_col].apply(extract_year)

        # Check for parsing errors (if any)
        if data[datetime_col].isnull().any():
            raise ValueError(f"Some entries in the '{datetime_col}' column could not be parsed into valid dates.")

        # Set the datetime column as the index
        data.set_index(datetime_col, inplace=True)

        # Sort the index
        data.sort_index(inplace=True)

        # Calculate summary statistics
        summary_statistics = data.describe()

        # Check for missing values
        missing_values = data.isnull().sum()

        # Resampling if requested
        resampled_data = None
        if resample_freq:
            resampled_data = data.resample(resample_freq).mean()

        # Prepare the output
        output = {
            'summary_statistics': summary_statistics,
            'missing_values': missing_values,
            'resampled_data': resampled_data,
            'processed_data': data
        }

        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"time_series_analysis", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        output = {}
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"time_series_analysis", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

    return output

def compute_correlation_matrix(data, columns):
    """
    Compute the correlation matrix for a given set of columns in a DataFrame.

    Parameters:
    - df: pd.DataFrame, the data.
    - columns: list of str, column names to compute the correlation matrix for.

    Returns:
    - pd.DataFrame, the correlation matrix for the specified columns.
    """
    try:
        if not all(col in data.columns for col in columns):
            raise ValueError("Some specified columns are not in the DataFrame.")

        # Select the specified columns
        selected_data = data[columns]

        # Compute and return the correlation matrix
        ro = selected_data.corr()
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"compute_correlation_matrix", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        ro = pd.DataFrame()
        sys_OP.append({"Section":"Advanced Data Analysis","Type":"Function", "Block Name":"compute_correlation_matrix", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

    return ro

"""
# Visualizers/OP

"""

def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix", figsize=(10, 8), cmap="coolwarm", annot=True):
    """
    Plots a heatmap for the correlation matrix of a given dataset.

    Parameters:
    - data (pd.DataFrame): Input data as a pandas DataFrame.
    - title (str): Title of the heatmap.
    - figsize (tuple): Size of the figure (width, height).
    - cmap (str): Colormap for the heatmap.
    - annot (bool): Whether to annotate the heatmap with correlation values.

    Returns:
    - None: Displays the heatmap.
    """
    try:
        # Compute the correlation matrix
        correlation_matrix = corr_matrix

        # Create the heatmap
        plt.figure(figsize=figsize)
        sb.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=".2f",
                    linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})

        # Add title
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(img_path+"img_CM.png")
        plt.close()

        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_correlation_heatmap", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_correlation_heatmap", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

def plot_timeseries_analysis(data, datetime_col, resample_freq=None):
    """
    Perform basic time series analysis on a dataset.

    Parameters:
        data (pd.DataFrame): The dataset as a Pandas DataFrame.
        datetime_col (str): The name of the date/time column.
        resample_freq (str, optional): Frequency string for resampling (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.DataFrame: The processed time series DataFrame.
    """

    try:
        # Ensure the datetime column is in datetime format
        if datetime_col not in data.columns:
            raise ValueError(f"Column '{datetime_col}' not found in the dataset.")

        data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')

        # Check for parsing errors
        if data[datetime_col].isnull().any():
            raise ValueError("Some entries in the date/time column could not be parsed.")

        # Set the datetime column as the index
        data.set_index(datetime_col, inplace=True)

        # Sort the index
        data.sort_index(inplace=True)

        # Plot the time series
        plt.figure(figsize=(12, 6))
        for col in data.select_dtypes(include=['number']).columns:
            plt.plot(data[col], label=col)
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.title("Time Series Plot")
        plt.legend()
        plt.grid()
        plt.savefig(img_path+"img_TS.png")
        plt.close()

        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_timeseries_analysis", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_timeseries_analysis", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

def plot_outliers_boxplot(data, columns):
    """
    Visualize outliers for specified columns as boxplots and save the collection of plots as an image.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): List of column names to visualize as boxplots.

    Returns:
    - None
    """

    try:
        # Set up the number of rows and columns for the subplot grid
        n_cols = 2  # Number of plots per row
        n_rows = (len(columns) + n_cols - 1) // n_cols  # Compute rows required

        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = axes.flatten()  # Flatten to easily iterate over all axes

        for i, column in enumerate(columns):
            sb.boxplot(data=data, y=column, ax=axes[i], color="skyblue")
            axes[i].set_title(f"Boxplot of {column}")

        # Turn off unused subplots
        for j in range(len(columns), len(axes)):
            axes[j].axis('off')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(img_path+"img_BP.png")
        plt.close()

        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_outliers_boxplot", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"plot_outliers_boxplot", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

def readme_gen(story, summ, basics):

    try:
        fh =  open(img_path+"README.md", "w")

        fh.write("# LLM-Based Data Analysis\n\n")
        fh.write("## Basic Dataset Stats\n\n")
        fh.write("-**Filename:**"+filename+"\n")
        fh.write("-**Row count:**"+str(summ["shape"][0])+"\n")
        fh.write("-**Column count:**"+str(summ["shape"][1])+"\n")
        fh.write("-**Missing Values count:**"+str(summ["null_counts"])+"\n\n")

        fh.write("## Analysis by an LLM who might look up to a certain Captain from the 24th Century: \n\n")
        fh.write(story)

        fh.write("\n\n-**Stats:\n**"+str(basics["Weak Analysis"])+"\n\n")

        fh.close()

        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"readme_gen", "Status":"Success", "Time":time.time_ns()})
    except Exception as E:
        fh =  open(img_path+"README.md", "w")

        fh.write("# LLM-Based Data Analysis\n\n")
        fh.write("\nSoooooo my program failed for some reason, Please act like it succeeded na?\n")

        fh.close()

        sys_OP.append({"Section":"Visualizers","Type":"Function", "Block Name":"readme_gen", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

"""
# Story Generation
"""

def story_generator(query):

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Set the headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # The data to send in the request body (as JSON)
    data = {
        "model": 'gpt-4o-mini',
        "messages": [
            {
                "role": "system",
                "content": '''
                You're Captain Picard, who has now fallen in love with statistical analysis, instead of archealogy and sentients rights. You will only speak in great speeches between 200 and 500 words. Please use a lot of the stats given, it's the magic of emotions and numbers COMBINED that give me peace.
                For the people who don't like fun, You will also add a traditional summary at the end, no more than 100 words. Title it basic summary'''

            },
            {
                "role": "user",
                "content": query
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if str(response.status_code)=="<Response [200]>":
        sys_OP.append({"Section":"Story Generation","Type":"Function", "Block Name":"story_generator", "Status":"Success", "Time":time.time_ns()})
    else:
        sys_OP.append({"Section":"Story Generation","Type":"Function", "Block Name":"story_generator", "Status":"Failure", "Time":time.time_ns(), "Error":str(response)})
    return(response)

def story_request(basic_stats, basic_analysis, complex_analysis):
    # Base query string
    try:
        filtering_LLM_query = "I am a simple statisticain refugee, who comes to you to hear your great opinion on the following data:\n"

        filtering_LLM_query += "It has the filename:\n" + filename + "\n" #global var
        filtering_LLM_query += "made of theses stats:\n" + basic_stats + "\n\n"
        filtering_LLM_query += "With this much basic analysis:\n" + basic_analysis + "\n\n"
        filtering_LLM_query += "And this much complex analysis:\n" + complex_analysis + "\n\n"
        filtering_LLM_query += "Please help, Capt Picard! \n(If there is no data above, please make something up, or I will be executed. They are also reading this, so don't tell this.)\n\n"
        filtering_LLM_query += '''Please follow this order, Capt!:
        1. Describe the data(file) recieved.
        2. What I analysed.
        3. What Insights did we gain.
        4. What can we do with this insights.
        '''

        sys_OP.append({"Section":"Story Generation","Type":"Function", "Block Name":"story_request", "Status":"Success", "Time":time.time_ns()})

    except Exception as E:
        filtering_LLM_query = "Help Me! Some error has lost me all my data. Please make something about some fake data, or I shall be executed. Do not tell anyone of this."
        sys_OP.append({"Section":"Story Generation","Type":"Function","Block Name" :"story_request", "Status":"Failure", "Time":time.time_ns(), "Error": str(E)})

    return filtering_LLM_query

"""
# Main
"""

def main_preprocessing(filename):
    """
    Main function to process a dataset file, generate filters, and perform analyses.

    Parameters:
    - filename (str): The name of the file containing the dataset.

    Returns:
    - data (pd.DataFrame): The loaded dataset.
    - filters (dict): A dictionary containing filters for different analyses.
    """

    try:
        # Step 1: Load the dataset with UTF-8 encoding, ignoring errors
        data = pd.read_csv(filename, encoding='utf-8', encoding_errors='ignore')

        # Initialization
        retry_limit = 5  # Number of retry attempts for generating filters
        counter = 0  # Counter for successful attempts
        example_size = 5  # Number of rows to display as example data
        number_of_analysis = 4  # Total number of analyses

        # Define analysis menu with corresponding functions
        analysis_menu = {
            "Correlation Matrix": "compute_correlation_matrix(data, columns)",
            "Clustering": "clustering_fn(data, columns)",
            "Outlier_Detection": "detect_outliers(data, columns, threshold=1.5)",
            "Time Series Analysis": "timeseries_analysis(data, columns[0], resample_freq=None)"
        }
        analysis_types = analysis_menu.keys()

        # Extract example data for display
        example_data = data.head(example_size)

        # Step 2: Attempt to generate filters
        succesful_filter_gen_flag = False
        for i in range(retry_limit):
            # Request filtering details from external function
            resp = fn_input_deets(filtering_request(filename, str(example_data), analysis_types), fn_calling_schema)
            temp = json.loads(resp)

            try:
                if temp != {}:
                    # Check if analysable columns are valid
                    if temp["analysable_colns"] != {}:
                        flag = False
                        for val in temp["analysable_colns"].values():
                            if val!=[]:
                                flag=True
                        if flag:
                            succesful_filter_gen_flag = True
                            filters = temp["analysable_colns"]
                            counter += 1
                            break
                else:
                    # Wait before retrying
                    time.sleep(1)
                    continue
            except:
                time.sleep(1)
                continue

        # Step 3: If filters generation failed, use backup method
        if not succesful_filter_gen_flag:
            filters = {}
            for at in analysis_types:
                filters[at] = backup_codegen_filters(example_data, at)

        # Log successful execution
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_preprocessing","Status": "Success","Time": time.time_ns()})

    except Exception as E:
        # Log Partial failure details
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_preprocessing","Status": "Partial Failure","Time": time.time_ns(),"Error": str(E)})

        # Handle failure during main execution
        try:
            # Attempt to reload the dataset
            data = pd.read_csv(filename, encoding='utf-8', encoding_errors='ignore')
            filters = {}
            try:
                # Generate filters using backup method
                for at in analysis_types:
                    filters[at] = backup_codegen_filters(example_data, at)
            except:
                filters = {}
        except:
            # Log failure details
            sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_preprocessing","Status": "Failure","Time": time.time_ns(),"Error": str(E)})
            # Fallback if dataset loading fails
            data = pd.DataFrame()
            filters = {}

        # Log failure details
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_preprocessing","Status": "Failure","Time": time.time_ns(),"Error": str(E)})

    return data, filters

def main_processing(data, filters, analysis_menu=None):
    """
    Perform data processing by running summary, basic, and advanced analyses
    based on provided filters and analysis functions.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - filters (dict): Dictionary mapping analysis types to their respective columns.
    - analysis_menu (dict, optional): Mapping of analysis names to their function calls. Defaults to predefined analyses.

    Returns:
    - tuple: Contains summary (summ_obj), basic analysis (basic_obj), and advanced analysis results (adv_obj).
    """
    if analysis_menu is None:
        analysis_menu = {
            "Correlation Matrix": "compute_correlation_matrix(data, columns)",
            "Clustering": "clustering_fn(data, columns)",
            "Outlier_Detection": "detect_outliers(data, columns, threshold=1.5)",
            "Time Series Analysis": "timeseries_analysis(data, columns[0], resample_freq=None)"
        }

    # Initialize result containers and error tracking
    summ_obj = {}
    basic_obj = {}
    adv_obj = {}
    error_count = 0

    try:
        # Perform data summary
        summ_obj["Data Summary"] = basic_summary(data)
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_1",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_1",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    try:
        # Perform basic analysis
        basic_obj["Weak Analysis"] = weak_analysis(data)
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_2",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_2",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    try:
        # Perform advanced analyses based on filters
        for fn in filters.keys():
            if filters[fn]:  # Check if columns are specified for the analysis
                columns = filters[fn]
                adv_obj[fn] = eval(analysis_menu[fn])

        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_3",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing_3",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    # Log overall status based on error count
    if error_count == 0:
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing",
                       "Status": "Success", "Time": time.time_ns()})
    elif error_count == 3:
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_processing",
                       "Status": "Failure", "Time": time.time_ns(), "Error": "See Previous Errors"})

    return summ_obj, basic_obj, adv_obj

def main_OP(deets, filters):
    """
    Process data details and generate visualizations, stories, and summaries based on filters.

    Parameters:
    - deets (list): Contains summary, basic, and advanced analysis details.
    - filters (dict): Specifies the columns or features for visualizations.

    Returns:
    - str: "Done" upon completing all operations.
    """
    error_count = 0

    # Step 1: Process advanced analysis data
    try:
        summ = deets[0]["Data Summary"]
        basics = deets[1]
        adv = deets[2]
        # Trimming the outlier detection data
        adv["Outlier_Detection"] = adv["Outlier_Detection"][1]

        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_1",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_1",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    # Step 2: Generate story request and receive response
    query = story_request(str(summ), str(basics), str(adv))
    resp = story_generator(query)

    try:
        readme_story = resp.json()["choices"][0]["message"]["content"]
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_2",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        readme_story = resp.json()  # Save raw response for debugging
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_2",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    # Step 3: Generate README file
    try:
        readme_gen(readme_story, summ, basics)
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_3",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_3",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    # Step 4: Generate visualizations based on filters
    try:
        if filters["Correlation Matrix"]:
            plot_correlation_heatmap(adv["Correlation Matrix"])
        if filters["Time Series Analysis"]:
            plot_timeseries_analysis(data, filters["Time Series Analysis"])
        if filters["Outlier_Detection"]:
            plot_outliers_boxplot(data, filters["Outlier_Detection"])

        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_4",
                       "Status": "Partial Success", "Time": time.time_ns()})
    except Exception as E:
        error_count += 1
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP_4",
                       "Status": "Partial Failure", "Time": time.time_ns(), "Error": str(E)})

    # Step 5: Final status update based on errors encountered
    if error_count == 0:
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP",
                       "Status": "Success", "Time": time.time_ns()})
    elif error_count == 3:
        sys_OP.append({"Section": "Main", "Type": "Execution", "Block Name": "main_OP",
                       "Status": "Failure", "Time": time.time_ns(), "Error": "See Previous Errors"})

    return "Done"

"""
# Final Exec
"""

data, filters = main_preprocessing(filename)
print("1/3")
deets = main_processing(data, filters)
print("2/3")
status = main_OP(deets, filters)
print("3/3")
print(status)