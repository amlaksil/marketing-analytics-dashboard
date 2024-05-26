#!/usr/bin/python3
"""
This module contains a class that performs ExploratoryDataAnalysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns


class ExploratoryDataAnalysis:
    """
    A class for conducting exploratory data analysis (EDA) on a
    relational database.

    Attributes:
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine for
        connecting to the database.
    """
    def __init__(self, db_url):
        """
        Initializes the ExploratoryDataAnalysis class with a database URL.

        Args:
            db_url (str): The URL for connecting to the relational database.
        """
        self.engine = create_engine(db_url)

    def load_data(self, table_name):
        """
        Loads data from a specified table in the database.

        Args:
            table_name (str): The name of the table to load data from.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the
        specified table.
        """
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)

    def data_summarization(self, df):
        """
        Perform data summarization on the DataFrame.

        Args:
        - df (pandas.DataFrame): DataFrame to summarize.

        Returns:
        - descriptive_stats (pandas.DataFrame): A DataFrame containing
        descriptive statistics of the data.
        - data_structure (pandas.Series): A Series containing the data types
        of each column in the DataFrame.
        """
        descriptive_stats = df.describe()

        data_structure = df.dtypes

        return descriptive_stats, data_structure

    def summarize_all_tables(self):
        """
        Load and summarize all tables in the database.

        Returns:
        - tables (dict): A dictionary containing summaries for each table.
        Keys are table names, and values are dictionaries with the following
        structure:
        {
            "descriptive_stats": pandas.DataFrame,
            "data_structure": pandas.Series
        }
        """
        tables = {
            "app_download_data": None,
            "google_play_store_reviews": None,
            "telegram_post_performance": None,
            "telegram_subscription_growth": None
        }

        # Load and summarize each table
        for table in tables:
            df = self.load_data(table)
            descriptive_stats, data_structure = self.data_summarization(df)
            tables[table] = {
                "descriptive_stats": descriptive_stats,
                "data_structure": data_structure
            }
        return tables

    def data_quality_assessment(self, df):
        """
        Perform a data quality assessment by checking for missing values in
        the DataFrame.

        Returns:
            - missing_values (pandas.Series): A Series containing the count
            of missing values for each column in the DataFrame.
        """
        # Check for missing values
        missing_values = df.isnull().sum()

        return missing_values

    def data_quality_check_all_tables(self):
        """
        Load and perform data quality assessment on all tables in the database.

        Returns:
        - tables (dict): A dictionary containing data quality assessment
        results for each table.
        Keys are table names, and values are dictionaries with the following
        structure:
        {
        "missing_values": pandas.Series
        }
        """
        tables = {
            "app_download_data": None,
            "google_play_store_reviews": None,
            "telegram_post_performance": None,
            "telegram_subscription_growth": None
        }

        # Load and perform data quality assessment for each table
        for table in tables:
            df = self.load_data(table)
            missing_values = self.data_quality_assessment(df)
            tables[table] = {"missing_values": missing_values}
        return tables

    def univariate_analysis(self, df):
        """
        Perform univariate analysis by creating histograms for numerical
        columns and bar charts for categorical columns.

        Args:
        - df (pandas.DataFrame): DataFrame to summarize.

        Returns:
        - univariate_plots (list): A list of matplotlib figures, each
        containing a histogram or bar chart.
        - univariate_data (dict): A dictionary containing the data used for
        each plot.
        """
        univariate_data = {}
        univariate_plots = []

        # Setting style for the plots
        sns.set(style="whitegrid")

        # Histograms for numerical columns
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Drop NaN values before calculating the histogram
            non_nan_values = df[col].dropna()
            counts, bin_edges = np.histogram(non_nan_values, bins=30)
            sns.histplot(
                non_nan_values, bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title(f'Histogram of {col}', fontsize=15)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            univariate_plots.append(fig)
            plt.close(fig)

            # Extract histogram data
            univariate_data[col] = {
                'type': 'histogram', 'counts': counts, 'bin_edges': bin_edges}

        # Bar charts for categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            fig, ax = plt.subplots(figsize=(16, 10))
            # Only plot top 20 most common categories to avoid clutter
            top_categories = df[col].value_counts().nlargest(20)
            sns.barplot(
                    x=top_categories.index, y=top_categories.values,
                    hue=top_categories.index, palette='viridis',
                    ax=ax, dodge=False, legend=False)
            ax.set_title(f'Bar chart of {col}', fontsize=15)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            # Rotate x-axis labels if they are too long
            if len(max(top_categories.index, key=len)) > 10:
                plt.xticks(rotation=25, ha='right')

            univariate_plots.append(fig)

            # Extract bar chart data
            univariate_data[col] = {
                'type': 'bar_chart',
                'categories': top_categories.index.tolist(),
                'counts': top_categories.values.tolist()
            }

            plt.close(fig)

        return univariate_plots, univariate_data

    def univariant_analysis_for_all_tables(self):
        """
        Perform univariate analysis for all tables in the database.

        Returns:
        - tables (dict): A dictionary containing univariate analysis results
        for each table.
        Keys are table names, and values are dictionaries with the
        following structure:
        {
            "univariate_data": pandas.DataFrame,
            "univariate_plots": dict
        }
        """
        tables = {
            "app_download_data": None,
            "google_play_store_reviews": None,
            "telegram_post_performance": None,
            "telegram_subscription_growth": None
        }

        # Load and perform univariate analysis for each table
        for table in tables.items():
            df = self.load_data(table)
            univariate_data, univariate_plots = self.univariate_analysis(df)
            tables[table] = {
                "univariate_data": univariate_data,
                "univariate_plots": univariate_plots
            }
        return tables

    def bivariate_multivariate_analysis(self):
        """
        Perform bivariate and multivariate analysis on the loaded datasets.

        This method analyzes the correlation between Telegram ads views and
        Apollo app ratings as well as likes, both from Tikvah and BoAEth.
        It calculates Pearson correlation coefficients and creates scatter
        plots to visualize the relationships.

        Returns:
        tuple:
        - correlation_matrix (dict): A dictionary containing Pearson
        correlation coefficients for different combinations of variables.
        - plots (dict): A dictionary containing scatter plot objects for
        each bivariate analysis.
        """
        df_ads_tikvah_telegram = self.load_data("telegram_post_performance")
        df_app_reviews = self.load_data("google_play_store_reviews")
        df_ads_boa_telegram = self.load_data("telegram_subscription_growth")

        # Define data and labels dictionaries for Tikvah rating
        data_tikvah_rating = {
            'x': df_ads_tikvah_telegram['views'],
            'y': df_app_reviews['score']
        }
        labels_tikvah_rating = {
            'xlabel': 'Telegram Ads Views',
            'ylabel': 'Apollo Rating',
            'title': 'Correlation between Telegram Ads Views from ' +
            'Tikvah and Apollo Rating'
        }

        # Perform correlation and plot for Tikvah rating
        correlation_tikvah_rating, plot_tikvah_rating = \
            self.correlation_and_plot(data_tikvah_rating, labels_tikvah_rating)

        # Define data and labels dictionaries for BoAEth rating
        data_boa_rating = {
            'x': df_ads_boa_telegram['views'],
            'y': df_app_reviews['score']
        }
        labels_boa_rating = {
            'xlabel': 'Telegram Ads Views',
            'ylabel': 'Apollo Rating',
            'title': 'Correlation between Telegram Ads Views from' +
            'BoAEth and Apollo Rating'
        }

        # Perform correlation and plot for BoAEth rating
        correlation_boa_rating, plot_boa_rating = \
            self.correlation_and_plot(data_boa_rating, labels_boa_rating)

        # Define data and labels dictionaries for Tikvah likes
        data_tikvah_likes = {
            'x': df_ads_tikvah_telegram['views'],
            'y': df_app_reviews['thumbs_up_count']
        }
        labels_tikvah_likes = {
            'xlabel': 'Telegram Ads Views',
            'ylabel': 'Apollo Likes',
            'title': 'Correlation between Telegram Ads Views from Tikvah' +
            'and Likes'
        }

        # Perform correlation and plot for Tikvah likes
        correlation_tikvah_likes, plot_tikvah_likes = \
            self.correlation_and_plot(data_tikvah_likes, labels_tikvah_likes)

        # Define data and labels dictionaries for BoAEth likes
        data_boa_likes = {
            'x': df_ads_boa_telegram['views'],
            'y': df_app_reviews['thumbs_up_count']
        }
        labels_boa_likes = {
            'xlabel': 'Telegram Ads Views',
            'ylabel': 'Apollo Likes',
            'title': 'Correlation between Telegram Ads Views from BoAEth' +
            'and Likes'
        }

        # Perform correlation and plot for BoAEth likes
        correlation_boa_likes, plot_boa_likes = \
            self.correlation_and_plot(data_boa_likes, labels_boa_likes)

        # Create correlation matrix
        correlation_matrix = {
            'Tikvah_Apollo_Rating': correlation_tikvah_rating,
            'Boa_Apollo_Rating': correlation_boa_rating,
            'Tikvah_Apollo_Likes': correlation_tikvah_likes,
            'Boa_Apollo_Likes': correlation_boa_likes
        }

        # Create plot objects dictionary
        plots = {
            'Tikvah_Apollo_Rating': plot_tikvah_rating,
            'Boa_Apollo_Rating': plot_boa_rating,
            'Tikvah_Apollo_Likes': plot_tikvah_likes,
            'Boa_Apollo_Likes': plot_boa_likes
        }

        return correlation_matrix, plots

    def correlation_and_plot(self, data, labels):
        """
        Calculate Pearson correlation coefficient and create a scatter plot.

        Args:
        data (dict): A dictionary containing the data for correlation analysis.
        Should contain keys 'x', 'y', 'x_label', 'y_label', 'title'.
        labels (dict): A dictionary containing the labels for the plot.
        Should contain keys 'x', 'y', 'xlabel', 'ylabel', 'title'.

        Returns:
        tuple:
        - correlation (float): The Pearson correlation coefficient
        between x and y.
        - plot (matplotlib.axes.Axes): The scatter plot object.
        """
        correlation = data['x'].corr(data['y'])

        plt.figure(figsize=(12, 6))
        plot = sns.scatterplot(x=data['x'], y=data['y'])
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'])

        return correlation, plot

    def data_comparison_trends_over_time(self):
        """
        Visualize data comparison trends over time for multiple datasets.

        This method loads data from multiple tables in the database and
        creates scatter plots to visualize the trend of views over time
        for each dataset.

        Returns:
        None
        """
        datasets = {
            "telegram_post_performance":
            "Telegram Ad Views Over Time from TikvahEthiopia",
            "telegram_subscription_growth":
            "Telegram Ad Views Over Time from BoAEth"
        }

        sns.set_style("whitegrid")
        for table, title in datasets.items():
            df = self.load_data(table)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            plt.figure(figsize=(12, 6))
            plt.scatter(
                df['timestamp'], df['views'], color='royalblue',
                edgecolor='black', alpha=0.75, s=100)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14, labelpad=10)
            plt.ylabel('Views', fontsize=14, labelpad=10)
            plt.xticks(rotation=45, ha='right')
            plt.gca().xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(
                plt.matplotlib.dates.MonthLocator(interval=2))
            plt.grid(
                True, which='both', linestyle='--',
                linewidth=0.5, color='grey')
            plt.tight_layout()
            plt.show()

    def has_outliers(self, series, k=1.5):
        """
        Determines if a numerical series has outliers using the IQR method.

        This method calculates the first and third quartiles, the IQR, and the
        lower and upper bounds.
        It then checks if any values in the series lie outside these bounds
        indicating the presence of outliers.

        Args:
            series (pd.Series): The numerical series to check for outliers.
            k (float, optional): The multiplier for the IQR to determine the
        bounds. Default is 1.5.

        Returns:
            bool: True if outliers are present, False otherwise.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        return any((series < lower_bound) | (series > upper_bound))

    def outlier_detection(self):
        """
        Detects and visualizes outliers in numerical columns of the DataFrame.
        This method creates a box plot for each numerical column to visually
        inspect for outliers. It then identifies columns with outliers using
        the IQR method and returns the columns and their corresponding
        outlier data.

        Returns:
        tuple:
            - outlier_plots (list): List of columns for which box plots
            were created.
            - outlier_columns_df (pd.DataFrame): DataFrame containing only
            the columns with detected outliers.
        """
        outlier_plots = []
        outlier_columns = []

        df_post = self.load_data("telegram_post_performance")
        df_reviews = self.load_data("google_play_store_reviews")
        df_sub = self.load_data("telegram_subscription_growth")

        for dataset_name, dataset in [
                ("telegram_post_performance", df_post),
                ("google_play_store_reviews", df_reviews),
                ("telegram_subscription_growth", df_sub)]:
            for column in dataset.select_dtypes(include=['int64', 'float64']):
                _, ax = plt.subplots(figsize=(8, 6))
                ax.boxplot(dataset[column], vert=False)
                ax.set_title(
                    f'Box plot for {column} in {dataset_name}', fontsize=14)
                ax.set_xlabel('Value', fontsize=12)
                ax.set_yticklabels([column], fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                outlier_plots.append((column, dataset_name))

                if self.has_outliers(dataset[column]):
                    if column not in outlier_columns:
                        outlier_columns.append((column, dataset_name))
                        plt.show()

        outlier_columns_df = pd.DataFrame(
            outlier_columns, columns=["Column", "Dataset"])
        return outlier_plots, outlier_columns_df

    def create_line_plot(self, data, x, y, title, xlabel, ylabel, filename):
        """
        Create a line plot to visualize trends over time.

        Args:
            data (DataFrame): The DataFrame containing the data to be plotted.
            x (str): The name of the column to be used for the x-axis.
            y (str): The name of the column to be used for the y-axis.
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            filename (str): The filename to save the plot.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=x, y=y, data=data)
        plt.title(title, fontsize=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def visualization(self):
        """
        Generate line plots to visualize trends over time for advertising
        views and app ratings.
        This method creates line plots to visualize the trends over time
        for advertising views from Tikvah and BoA, as well as the app ratings
        and likes over time for the Apollo app.

        Returns:
        None
        """
        df_review = self.load_data('google_play_store_reviews')
        df_post = self.load_data('telegram_post_performance')
        df_sub = self.load_data('telegram_subscription_growth')
        self.create_line_plot(
            df_post, 'timestamp', 'views',
            'Trend of Ads views over Time from Tikvah', 'Dates',
            'Views', 'Ad_Views_Trend_1.png')
        self.create_line_plot(
            df_sub, 'timestamp', 'views',
            'Trend of Ads views over Time from BoA',
            'Dates', 'Views', 'Ad_Views_Trend_2.png')
        self.create_line_plot(df_review, 'review_at', 'score',
                              'Trend of Apollo App rating over Time',
                              'Dates', 'Ratings', 'App_Rating_Trend.png')
        self.create_line_plot(df_review, 'review_at', 'thumbs_up_count',
                              'Trend of Apollo App Like over Time',
                              'Dates', 'Likes', 'App_Like_Trend.png')

        correlation_matrix, _ = self.bivariate_multivariate_analysis()
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5, cbar=True,
                square=True, annot_kws={"size": 12})
        plt.title('Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
