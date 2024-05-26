#!/usr/bin/python3
"""
Script to fetch app details and reviews from Google Play Store using
the google_play_scraper library.
"""

import pandas as pd
from google_play_scraper import app, reviews_all


def fetch_app_details_and_reviews(app_id):
    """
    Fetches app details and reviews from Google Play Store.

    Args:
    - app_id (str): The package name of the app on Google Play Store.

    Returns:
    - tuple: A tuple containing app details dictionary and reviews DataFrame.
    """
    # Fetch app details
    app_details = app(app_id)

    # Fetch all reviews
    reviews = reviews_all(
        app_id,
        sleep_milliseconds=0,  # No delay between requests
        lang='en',  # Language of the reviews
        country='us'  # Country of the reviews
    )

    # Convert reviews to a DataFrame
    reviews_df = pd.DataFrame(reviews)

    return app_details, reviews_df


def save_data_to_csv(app_details, reviews_df):
    """
    Saves app details and reviews to CSV files.

    Args:
    - app_details (dict): Dictionary containing app details.
    - reviews_df (DataFrame): DataFrame containing reviews.
    """
    # Save app details to CSV
    app_details_df = pd.DataFrame([app_details])
    app_details_df.to_csv('app_details.csv', index=False)

    # Save reviews to CSV
    reviews_df.to_csv('app_reviews.csv', index=False)

    print("\nData has been saved to 'app_details.csv' and 'app_reviews.csv'.")


if __name__ == "__main__":
    # Define the app ID
    app_id = 'com.boa.apollo'

    # Fetch app details and reviews
    app_details, reviews_df = fetch_app_details_and_reviews(app_id)

    # Display app details
    print("App Details:")
    print(f"Title: {app_details['title']}")
    print(f"Developer: {app_details['developer']}")
    print(f"Score: {app_details['score']}")
    print(f"Installs: {app_details['installs']}")
    print(f"Content Rating: {app_details['contentRating']}")
    print(f"Updated: {app_details['updated']}")
    print(f"Genre: {app_details['genre']}")
    print(f"Description: {app_details['description']}")

    # Display the first few rows of the reviews DataFrame
    print("\nSample Reviews:")
    print(reviews_df.head())

    # Save the data to CSV files for further EDA
    save_data_to_csv(app_details, reviews_df)
