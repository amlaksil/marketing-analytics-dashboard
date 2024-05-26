#!/usr/bin/python3
"""
Script to fetch messages related to Abyssinia Bank from a Telegram channel
and save them to a CSV file.
"""

from os import getenv
import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

# Telegram API credentials
api_id = getenv('API_ID')
api_hash = getenv('API_HASH')

# Channel information
channel_username = 'tikvahethiopia'

# CSV filename to save the data
csv_filename = 'abyssinia_bank_ads.csv'

# Initialize the Telegram client
client = TelegramClient('session_name', api_id, api_hash)


def get_all_messages(client, channel_username):
    """
    Retrieves all messages from the specified Telegram channel.

    Args:
    - client (TelegramClient): The initialized Telegram client.
    - channel_username (str): The username of the Telegram channel.

    Returns:
    - list: A list of dictionaries representing the messages.
    """
    all_messages = []
    offset_id = 0
    limit = 100

    while True:
        history = client(GetHistoryRequest(
            peer=channel_username,
            offset_id=offset_id,
            offset_date=None,
            add_offset=0,
            limit=limit,
            max_id=0,
            min_id=0,
            hash=0
        ))

        if not history.messages:
            break

        messages = history.messages
        for message in messages:
            all_messages.append(message.to_dict())

        offset_id = messages[-1].id

    return all_messages


def filter_abyssinia_bank_messages(messages):
    """
    Filters messages related to Abyssinia Bank.

    Args:
    - messages (list): A list of dictionaries representing Telegram messages.

    Returns:
    - list: A list of dictionaries containing filtered messages.
    """
    filtered_messages = []
    keywords = ['Abyssinia', 'አቢሲንያ', 'BoA']

    for message in messages:
        if 'message' in message:
            for keyword in keywords:
                if keyword in message['message']:
                    filtered_messages.append({
                        'id': message['id'],
                        'date': message['date'],
                        'url':
                        f"https://t.me/{channel_username}/{message['id']}",
                        'views': message.get('views', 0)
                    })
                    break

    return filtered_messages


def save_to_csv(messages, filename):
    """
    Saves messages to a CSV file.

    Args:
    - messages (list): A list of dictionaries representing messages.
    - filename (str): The filename to save the CSV file.
    """
    df = pd.DataFrame(messages)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    with client:
        # Get all messages from the channel
        all_messages = get_all_messages(client, channel_username)

        # Filter messages related to Abyssinia Bank
        abyssinia_bank_messages = filter_abyssinia_bank_messages(all_messages)

        # Save the filtered messages to a CSV file
        save_to_csv(abyssinia_bank_messages, csv_filename)
