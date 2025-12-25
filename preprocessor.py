import re
import pandas as pd

def preprocess(data):
    # Pattern to find the start of each message (date-time)
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[AP]M\s-\s'

    # Split the data into messages and dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create the initial DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean the special space character and convert to datetime
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ')
    df['date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ')
    df.drop(columns=['message_date'], inplace=True)

    # (IMPROVEMENT) Use .str.extract() for a faster, vectorized way to separate users and messages
    df[['user', 'message']] = df['user_message'].str.extract(r'([\w\W]+?):\s(.*)')
    df.drop(columns=['user_message'], inplace=True)

    # Handle group notifications where user is NaN (Not a Number) after extraction
    df['user'].fillna('group_notification', inplace=True)

    # Extract detailed date-time components
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # (IMPROVEMENT) Create the 'period' column efficiently without a loop
    df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h+1)%24:02d}")

    return df