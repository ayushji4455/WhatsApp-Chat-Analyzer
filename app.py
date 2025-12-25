import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Segoe UI Emoji'

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique users for the dropdown
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # (IMPROVEMENT) Filter the dataframe once and pass the result to all functions
        if selected_user != 'Overall':
            filtered_df = df[df['user'] == selected_user]
        else:
            filtered_df = df

        # --- STATS AREA ---
        # (IMPROVEMENT) Correctly unpack all 4 values returned by fetch_stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(filtered_df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # --- TIMELINE AREA ---
        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(filtered_df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(filtered_df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # --- ACTIVITY MAP ---
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(filtered_df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(filtered_df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(filtered_df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)  # Tells heatmap to draw on the ax object
        st.pyplot(fig)

        # --- BUSIEST USERS (Overall only) ---
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df) # Note: Pass original df for overall stats
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # --- WORD ANALYSIS ---
        # WordCloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(filtered_df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(filtered_df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        # --- EMOJI ANALYSIS ---
        emoji_df = helper.emoji_helper(filtered_df)
        if not emoji_df.empty:
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)

        # --- SENTIMENT ANALYSIS ---
        st.title("Sentiment Analysis")

        # Apply sentiment analysis to the filtered dataframe
        sentiment_df = helper.sentiment_analysis(filtered_df.copy())  # Use .copy() to avoid warnings

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            st.header("Overall Sentiment")
            # Create a pie chart of sentiment distribution
            sentiment_counts = sentiment_df['sentiment_label'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%0.2f",
                   colors=['#2ecc71', '#e74c3c', '#3498db'])
            st.pyplot(fig)

        with col2:
            st.header("Most Positive Message")
            # Find and display the most positive message
            most_positive_msg = sentiment_df.loc[sentiment_df['sentiment_compound'].idxmax()]
            st.info(f"**User:** {most_positive_msg['user']}")
            st.write(f"**Message:** {most_positive_msg['message']}")
            st.write(f"**Score:** {most_positive_msg['sentiment_compound']:.2f}")

        st.header("Most Negative Message")
        # Find and display the most negative message
        most_negative_msg = sentiment_df.loc[sentiment_df['sentiment_compound'].idxmin()]
        st.error(f"**User:** {most_negative_msg['user']}")
        st.write(f"**Message:** {most_negative_msg['message']}")
        st.write(f"**Score:** {most_negative_msg['sentiment_compound']:.2f}")

        # --- TOPIC MODELING ---
        st.title("Discovered Chat Topics")

        # Call the helper function to get the topics
        chat_topics = helper.find_topics(filtered_df)

        # Display each topic
        if chat_topics:
            for topic in chat_topics:
                st.subheader(topic)
        else:
            st.write("Not enough data to determine topics.")

