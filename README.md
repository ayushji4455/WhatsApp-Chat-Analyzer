# WhatsApp Chat Analyzer & NLP Insights

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/models)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


An interactive web application built with Streamlit to analyze, visualize, and derive deep insights from WhatsApp chat logs. This tool performs everything from statistical analysis to advanced NLP tasks like sentiment analysis on Hinglish text and automatic topic modeling.

Live Demo: [https://whatsapp-chat-analyzer-and-sentiment-analysis-fwx5d3bsekcu8ahc.streamlit.app/]


---

## 1. Project Overview

WhatsApp chat logs are unstructured text files filled with valuable data. However, manually extracting insights is impossible. This project was built to solve that problem by providing a user-friendly interface to upload a chat file and automatically generate a comprehensive report. The application answers questions like "Who is the most active user?", "What are the main topics we discuss?", and "Is the overall mood of our group positive or negative?".

The project's key challenge and feature is its ability to accurately analyze **Hinglish** (a mix of Hindi and English), a common language in informal chats where standard NLP tools fail.

![Alt Text](Images/Screenshot(57).png)

![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/7e866fc035fb43c824c7ca289c56985695a1073c/Images/Screenshot%20(58).png)
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(59).png)

---

## 2. Key Features

-   **Comprehensive Statistical Dashboard:** The main view displays key metrics, including total messages, words, media shared, and links. It also visualizes user activity through timelines, heatmaps, and rankings of the busiest users, days, and months.
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(60).png)
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(61).png)
    -   **Accurate Hinglish Sentiment Analysis:** The app goes beyond simple analysis by using a state-of-the-art **Hugging Face transformer model** (`pascalrai/hinglish-twitter-roberta-base-sentiment`). This model is specifically fine-tuned on Hinglish social media text, allowing it to accurately determine if messages are positive, negative, or neutral.
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(65).png)
    -   **Automatic Topic Modeling:** Uses **Latent Dirichlet Allocation (LDA)** to automatically discover the top 5 underlying themes of the conversation, providing a high-level summary of what the group talks about most.
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(63).png)
-   **In-Depth Word and Emoji Analysis:** Identifies and visualizes the most frequently used words and emojis in the chat. This feature uses a custom **Hinglish stop-word list** to filter out common, non-essential words, ensuring the results are meaningful.
![Alt Text](https://github.com/AdityaVerma1405/WhatsApp-Chat-Analyzer-and-Sentiment-Analysis/blob/0e744193e264b8355768f1bdce07228073a87298/Images/Screenshot%20(64).png)
    ---

## 3. Workflow Explained

The project follows a standard data science application workflow:

1.  **Data Ingestion & Preprocessing:** The user uploads a `.txt` file via the Streamlit interface. A dedicated preprocessing module (`preprocessor.py`) parses the raw text using regular expressions and converts the data into a clean Pandas DataFrame.

2.  **Feature Engineering:** New columns (like `year`, `month`, `day`, `hour`) are extracted from the datetime objects to enable time-based analysis.

3.  **Backend Analysis:** The `helper.py` script takes the clean DataFrame and applies various functions to calculate statistics and perform NLP tasks like sentiment analysis and topic modeling.

4.  **Frontend Visualization:** The main application script (`app.py`) uses Matplotlib and Seaborn to generate plots, charts, and tables, which are then displayed to the user in an organized layout.

---

## 4. Folder Structure
```bash
├── app.py                  # Main Streamlit app
├── helper.py               # Analysis & NLP functions
├── preprocessor.py         # Data cleaning
├── requirements.txt        # Dependencies
├── stop_hinglish.txt       # Custom stopwords
├── screenshots/            # Images for README
└── README.md
```
---
## 5. Tech Stack

| Area      | Tools                                                              |
| :-------- | :----------------------------------------------------------------- |
| **Backend** | Python, Pandas, NumPy                                              |
| **Frontend** | Streamlit                                                          |
| **ML/NLP** | Scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`)        |
| **Sentiment** | Hugging Face Transformers (`pipeline`), NLTK                       |
| **Data Viz** | Matplotlib, Seaborn, WordCloud                                     |
| **Dev Tools** | Git, GitHub, VS Code                                               |

---
## 6. Future Enhancements

- **Interactive Visualizations:** Replace current Matplotlib charts with Plotly to allow for interactive exploration (hovering, zooming).

- **Social Network Graph:** Create a graph visualization to map who talks to whom most frequently, revealing social dynamics and subgroups within the chat.

- **Response Time Analysis:** Calculate the average time it takes for users to respond to messages, identifying the "fastest repliers."

- **User-Level Topic Modeling:** Allow the user to see the main topics of conversation for a specific individual, not just the overall group.

---

## 7. License
This project is licensed under the **MIT License** — feel free to use and modify.
