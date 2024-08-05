import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, regexp_tokenize
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import spacy
nlp = spacy.load('en_core_web_sm')

#--------------------------------- Data Preprocessing ---------------------------------#

def get_dummies(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to convert the categorical feature into one-hot encoding.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be converted.
    :return: The dataframe with the one-hot encoding of the feature.
    """
    return pd.get_dummies(df, columns=[feature])

def standardize(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to standardize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be standardized.
    :return: The dataframe with the standardized feature.
    """
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])
    return df

def normalize(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to normalize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be normalized.
    :return: The dataframe with the normalized feature.
    """
    df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df

def binarize(df: pd.DataFrame, feature: str, threshold: float) -> pd.DataFrame:
    """
    This function is used to binarize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be binarized.
    :param threshold: The threshold to binarize the feature.
    :return: The dataframe with the binarized feature.
    """
    df[feature] = np.where(df[feature] > threshold, 1, 0)
    return df

def discretize(df: pd.DataFrame, feature: str, bins: int) -> pd.DataFrame:
    """
    This function is used to discretize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be discretized.
    :param bins: The number of bins to discretize the feature.
    :return: The dataframe with the discretized feature.
    """
    df[f"{feature}_discretize"] = pd.cut(df[feature], bins, labels=False)
    return df

def remove_outliers(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to remove the outliers from the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to remove the outliers.
    :return: The dataframe with the outliers removed from the feature.
    """
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
    return df

def minmax_scale_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    This function is used to min-max scale the column.
    :param df: The dataframe containing the column.
    :param column_name: The column to be min-max scaled.
    :return: The dataframe with the min-max scaled column.
    """
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    
    return df

#--------------------------------- Feature Engineering Date ---------------------------------#

def encode_date_numerically(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to encode the date feature numerically.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be encoded.
    :return: The dataframe with the date feature encoded numerically.
    """
    df[feature] = pd.to_datetime(df[feature])
    df[feature] = df[feature].map(lambda x: 10000*x.year + 100*x.month + x.day)
    return df

def convert_date(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to convert the date feature to a specific format.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be converted.
    :return: The dataframe with the date feature converted to a specific format.
    """
    try:
        df[feature] = pd.to_datetime(df[feature], format='%Y-%m-%d %H:%M:%S.%f').dt.strftime('%Y-%m-%d')
    except:
        df[feature] = pd.to_datetime(df[feature], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
    return df

def convert_millisecond_date(date_str: str) -> pd.Timestamp:
    """
    This function is used to convert the date in milliseconds to a specific format.
    :param date_str: The date in milliseconds.
    :return: The date in a specific format.
    """
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')

def extract_date_features(df: pd.DataFrame, feature: str, fill: str = '1900-01-01') -> pd.DataFrame:
    """
    This function is used to extract the date features from the date feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to extract the date features.
    :param fill: The value to fill the missing values in the feature.
    :return: The dataframe with the date features extracted from the date feature.
    """
    df[feature] = df[feature].fillna(fill)
    df[feature + '_year'] = (pd.to_datetime(df[feature]).dt.year).astype(int)
    df[feature + '_month'] = (pd.to_datetime(df[feature]).dt.month).astype(int)
    df[feature + '_day'] = (pd.to_datetime(df[feature]).dt.day).astype(int)
    df[feature + '_dayofweek'] = (pd.to_datetime(df[feature]).dt.dayofweek).astype(int)
    df[feature + '_is_weekend'] = pd.to_datetime(df[feature]).dt.dayofweek.isin([5, 6]).astype(int)
    df[feature] = pd.to_datetime(df[feature])
    return df

def encode_date_cyclically(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to encode the date feature cyclically.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be encoded.
    :return: The dataframe with the date feature encoded cyclically.
    """
    df[feature] = pd.to_datetime(df[feature], format='%Y-%m')
    df[feature + '_month_sin'] = np.sin(2 * np.pi * df[feature].dt.month / 12)
    df[feature + '_month_cos'] = np.cos(2 * np.pi * df[feature].dt.month / 12)
    df[feature + '_year_sin'] = np.sin(2 * np.pi * df[feature].dt.year)
    df[feature + '_year_cos'] = np.cos(2 * np.pi * df[feature].dt.year)
    return df

#--------------------------------- Feature Engineering Text ---------------------------------#

def remove_short_words(text: str, min_length: int = 3) -> str:
    """
    Removes words shorter than min_length characters.
    :param text: the text to process
    :param min_length: minimum length of words to keep (default is 3)
    :return: the text with short words removed
    """
    words = text.split()
    filtered_words = [word for word in words if len(word) >= min_length]
    return ' '.join(filtered_words)

def remove_common_words(text: str, common_words: set) -> str:
    """
    Removes common words from text.
    :param text: the text to process
    :param common_words: set of common words to remove
    :return: the text with common words removed
    """
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in common_words]
    return ' '.join(filtered_words)

def advanced_tokenization(text: str) -> list:
    """
    Tokenizes text using advanced tokenization techniques.
    :param text: the text to tokenize
    :return: a list of tokens
    """
    return regexp_tokenize(text, pattern=r'\s|[\.,;?!]', gaps=True)

def remove_base_words(text: str, base_words: set) -> str:
    """
    Removes base words from text.
    :param text: the text to process
    :param base_words: set of base words to remove
    :return: the text with base words removed
    """
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in base_words]
    return ' '.join(filtered_words)

def remove_low_frequency_words(text: str, corpus: str, min_frequency: int = 2) -> str:
    """
    Removes words with low frequency of occurrence.
    :param text: the text to process
    :param corpus: the corpus used to calculate word frequency
    :param min_frequency: minimum frequency of words to keep
    :return: the text with low-frequency words removed
    """
    corpus_words = word_tokenize(corpus)
    word_freq = Counter(corpus_words)
    words = word_tokenize(text)
    filtered_words = [word for word in words if word_freq[word.lower()] >= min_frequency]
    return ' '.join(filtered_words)

def pos_tagging(text: str) -> list:
    """
    Performs POS tagging on the text.
    :param text: the text to tag
    :return: a list of tuples where each tuple contains a word and its POS tag
    """
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return tagged

def named_entity_recognition(text: str) -> list:
    """
    Performs Named Entity Recognition (NER) on the text.
    :param text: the text to analyze
    :return: a list of named entities and their types
    """
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    named_entities = nltk.chunk.ne_chunk(tagged)
    return named_entities

def stemming(text: str) -> str:
    """
    Applies stemming to the text.
    :param text: the text to process
    :return: the text with words reduced to their stem
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_words)

def dependency_parsing(text: str) -> list:
    """
    Performs dependency parsing on the text.
    :param text: the text to parse
    :return: a list of tuples where each tuple contains a word and its dependency relation
    """
    doc = nlp(text)
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return dependencies

def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the text using spaCy.
    :param text: the text to lemmatize
    :return: the lemmatized text
    """
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(lemmatized_words)

def advanced_preprocessing(text: str) -> str:
    """
    Performs advanced preprocessing using both NLTK and spaCy.
    :param text: the text to preprocess
    :return: the preprocessed text
    """
    # SpaCy processing
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    lemmatized_text = ' '.join(lemmatized_words)
    
    # NLTK processing
    tokens = word_tokenize(lemmatized_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return ' '.join(filtered_tokens)

def preprocess_url(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    This function is used to preprocess the URL feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be preprocessed.
    :return: The dataframe with the preprocessed URL feature.
    """
    df[feature] = df[feature].str.replace('http://', '')
    df[feature] = df[feature].str.replace('https://', '')
    df[feature] = df[feature].str.replace('www.', '')
    df[feature] = df[feature].str.split('.').str[0]
    return df

def preprocess_text(text: str) -> str:
    """
    Preprocesses text by removing extra spaces, digits, and leading/trailing spaces.
    :param text: the text to preprocess
    :return: the preprocessed text
    """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    
    return text

def preprocess_title(title: str) -> str:
    """
    Preprocesses text using tokenization, stopword removal, and lemmatization.
    :param title: the text to preprocess
    :return: the preprocessed text
    """
    # Tokenize and filter tokens
    tokens = [token for token in word_tokenize(title) if token.isalpha()]

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

def preprocess_content(content: str, lowercase: bool = True) -> str:
    """
    Preprocesses the content of a news article by tokenizing, optionally lowercasing, and removing stopwords.
    :param content: the content of a news article
    :param lowercase: whether to convert text to lowercase (default is True)
    :return: the preprocessed content
    """
    # Tokenize the content
    tokens = word_tokenize(content)
    
    # Optionally convert to lowercase
    if lowercase:
        tokens = [token.lower() for token in tokens]
    
    # Filter out non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    return ' '.join(tokens)

def remove_html_tags(text: str) -> str:
    """
    Removes HTML tags from text.
    :param text: the text to clean
    :return: the cleaned text
    """
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def remove_special_characters(text: str) -> str:
    """
    Removes special characters from text.
    :param text: the text to clean
    :return: the cleaned text
    """
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return clean_text

def remove_emojis(text: str) -> str:
    """
    Removes emojis from text.
    :param text: the text to clean
    :return: the cleaned text
    """
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace in text by removing extra spaces.
    :param text: the text to normalize
    :return: the normalized text
    """
    return re.sub(r'\s+', ' ', text).strip()

def expand_contractions(text: str, contractions_dict: dict) -> str:
    """
    Expands contractions in text using a contractions dictionary.
    :param text: the text with contractions
    :param contractions_dict: dictionary of contractions and their expanded forms
    :return: the text with expanded contractions
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), flags=re.IGNORECASE)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match.lower())
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def preprocess_text_full(text: str, contractions_dict: dict) -> str:
    """
    Fully preprocesses text by removing HTML tags, special characters, emojis, expanding contractions,
    normalizing whitespace, and converting to lowercase.
    :param text: the text to preprocess
    :param contractions_dict: dictionary of contractions and their expanded forms
    :return: the preprocessed text
    """
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    text = remove_emojis(text)
    text = expand_contractions(text, contractions_dict)
    text = normalize_whitespace(text)
    text = text.lower()
    return text

#--------------------------------- Time Series Analysis ---------------------------------#

def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str, title: str = 'Time Series Plot') -> None:
    """
    Plots time series data.
    :param df: DataFrame containing the data
    :param date_column: Name of the date column
    :param value_column: Name of the value column
    :param title: Title of the plot
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[value_column], label=value_column)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_histogram(value: pd.Series, title: str = 'Histogram of Values') -> None:
    """
    Plots a histogram of values.
    :param value: Series containing the values to plot
    :param title: Title of the plot
    :return: None
    """
    plt.figure(figsize=(10, 6))
    value.plot(kind='bar')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def adf_test(series: pd.Series, title: str = '') -> bool:
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.
    :param series: Time series data
    :param title: Optional title for the test output
    :return: Boolean indicating whether the series is stationary
    """
    stationary = False
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    print(out.to_string())

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
        stationary = True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

    return stationary

def kpss_test(series: pd.Series, title: str = '', regression: str = 'c') -> bool:
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test to check for stationarity.
    :param series: Time series data
    :param title: Optional title for the test output
    :param regression: Type of regression for the test ('c' for constant, 'ct' for constant and trend)
    :return: Boolean indicating whether the series is stationary
    """
    stationary = False
    print(f'Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test: {title}')
    result = kpss(series, regression=regression)

    labels = ['KPSS test statistic', 'p-value', '# lags used']
    out = pd.Series(result[0:3], index=labels)

    for key, val in result[3].items():
        out[f'critical value ({key})'] = val

    print(out.to_string())

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
        stationary = True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

    return stationary

def plot_rolling_mean_std(df: pd.DataFrame, column: str, window: int) -> None:
    """
    Plots the rolling mean and standard deviation for a time series.
    :param df: DataFrame containing the time series data
    :param column: Name of the column to analyze
    :param window: Window size for calculating rolling statistics
    :return: None
    """
    plt.figure(figsize=(10, 6))
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    plt.plot(df[column], label='Original')
    plt.plot(rolling_mean, label=f'{window}-period Rolling Mean')
    plt.plot(rolling_std, label=f'{window}-period Rolling Std')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Rolling Mean & Standard Deviation (Window={window})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(series: pd.Series, lags: int, title_prefix: str = '') -> None:
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for a time series.
    :param series: Time series data
    :param lags: Number of lags to include in the plots
    :param title_prefix: Prefix for the plot titles
    :return: None
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plot_acf(series, lags=lags, title=f'{title_prefix} Autocorrelation Function')
    plt.subplot(122)
    plot_pacf(series, lags=lags, title=f'{title_prefix} Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

