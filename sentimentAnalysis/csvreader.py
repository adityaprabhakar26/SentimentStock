import pandas as pd
import glob
from natsort import natsorted
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline
import pandas as pd
import emoji
import re
import requests
import json
import matplotlib.pyplot as plt
import time

API_URL = "https://api-inference.huggingface.co/models/zhayunduo/roberta-base-stocktwits-finetuned"
headers = {"Authorization": -Redacted-}
f = open("text.txt", "w")
f.close()
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
def query_with_retry(payload, max_retries=5, wait_time=20):
    retries = 0
    while retries < max_retries:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if "error" in data and data["error"] == "Model zhayunduo/roberta-base-stocktwits-finetuned is currently loading":
            print("Model is still loading. Retrying in {} seconds...".format(wait_time))
            time.sleep(wait_time)
            retries += 1
        else:
            return data
    raise Exception("Model loading took too long. Please try again later.")


def process_text(texts):
  texts = re.sub(r'https?://\S+', "", texts)
  texts = re.sub(r'www.\S+', "", texts)
  texts = texts.replace('&#39;', "'")
  texts = re.sub(r'(\#)(\S+)', r'hashtag_\2', texts)
  texts = re.sub(r'(\$)([A-Za-z]+)', r'cashtag_\2', texts)
  texts = re.sub(r'(\@)(\S+)', r'mention_\2', texts)
  texts = emoji.demojize(texts, delimiters=("", " "))
  return texts.strip()

def get_sentiment_label(sentence):
   try:
        response = nlp(sentence)
        label = response[0]["label"]
        return 1 if label == "Positive" else -1
   except Exception as e:
        print("Error occurred:", e)
        return 0

def clean_files(csvfiles):
  csvfiles.reverse()
  dfs = []
  for file_path in csvfiles:
    df = pd.read_csv(file_path, usecols=['body', 'created_at', 'entities'])
    dfs.append(df)
  combined_df = pd.concat(dfs)
  combined_df = combined_df[::-1].reset_index(drop=True)
  combined_df['created_at'] = pd.to_datetime(combined_df['created_at']).dt.date
  combined_df['entities'] = combined_df['entities'].apply(lambda x: {'sentiment': eval(x)['sentiment']} if 'sentiment' in eval(x) else {})
  combined_df['UncleanS'] = combined_df.apply(lambda row: row['body'] if pd.isnull(row['entities']['sentiment']) else 'Bullish' if row['entities']['sentiment']['basic'] == 'Bullish' else 'Bearish' if row['entities']['sentiment']['basic'] == 'Bearish' else None, axis=1)
  return combined_df

tokenizer_loaded = RobertaTokenizer.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
model_loaded = RobertaForSequenceClassification.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
nlp = pipeline("text-classification", model=model_loaded, tokenizer=tokenizer_loaded)
csvfiles = natsorted(glob.glob('/Users/adi/Downloads/StockTwits_2020_2022_Raw4/AAPL_2020_2022/*.csv'))
combined_df = clean_files(csvfiles)
sentences = combined_df['UncleanS']
sentences = list(sentences.apply(process_text))  # if input text contains https, @ or # or $ symbols, better apply preprocess to get a more accurate result
results = []
for sentence in sentences:
    '''response = query_with_retry({"inputs": sentence})
    results.append(get_sentiment_label(response[0][0]['label']))
    print(get_sentiment_label(response[0][0]['label']))
    '''
    results.append(get_sentiment_label(sentence))
    print(results[-1])
    #print(response)
    #results.append(get_sentiment_label(response[0][0]['label']))
    #print(get_sentiment_label(response[0][0]['label']))
combined_df['CleanS'] = results
print(combined_df)
average_by_day = combined_df.groupby('created_at')['CleanS'].mean().reset_index()
print(average_by_day)
average_by_day['created_at'] = pd.to_datetime(average_by_day['created_at'])
average_by_day.to_csv('AAPL_SENTIMENT.csv', index=True)
plt.figure(figsize=(10, 6))
plt.plot(average_by_day['created_at'], average_by_day['CleanS'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.title('Average Sentiment by Day')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
