import pandas as pd
import requests
from bs4 import BeautifulSoup

df = pd.read_csv(r'G:\Lock in\New folder\Dataset\gossipcop_real.csv')
df_test = df.head(2)

def fetch_article_body(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            article_div = soup.find('div', itemprop='articleBody')
            if article_div:
                paragraphs = article_div.find_all('p')
                article_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
            else:
                article_text = ""
            return "Working", article_text
        else:
            return f"Not working ({response.status_code})", ""
    except requests.RequestException as e:
        return "Error", str(e)

df_test[['url_status', 'content_full']] = df_test['news_url'].apply(lambda x: pd.Series(fetch_article_body(x)))

df_test.to_csv('urls_checked_article_body_test.csv', index=False)

print(df_test[['news_url', 'url_status']])
