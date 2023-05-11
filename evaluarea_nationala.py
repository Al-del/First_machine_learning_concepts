import requests
import pandas as pd

url = 'http://admitere.edu.ro/2022/repartizare/CT/index.html'
html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[7]
print(df)