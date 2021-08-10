#!/usr/bin/env python
# coding: utf-8

# Облако слов по заголовкам англоязычных статей с сайта google news (https://news.google.com)

## Результат
#В папку ./out выгружается картинка облака слов в формате .jpg и файл excell. 
#В файле excell содержатся заголовки статей с ссылками на них и издателя. В столбце title - наименования статей найденных по ключевой фразе за заданный #период времени. В столбце tokens - простые формы слов, к которым были приведены заголовки статей. По столбцу tokens строится облако слов.
#Для анализа контекста слова из облака используйте колонку tokens и ссылки на конкретные статьи, находящиеся в соответствующих строках.

## Для запуска требуется Python3. Необходимые библиотеки автоматически загрузятся согласно файлу requirements.txt
#Перенесите файлы Google_article.py и requirements.txt в свою папку. 
#Выполните в консоли команду: 
#$ python3 Google_article.py 
#или укажите путь к файлу:
#$ python3 ~/home/... /Google_article.py
#Далее программа попросит ввести несколько параметров для формирования запроса к google news. Если просто нажимать ввод, по умолчанию запрос будет сделан #за период 30 дней (30d) с ключевым словом Rossia, регион поиска United States.


# установка библиотек из requirements
#import os
#o=os.system('pip install -r requirements.txt')

# библиотеки
from GoogleNews import GoogleNews #https://pypi.org/project/GoogleNews/
from newspaper import Article
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import os
import pandas as pd
import nltk # pip install -U nltk
nltk.download("stopwords") # nltk.download() для скачивания всех дополнительных возможностей
nltk.download('punkt')
from string import punctuation #из библиотеки string питона достаём пунктуацию и импортируем её
from nltk.corpus import stopwords #библиотека nltk содержит стоп следующие слова
from pymystem3 import Mystem
from collections import Counter

# Подготавливаем запрос и объявляем класс googlenews
per=str(input('\n\nВведите период поиска в GoogleNews, например 30d:\n') or '30d')
key = str(input('Введите на английском ключевую фразу поиска, например Russian Sport:\n') or 'Russia')
reg = str(input('Введите регион поиска, например United Kingdom, или United States, \nCanada, India, Israel... см. варианты на сайте news.google.com:\n') or 'United States')
num_words=int(input('Введите число слов в облаке, например, 50:\n') or 50)

googlenews = GoogleNews(lang='en', region=reg,period=per,encode='utf-8')

# Загрузка новостей
googlenews.get_news(key)#googlenews.search('Russia')
result=googlenews.result()
df=pd.DataFrame(result)

# Токенизация текста
def norm_text(txt_list):
    """tokens_sent = norm_text(txt_list)
    Функция возвращает список слов в основной форме с нижним регистром без стоп слов.
    Принимает txt_list - список предложений"""
    # токенизация по словам
    tokens_sentences = [nltk.word_tokenize(sentence) for sentence in txt_list] 
    # приведение к нижнему регистру
    tokens_sentences = [[token.lower() for token in token_sentence] for token_sentence in tokens_sentences] 
    mystem = Mystem() # лемматизатор
    #слова для удаления + добавили недостающие
    stop_words = stopwords.words('english')+['far','is']
    # удаление знаков пунктуации, стоп слов и не являются буквенными
    tokens_sent = [[token for token in token_sentence if token not in punctuation and len(token)>2 and token not in stop_words and token.isalpha()] 
                        for token_sentence in tokens_sentences]
    return tokens_sent

txt_list=list(df.title)
tokens_sent = norm_text(txt_list)

#Добавим столбец с токенами в таблицу пандас
token=pd.Series(tokens_sent,name='token') #создаю столбец с токенами
df['token']=token

columns=list(df.columns) #
col=[]
 
# столбец с токенами поместим во вторую колонку общей таблицы
for c in columns[:-1]:
    if columns.index(c)==1:
        col.append(columns[-1])
    col.append(c)
df = df.reindex(columns=col)
df.head(2)

# Сохранение таблицы пандас в excel
# создаём директорию для таблицы и картинок
path = "out/"
try:
    os.makedirs(path)
except OSError:
    print ("Директория %s возможно есть" % path)
else:
    print ("Успешно создана директория %s" % path)
df.to_excel('out/googlenews_result.xlsx')

# Находим статистику уникальных токенов
tokens = [token for sublist in tokens_sent for token in sublist]
freq_tokens = dict(Counter(tokens))
sorted_tuple = sorted(freq_tokens.items(), key=lambda x: x[1],reverse=True)
if len(sorted_tuple)>=num_words:
    print(sorted_tuple[:50])
else:
    print(sorted_tuple)

# создает маску круга и рисуем в ней облако слов

def create_ellips_mask(h, w, center=None, radius=None):

    if center is None: 
        center = (int(w/2), int(h/2))
    if radius is None: 
        radius = min(center[0], center[1], h-center[1],w-center[0])

    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0]) ** 2 + ((Y-center[1])*1.9) **2)
    
    mask = [[False for i in Y] for x in X]
    for x in X:
        for y in Y:
            mask
    mask = dist_from_center
    mask = dist_from_center <= radius
    mask = ~mask
    
    return mask

mask = create_ellips_mask(2000, 4000, radius=4000/2)
mask = 255 * mask.astype(int)

wc = WordCloud(width=4000, height=2000, max_words=num_words, 
               background_color='white', mask=mask,colormap='plasma').generate_from_frequencies(freq_tokens)

plt.figure(figsize=(20, 10), dpi=50)
plt.imshow(wc)
plt.tight_layout(pad=0)
plt.axis("off")
plt.savefig('out/top'+str(num_words)+'_plasma.jpg')
plt.show()

