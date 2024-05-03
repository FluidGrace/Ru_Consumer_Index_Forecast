import urllib
from urllib.request import urlretrieve

if __name__ == "__main__":
    #ищем страницы с сайта Госкомстата, где вероятнее лежит информация о индексах цен
    sitemap = urllib.request.urlopen("https://rosstat.gov.ru/sitemap.xml")
    html = str(sitemap.read())
    pages = set()

    #скорее всего в названии страницы с данными по индексам потребительских цен есть ключевое слово
    keywords = ('price','index','indice','inflat')
    for i in range(1, len(html.split('<loc>'))):
        pos = html.split('<loc>')[i].find('</loc>')
        if any(x in html.split('<loc>')[i][:pos] for x in keywords):
               # print(html.split('<loc>')[i][:pos])
                pages.add(html.split('<loc>')[i][:pos])

    #видим, что нужные данные лежат в формате .xlsx
    #качаем с этих страниц все иксели со словами "индекс" и "потребитель"
    for page in pages:
        html = urllib.request.urlopen(page).read().decode('utf-8')
        for i in range(1, len(html.split('item--row">'))):
            rowsplit = html.split('item--row">')[i]
            if 'ндекс' in rowsplit and 'отребител' in rowsplit:
                if '.xls' in rowsplit or '.csv' in rowsplit:
                    href=rowsplit.split('href="')
                    for j in range(1, len(href)):
                        if all(x in href[j] for x in ['mes' or 'mon','.xls' or '.csv']):
                            print(rowsplit.split('title">')[1].split('</div')[0].strip())
                            downloadurl = "https://rosstat.gov.ru" + rowsplit.split('href="')[j].split('">')[0]
                            print(downloadurl)
                            urllib.request.urlretrieve(downloadurl, downloadurl.split('/')[-1])
