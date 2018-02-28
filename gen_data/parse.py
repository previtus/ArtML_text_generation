# download data from https://www.kaggle.com/snapcrack/all-the-news/data

import sys
import csv

csv.field_size_limit(sys.maxsize)
 
data = []
attributes = {}
with open('articles1.csv') as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	i = 0
	for row in csvReader:
		if i is 0:	
			print("attributes",row)
			for j,val in enumerate(row):
				attributes[val] = j
				data.append([])

			print(data)
		else:
			for j,val in enumerate(row):
				data[j].append(val)

		i =+ 1

titles = data[attributes['title']]
articles = data[attributes['content']]
print("Titles and article len:",len(titles), len(articles))

titles_str = "\n".join(titles)
articles_str = "\n".join(articles)

with open("titles.txt", "w") as text_file:
    text_file.write(titles_str)

with open("articles.txt", "w") as text_file:
    text_file.write(articles_str)
