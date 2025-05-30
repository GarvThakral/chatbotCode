# importing required classes
from pypdf import PdfReader
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

reader = PdfReader('example.pdf')

page = reader.get_num_pages()

pages_text = []
for i in range(page):
    pages_text.append(reader.pages[i].extract_text())

vector_arr = []

for i in range(len(pages_text)):
    pages_text[i] = sent_tokenize(pages_text)

print(pages_text[0])