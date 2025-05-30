# importing required classes
from pypdf import PdfReader
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from app.db.database import store_in_db


# def embedd_file(filename):
reader = PdfReader('./app/services/example.pdf')

num_pages = reader.get_num_pages()

pages_text = []
for i in range(num_pages):
    pages_text.append(reader.pages[i].extract_text())

pages_text_joined = " ".join(pages_text)

sentences = sent_tokenize(pages_text_joined)

chunk = ""
chunk_vec = []


for sentence in sentences:
    if(len(sentence+chunk) <= 500):
        chunk += sentence
    else:
        chunk_vec.append(chunk)
        chunk = sentence

chunk_vec.append(chunk)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunk_vec)

store_in_db(chunk_vec,embeddings,21)
