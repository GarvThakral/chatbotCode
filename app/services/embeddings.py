# importing required classes
from pypdf import PdfReader
from nltk import sent_tokenize
from app.db.database import store_in_db
from langchain_cohere import CohereEmbeddings

def read_and_embedd(fileLocation,id):
    reader = PdfReader(fileLocation)
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
    
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    embeddings = embeddings.embed_documents(chunk_vec)
    store_in_db(chunk_vec,embeddings,id)
