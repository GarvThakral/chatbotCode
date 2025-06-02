from sentence_transformers import SentenceTransformer

# This will be the globally loaded model instance
# It will be loaded only when this module is first imported
embedding_model_instance = None

def load_embedding_model():
    """
    Loads the SentenceTransformer model if it hasn't been loaded already.
    This function should be called during application startup.
    """
    global embedding_model_instance
    if embedding_model_instance is None:
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        embedding_model_instance = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully")
    return embedding_model_instance