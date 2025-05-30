import chromadb
import uuid
from datetime import datetime

client = chromadb.PersistentClient(path="./chroma_store")

def store_in_db(chunk_vec, embeddings, userId):
    try:
        ids = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(len(chunk_vec)):
            ids.append(f"user_{userId}_chunk_{i}_{timestamp}")

        coll = client.get_or_create_collection("chatbotDocs")
        
        coll.add(
            documents=chunk_vec,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=[{"user_id": str(userId), "chunk_index": i, "timestamp": timestamp} 
                      for i in range(len(chunk_vec))]
        )

        print(f"✅ Added {len(chunk_vec)} chunks to ChromaDB collection")
        return ids
        
    except Exception as e:
        print(f"❌ Error storing in database: {e}")
        raise

def query_from_database(embeddings,userId):
    coll = client.get_or_create_collection("chatbotDocs")
    results = coll.query(
        query_embeddings=embeddings,
        n_results=3,
        where={"user_id": str(userId)}
    )
    return {"texts": results["documents"]}