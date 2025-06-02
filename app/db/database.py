import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime

client = chromadb.PersistentClient(path="./chroma_store")

def store_in_db(chunk_vec, embeddings, userId):
    try:
        userId = str(userId).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ids = [f"user_{userId}_chunk_{i}_{timestamp}" for i in range(len(chunk_vec))]
        metadatas = [{"user_id": userId, "chunk_index": i, "timestamp": timestamp}
                     for i in range(len(chunk_vec))]

        coll = client.get_or_create_collection("chatbotDocs")
        coll.add(
            documents=chunk_vec,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(chunk_vec)} chunks for user: {userId}")
        return ids
    except Exception as e:
        print(f"❌ DB store error: {e}")
        raise

def query_from_database(embeddings, userId):
    userId = str(userId).lower()
    coll = client.get_or_create_collection("chatbotDocs")

    print(f"🔍 Looking for documents for user_id={userId}...")
    all_user_docs = coll.get(where={"user_id": userId})

    if not all_user_docs["metadatas"]:
        print("⚠️ No documents found for user.")
        return {"texts": []}

    print("📄 Matching stored docs:", all_user_docs["metadatas"])
    results = coll.query(
        query_embeddings=embeddings,
        n_results=3,
        where={"user_id": userId}
    )
    return {"texts": results["documents"]}

def delete_user_embeddings(userId):
    userId = str(userId).lower()
    coll = client.get_or_create_collection("chatbotDocs")
    user_docs = coll.get(where={"user_id": userId})
    if not user_docs["ids"]:
        print(f"⚠️ No embeddings found for user: {userId}")
        return False

    coll.delete(ids=user_docs["ids"])
    print(f"🗑️ Deleted {len(user_docs['ids'])} embeddings for user: {userId}")
    return True

def has_user_embeddings(userId):
    userId = str(userId).lower()
    coll = client.get_or_create_collection("chatbotDocs")
    docs = coll.get(where={"user_id": userId})
    return len(docs["ids"]) > 0

def get_all_user_ids():
    coll = client.get_or_create_collection("chatbotDocs")
    all_docs = coll.get()
    user_ids = {meta["user_id"] for meta in all_docs["metadatas"] if "user_id" in meta}
    return list(user_ids)

def clear_database():
    coll = client.get_or_create_collection("chatbotDocs")
    all_docs = coll.get()
    coll.delete(ids=all_docs["ids"])
    print("💣 All documents cleared from the collection.")
