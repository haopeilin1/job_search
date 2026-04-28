import chromadb

client = chromadb.PersistentClient(path='data/chroma_db')
collections = client.list_collections()
print('Collections:', [c.name for c in collections])
for c in collections:
    col = client.get_collection(c.name)
    print(f'  {c.name}: count={col.count()}')
    if col.count() > 0:
        result = col.get(limit=3)
        for i in range(min(3, len(result['ids']))):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            print(f"    ID: {result['ids'][i]}, Company: {meta.get('company', 'N/A')}, Position: {meta.get('position', 'N/A')}")
