# enhanced_product_db.py
import os
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import pandas as pd

class EnhancedProductDB:
    def __init__(self, persist_dir: str = "/tmp/chroma_store"):
        self.persist_dir = persist_dir
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        try:
            self.collection = self.client.get_collection(
                name="products",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = None

    def initialize_db(self, products: List[Dict[str, Any]]):
        """(Re)create the 'products' collection from a list of dicts."""
        try:
            self.client.delete_collection("products")
        except:
            pass

        self.collection = self.client.create_collection(
            name="products",
            embedding_function=self.embedding_function
        )

        docs, metas, ids = [], [], []
        for idx, p in enumerate(products):
            docs.append(self._create_full_description(p))
            metas.append({
                "name": p["name"],
                "price": p["price"],
                "category": p.get("category", "laptop"),
                "specs": json.dumps(p["specs"])
            })
            ids.append(f"prod_{idx}")

        self.collection.add(documents=docs, metadatas=metas, ids=ids)

    def _create_full_description(self, p: Dict[str, Any]) -> str:
        specs = ", ".join(f"{k}: {v}" for k, v in p["specs"].items())
        return f"{p['name']}: {specs}, Price: RM{p['price']:.2f}"

    def query(self, query_text: str, filter: Optional[Dict]=None, top_k: int=3) -> List[Dict[str, Any]]:
        if not self.collection:
            raise ValueError("Database not initialized")
        res = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filter,
            include=["metadatas","documents","distances"]
        )
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "content": res["documents"][0][i],
                "score": 1 - float(res["distances"][0][i]),
                "metadata": res["metadatas"][0][i],
                "id": res["ids"][0][i]
            })
        return out

    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.collection:
            raise ValueError("Database not initialized")
        results = self.collection.get()
        data = []
        for i in range(len(results["ids"])):
            md = results["metadatas"][i]
            md["specs"] = json.loads(md["specs"])
            md["id"] = results["ids"][i]
            md["content"] = results["documents"][i]
            data.append(md)
        return pd.DataFrame(data)

# ——— Hard-coded sample products ———
SAMPLE_PRODUCTS = [
    {
        "name": "Dell Inspiron 15",
        "specs": {
            "display": "15.6-inch FHD",
            "processor": "Intel Core i5",
            "ram": "8GB",
            "storage": "512GB SSD"
        },
        "price": 3200.00
    },
    {
        "name": "ASUS ROG Zephyrus",
        "specs": {
            "display": "14-inch QHD",
            "processor": "AMD Ryzen 9",
            "ram": "32GB",
            "gpu": "RTX 4060",
            "storage": "1TB SSD"
        },
        "price": 6900.00
    },
    {
        "name": "HP Pavilion x360",
        "specs": {
            "display": "14-inch FHD Touch",
            "processor": "Intel Core i7",
            "ram": "16GB",
            "storage": "1TB SSD"
        },
        "price": 4200.00
    },
    {
        "name": "MacBook Air M2",
        "specs": {
            "display": "13.6-inch Retina",
            "processor": "Apple M2",
            "ram": "8GB",
            "storage": "512GB SSD"
        },
        "price": 5600.00
    },
    {
        "name": "Lenovo Legion 5",
        "specs": {
            "display": "15.6-inch FHD",
            "processor": "AMD Ryzen 7",
            "gpu": "RTX 3060",
            "ram": "16GB",
            "storage": "512GB SSD"
        },
        "price": 5800.00
    },
    {
        "name": "Acer Swift 3",
        "specs": {
            "display": "14-inch FHD",
            "processor": "Intel Core i5",
            "ram": "8GB",
            "storage": "512GB SSD"
        },
        "price": 2900.00
    },
    {
        "name": "MSI GF63 Thin",
        "specs": {
            "display": "15.6-inch FHD",
            "processor": "Intel Core i7",
            "gpu": "GTX 1650",
            "ram": "16GB",
            "storage": "512GB SSD"
        },
        "price": 4100.00
    },
    {
        "name": "Asus Vivobook S14",
        "specs": {
            "display": "14-inch OLED",
            "processor": "Intel Core i7",
            "ram": "16GB",
            "storage": "1TB SSD"
        },
        "price": 4600.00
    },
    {
        "name": "HP Omen 16",
        "specs": {
            "display": "16.1-inch QHD",
            "processor": "Intel Core i7",
            "gpu": "RTX 4070",
            "ram": "32GB",
            "storage": "1TB SSD"
        },
        "price": 7500.00
    },
    {
        "name": "Razer Blade 15",
        "specs": {
            "display": "15.6-inch QHD",
            "processor": "Intel Core i9",
            "gpu": "RTX 4080",
            "ram": "32GB",
            "storage": "1TB SSD"
        },
        "price": 12000.00
    }
]

if __name__ == "__main__":
    db = EnhancedProductDB()
    if db.collection is None or db.collection.count() == 0:
        print("Initializing database with sample laptops…")
        db.initialize_db(SAMPLE_PRODUCTS)
        print("✅ Database initialized.")
    else:
        print(f"✅ Loaded existing database with {db.collection.count()} items.")

    # Quick query test
    print("\nQuery 'gaming laptop' top 3:")
    for res in db.query("gaming laptop", top_k=3):
        print(f"  • {res['metadata']['name']} (score {res['score']:.2f})")
