import os

import faiss
import numpy as np

from llms.clients.gpt import GPTClient
from llms.settings import settings

client = GPTClient(
    client_id=settings.CLIENT_ID,
    client_secret=settings.CLIENT_SECRET,
    auth_url=settings.AUTH_URL,
    api_base=settings.API_BASE,
    deployment_id="gpt-4-32k",
    max_response_tokens=1000,
    temperature=0.0,
)
folder_paths = [
    "D:/Uni/5_Semester/NLP/llms-using-rag-for-docs/pandas_doku/textfiles/textfiles1",
    "D:/Uni/5_Semester/NLP/llms-using-rag-for-docs/pandas_doku/textfiles/textfiles2",
    "D:/Uni/5_Semester/NLP/llms-using-rag-for-docs/pandas_doku/textfiles/textfiles3",
]
all_embeddings = []

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            print(f"File found: {filename}")

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            content = content.replace("\n", " ")
            response = client.get_embedding(content)
            embedding = response.json()["data"][0]["embedding"]
            embedding_np = np.array(embedding)

            all_embeddings.append(embedding_np)
            print(f"Done for file: {filename}")

all_embeddings_array = np.vstack(all_embeddings)

embedding_dim = all_embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings_array)
faiss.write_index(index, "embeddings.index")
