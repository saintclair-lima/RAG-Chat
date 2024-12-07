from sentence_transformers import SentenceTransformer
from ..environment.environment import environment
from ..utils.utils import FuncaoEmbeddings

import chromadb

client = chromadb.PersistentClient(path='api/conteudo/bancos_vetores/banco_vetores_500')
funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(nome_modelo="hkunlp/instructor-xl", tipo_modelo=SentenceTransformer, device='cpu')
c2 = client.get_collection(name='legisberto2', embedding_function=funcao_de_embeddings_sentence_tranformer)
l2 = c2.get(include=['embeddings', 'documents','metadatas'])
c1 = client.get_collection(name='legisberto4', embedding_function=funcao_de_embeddings_sentence_tranformer)
l1 = c1.get(include=['embeddings', 'documents', 'metadatas'])

dados = {'l1': l1, 'l2': l2}

import pickle
with open('comparacao.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dados, f, pickle.HIGHEST_PROTOCOL)

client._system.stop()