## print('Para simplicidade, mover o arquivo para a pasta principal para executar')
print('Importando bibliotecas')
import json
from time import time
from sentence_transformers import SentenceTransformer
from chromadb import chromadb
from ..environment.environment import environment
from ..utils.utils import FuncaoEmbeddings, InterfaceOllama
import asyncio
import os
from torch import cuda
import sys

FAZER_LOG = False



URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
URL_LLAMA = 'http://localhost:11434'
MODELO_LLAMA='llama3.1'
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
URL_BANCO_VETORES=os.path.join(URL_LOCAL,"../conteudo/bancos_vetores/banco_vetores_regimento_resolucoes_rh")
NOME_COLECAO='regimento_resolucoes_rh'
DEVICE='cuda' if cuda.is_available() else 'cpu'

async def avaliar_respostas_llama(url_arq_entrada, url_arq_saida=None):
    if not url_arq_saida: url_arq_saida = url_arq_entrada
    if FAZER_LOG: print('Carregando JSON')
    with open(url_arq_entrada, 'r', encoding='utf-8') as arq:
        dados = json.load(arq)
    if FAZER_LOG: print('Criando interface Ollama')
    interface_ollama = InterfaceOllama(url_llama=URL_LLAMA, nome_modelo=MODELO_LLAMA)

    if FAZER_LOG: print('Criando cliente Chroma')
    client = chromadb.PersistentClient(path=URL_BANCO_VETORES)
    if FAZER_LOG: print('Criando função de embeddings')
    funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(nome_modelo=EMBEDDING_INSTRUCTOR, tipo_modelo=SentenceTransformer, device=DEVICE)
    if FAZER_LOG: print('Definindo Coleção')
    collection = client.get_collection(name=NOME_COLECAO, embedding_function=funcao_de_embeddings_sentence_tranformer)
    
    if FAZER_LOG: print('Processando perguntas')
    num_itens = len(dados)
    for idx in range(num_itens):
        print(f'\rPergunta {idx+1} de {num_itens}', end="")
        item = dados[idx]

        # Ignora Cada item que já tem uma resposta do llama
        if 'llama' in item: continue

        pergunta = item['pergunta']
        if FAZER_LOG: print('Recuperando documentos')
        documentos = collection.get(
            ids=[doc['id'] for doc in item['documentos']]
        )
        if FAZER_LOG: print('Enviando dados para o ollama')
        texto_resposta_llama = ''
        async for resp_llama in interface_ollama.gerar_resposta_llama(
                    pergunta=pergunta,
                    # Inclui o título dos documentos no prompt do Llama
                    documentos=[f"{doc[0]['titulo']} - {doc[1]}" for doc in zip(documentos['metadatas'], documentos['documents'])],
                    contexto=[]):
            
            texto_resposta_llama += resp_llama['response']

        resp_llama['response'] = texto_resposta_llama
        resp_llama['context'] = []

        item['llama'] = resp_llama

        if FAZER_LOG: print('salvando json')
        with open(os.path.join(url_arq_saida), 'w', encoding='utf-8') as arq:
            arq.write(json.dumps(dados, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    url_json_entrada = sys.argv[1]
    asyncio.run(avaliar_respostas_llama(url_arq_entrada=url_json_entrada))
else:
    url_json_entrada = sys.argv[1]
    avaliar_respostas_llama(url_arq_entrada=url_json_entrada)