## print('Para simplicidade, mover o arquivo para a pasta principal para executar')
print('Importando bibliotecas')
import json
from time import time
from sentence_transformers import SentenceTransformer
from chromadb import chromadb
import api.environment.environment as environment
from utils.utils import FuncaoEmbeddings, InterfaceOllama
import asyncio

FAZER_LOG = False

async def avaliar_respostas_llama(url_arq_perg_docs):
    if FAZER_LOG: print('Carregando JSON')
    with open(url_arq_perg_docs, 'r', encoding='utf-8') as arq:
        dados = json.load(arq)
    if FAZER_LOG: print('Criando interface Ollama')
    interface_ollama = InterfaceOllama(url_llama=environment.URL_LLAMA, nome_modelo=environment.MODELO_LLAMA)
    persist_directory = environment.URL_BANCO_VETORES

    if FAZER_LOG: print('Criando cliente Chroma')
    client = chromadb.PersistentClient(path=persist_directory)
    if FAZER_LOG: print('Criando função de embeddings')
    funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(nome_modelo=environment.MODELO_DE_EMBEDDINGS, tipo_modelo=SentenceTransformer, device=environment.DEVICE)
    if FAZER_LOG: print('Definindo Coleção')
    collection = client.get_collection(name='legisberto', embedding_function=funcao_de_embeddings_sentence_tranformer)
    
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
        with open('./testes/testes_llama_10_docs.json', 'w', encoding='utf-8') as arq:
            arq.write(json.dumps(dados, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    asyncio.run(avaliar_respostas_llama('./testes/testes_llama_10_docs.json'))
else:
    avaliar_respostas_llama('./testes/testes_llama_10_docs.json')