from chromadb import chromadb, Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import httpx
import json
import environment
from typing import List


class DadosChat(BaseModel):
    pergunta: str
    contexto: list

class FuncaoEmbeddings(EmbeddingFunction):
    def __init__(self, model_name: str, biblioteca=SentenceTransformer):
        self.model = biblioteca(model_name, cache_folder=environment.URL_MODEL_CACHE)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()
    
class ClienteOllama:
    def __init__(self, nome_modelo: str, url_llama: str, temperature: float=0):
        self.modelo = nome_modelo
        self.url_llama = url_llama
        self.temperature = temperature

    async def stream(self, prompt: str, contexto=[]):
        url = f"{self.url_llama}/api/generate"
        
        payload = {
            "model": self.modelo,
            "prompt": prompt,
            "temperature": self.temperature,
            "context": contexto,
            "stream": True
        }
        
        # Using httpx in synchronous mode
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=120) as resposta:
                resposta.raise_for_status()

                async for fragmento in resposta.aiter_bytes():
                    if fragmento:
                        yield json.loads(fragmento.decode())

class InterfaceOllama:
    def __init__(self, nome_modelo: str, url_llama: str, temperature: float=0):

        self.cliente_ollama = ClienteOllama(url_llama= url_llama, nome_modelo=nome_modelo, temperature=temperature)

        self.papel_do_LLM = '''\
            ALERN e ALRN significam Assembleia Legislativa do Estado do Rio Grande do Norte. \
            Você é um assistente que responde a dúvidas de servidores da ALERN. \
            Você tem conhecimento sobre o regimento interno da ALRN, o regime jurídico dos servidores estaduais do RN, bem como resoluções da ALRN. \
            Assuma um tom formal, porém caloroso, com gentileza nas respostas. \
            Utilize palavras e termos que sejam claros, autoexplicativos e linguagem simples, próximo do que o cidadão comum utiliza.'''
        
        self.diretrizes = ''' \
            Use as informações dos DOCUMENTOS fornecidos para gerar uma resposta clara para a PERGUNTA. \
            Na resposta, não mencione que foi fornecido um texto, agindo como se o contexto fornecido fosse parte do seu conhecimento próprio. \
            Quando adequado, pode citar os nomes dos DOCUMENTOS e números dos artigos em que a resposta se baseia. \
            A resposta não deve ter saudação, vocativo, nem qualquer tipo de introdução que dê a entender que não houve interação anterior. \
            Se você não souber a resposta, assuma um tom gentil e diga que não tem informações suficientes para responder.'''

    def formatar_prompt_usuario(self, pergunta: str, documentos: List[str]):
        return f'''\
            DOCUMENTOS:\n\
            {'\n'.join(documentos)}\n\
            PERGUNTA: {pergunta}'''

    def criar_prompt_llama(self, prompt_usuario: str):
        definicoes_sistema = f'''{self.papel_do_LLM}\n\
            DIRETRIZES PARA AS RESPOSTAS: {self.diretrizes}'''
        return f'<s>[INST]<<SYS>>\n{definicoes_sistema}\n<</SYS>>\n\n{prompt_usuario}[/INST]'
    
    async def gerar_resposta_llama(self, pergunta: str, documentos: List[str], contexto=[int]):
        prompt_usuario = self.formatar_prompt_usuario(pergunta, documentos)
        prompt = self.criar_prompt_llama(prompt_usuario=prompt_usuario)
        async for fragmento_resposta in self.cliente_ollama.stream(prompt=prompt, contexto=contexto):
            yield fragmento_resposta

class InterfaceChroma:
    def __init__(self,
                 url_banco_vetores=environment.URL_BANCO_VETORES,
                 colecao_de_documentos=environment.NOME_COLECAO_DE_DOCUMENTOS,
                 funcao_de_embeddings=None,
                 fazer_log=True):
    
        if fazer_log: print('--- interface do ChromaDB em inicialização')

        if not funcao_de_embeddings:
            print(f'--- criando a função de embeddings do ChromaDB com {environment.MODELO_DE_EMBEDDINGS}...')
            funcao_de_embeddings = FuncaoEmbeddings(model_name=environment.MODELO_DE_EMBEDDINGS, biblioteca=SentenceTransformer)
        
        if fazer_log: print('--- inicializando banco de vetores...')
        self.banco_de_vetores = chromadb.PersistentClient(path=url_banco_vetores)

        if fazer_log: print(f'--- definindo a coleção a ser usada ({colecao_de_documentos})...')
        self.colecao_documentos = self.banco_de_vetores.get_collection(name=colecao_de_documentos, embedding_function=funcao_de_embeddings)
    
    def consultar_documentos(self, termos_de_consulta: str, num_resultados=5):
        return self.colecao_documentos.query(query_texts=[termos_de_consulta], n_results=num_resultados)