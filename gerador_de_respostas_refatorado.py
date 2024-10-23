from chromadb import chromadb, Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

import json
import environment
import asyncio
from concurrent.futures import ThreadPoolExecutor

from time import time

class DadosChat(BaseModel):
    pergunta: str
    contexto: list

class FuncaoEmbeddings(EmbeddingFunction):
    def __init__(self, model_name: str, biblioteca=SentenceTransformer):
        self.model = biblioteca(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

import requests
import json
from typing import List
class ClienteOllama:
    def __init__(self, nome_modelo: str, url_llama: str, temperature: float=0):
        self.modelo = nome_modelo
        self.url_llama = url_llama
        self.temperature = temperature

    def stream(self, prompt: str, contexto=[]):
        url = f"{self.url_llama}/api/generate"
        
        payload = {
            "model": self.modelo,
            "prompt": prompt,
            "temperature": self.temperature,
            "context": contexto
        }
        
        resposta = requests.post(url, json=payload, stream=True)
        
        resposta.raise_for_status()
        
        for fragmento in resposta.iter_content(chunk_size=None):
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
    
    def gerar_resposta_llama(self, pergunta: str, documentos: List[str], contexto=[int]):
        prompt_usuario = self.formatar_prompt_usuario(pergunta, documentos)
        prompt = self.criar_prompt_llama(prompt_usuario=prompt_usuario)
        for fragmento_resposta in self.cliente_ollama.stream(prompt=prompt, contexto=contexto):
            yield fragmento_resposta

class InterfaceChroma:
    def __init__(self,
                 url_banco_vetores=environment.URL_BANCO_VETORES,
                 colecao_de_documentos=environment.COLECAO_DE_DOCUMENTOS,
                 funcao_de_embeddings=None,
                 verboso=True):
    
        if verboso: print('--- interface do ChromaDB em inicialização')

        if not funcao_de_embeddings:
            print(f'--- criando a função de embeddings do ChromaDB com {environment.MODELO_DE_EMBEDDINGS}...')
            funcao_de_embeddings = FuncaoEmbeddings(model_name=environment.MODELO_DE_EMBEDDINGS, biblioteca=SentenceTransformer)
        
        if verboso: print('--- inicializando banco de vetores...')
        self.banco_de_vetores = chromadb.PersistentClient(path=url_banco_vetores)

        if verboso: print(f'--- definindo a coleção a ser usada ({colecao_de_documentos})...')
        self.colecao_documentos = self.banco_de_vetores.get_collection(name=colecao_de_documentos, embedding_function=funcao_de_embeddings)
    
    def consultar_documentos(self, termos_de_consulta: str, num_resultados=5):
        return self.colecao_documentos.query(query_texts=[termos_de_consulta], n_results=num_resultados)
    

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self,
                #  url_banco_vetores=environment.URL_BANCO_VETORES,
                url_banco_vetores='banco_vetores_alrn_adicional_teste',
                colecao_de_documentos=environment.COLECAO_DE_DOCUMENTOS,
                funcao_de_embeddings=None,
                verboso=True):
        self.executor = ThreadPoolExecutor(max_workers=environment.THREADPOOL_MAX_WORKERS)
        
        if verboso: print('-- Gerador de respostas em inicialização...')

        self.interface_chromadb = InterfaceChroma(url_banco_vetores, colecao_de_documentos, funcao_de_embeddings, verboso)

        # Carregando modelo e tokenizador pre-treinados
        # optou-se por não usar pipeline, por ser mais lento que usar o modelo diretamente
        # self.modelo_bert_qa_pipeline = pipeline("question-answering", model=self.modelo_bert_qa, tokenizer=self.tokenizador_bert)
        if verboso: print('--- preparando modelo e tokenizador do Bert...')
        self.modelo_bert_qa = BertForQuestionAnswering.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE)
        self.tokenizador_bert = BertTokenizer.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE)

        self.interface_ollama = InterfaceOllama(url_llama=environment.URL_LLAMA, nome_modelo=environment.MODELO_LLAMA)

    def consultar_documentos_banco_vetores(self, pergunta: str, num_resultados=5):
        return self.interface_chromadb.consultar_documentos(pergunta, num_resultados)
    
    def formatar_lista_documentos(self, documentos: dict):
        return [{'id': documentos['ids'][0][idx],
                 'distancia': documentos['distances'][0][idx],
                 'metadados': documentos['metadatas'][0][idx],
                 'conteudo': documentos['documents'][0][idx]} for idx in range(len(documentos['ids'][0]))]

    async def async_stream_wrapper(self, sync_generator):
        loop = asyncio.get_running_loop()
        for item in sync_generator:
            yield await loop.run_in_executor(self.executor, lambda x=item: x)

    def avaliar_respostas_por_documento(self, pergunta, texto_documento):
        # Optou-se por não utilizar a abordagem com pipeline por ser mais lenta
        # input = {
        #     'question': pergunta,
        #     'context': f'{documento.page_content}'
        # }
        # res = self.modelo_bert_qa_pipeline(input)
        
        inputs = self.tokenizador_bert.encode_plus(
            pergunta,
            texto_documento,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.modelo_bert_qa(**inputs)

        logits_inicio = outputs.start_logits
        logits_fim = outputs.end_logits

        probabs_inicio = torch.softmax(logits_inicio, dim=-1)
        probabs_fim = torch.softmax(logits_fim, dim=-1)

        # Melhores k posições iniciais e finais
        k = 5

        # As k posições iniciais e finais mais prováveis, junto com suas probabilidades
        top_k_inicio = torch.topk(probabs_inicio, k, dim=-1)
        top_k_fim = torch.topk(probabs_fim, k, dim=-1)

        # Conversão dos índices dos tokens para string, junto com seus scores
        respostas_possiveis_com_score = []

        # Para cada posição inicial e final
        for i in range(k):
            for j in range(k):
                idx_inicio = top_k_inicio.indices.squeeze()[i].item()
                idx_fim = top_k_fim.indices.squeeze()[j].item()
                # verifica se a posição final é maior que a inicial
                if idx_fim >= idx_inicio:
                    # recupera os strings dos tokens
                    tokens_resposta = inputs['input_ids'][0][idx_inicio:idx_fim + 1]
                    resposta = self.tokenizador_bert.decode(tokens_resposta, skip_special_tokens=True)
                    
                    # Calcula o score como o produto das probabilidades finais e iniciais (ou dos logits)
                    score = probabs_inicio[0, idx_inicio].item() * probabs_fim[0, idx_fim].item()
                    respostas_possiveis_com_score.append((resposta, score))

        # Ordena pelo score
        respostas_possiveis_com_score = sorted(respostas_possiveis_com_score, key=lambda x: x[1], reverse=True)

        return {
            'score': respostas_possiveis_com_score[0][1],
            'possiveis_respostas': respostas_possiveis_com_score
        }

    async def consultar(self, dados_chat: DadosChat, verbose=True):
        contexto = dados_chat.contexto
        pergunta = dados_chat.pergunta

        if verbose: print(f'Gerador de respostas: realizando consulta para "{pergunta}"...')

        # Recuperando documentos usando o ChromaDB
        marcador_tempo_inicio = time()
        documentos = await asyncio.to_thread(self.consultar_documentos_banco_vetores, pergunta)
        lista_documentos = self.formatar_lista_documentos(documentos)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')

        # Atribuindo scores usando Bert
        if verbose: print(f'--- aplicando scores do Bert aos documentos recuperados...')
        marcador_tempo_inicio = time()
        for documento in lista_documentos:
            avaliacao_respostas = self.avaliar_respostas_por_documento(pergunta, documento['conteudo'])
            documento['score_bert'] = avaliacao_respostas['score']
            documento['possiveis_respostas'] = avaliacao_respostas['possiveis_respostas']
        marcador_tempo_fim = time()
        tempo_bert = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- scores atribuídos ({tempo_bert} segundos)')
        
        # Gerando resposta utilizando o Llama
        if verbose: print(f'--- gerando resposta com o Llama')
        marcador_tempo_inicio = time()
        texto_resposta_llama = ''
        flag_tempo_resposta = False     
        async for item in self.async_stream_wrapper(
                self.interface_ollama.gerar_resposta_llama(
                    pergunta=pergunta,
                    documentos=documentos['documents'][0],
                    contexto=contexto)):
            
            texto_resposta_llama += item['response']
            yield item['response']
            if not flag_tempo_resposta:
                flag_tempo_resposta = True
                tempo_inicio_resposta = time() - marcador_tempo_inicio
                if verbose: print(f'----- iniciou retorno da resposta ({tempo_inicio_resposta} segundos)')

        item['response'] = texto_resposta_llama
        marcador_tempo_fim = time()
        tempo_llama = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- resposta do Llama concluída ({tempo_llama} segundos)')

        yield "CHEGOU_AO_FIM_DO_TEXTO_DA_RESPOSTA"

        # Retornando dados compilados
        yield json.dumps(
            {
                "pergunta": pergunta,
                "documentos": lista_documentos,
                "resposta_llama": item,
                "resposta": texto_resposta_llama.replace('\n\n', '\n'),
                "tempo_consulta": tempo_consulta,
                "tempo_bert": tempo_bert,
                "tempo_inicio_resposta": tempo_inicio_resposta,
                "tempo_llama_total": tempo_llama
            },
            ensure_ascii=False
        )

    