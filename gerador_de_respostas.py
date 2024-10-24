import torch
from numpy import argmax, average, mean, median
from transformers import BertTokenizer, BertForQuestionAnswering
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import time

from utils import InterfaceChroma, InterfaceOllama, DadosChat
import environment
from typing import Callable, Generator
    

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self,
                url_banco_vetores:str=environment.URL_BANCO_VETORES,
                colecao_de_documentos:str=environment.NOME_COLECAO_DE_DOCUMENTOS,
                funcao_de_embeddings:Callable=None,
                fazer_log:bool=True):
        self.executor = ThreadPoolExecutor(max_workers=environment.THREADPOOL_MAX_WORKERS)
        
        if fazer_log: print('-- Gerador de respostas em inicialização...')

        self.interface_chromadb = InterfaceChroma(url_banco_vetores, colecao_de_documentos, funcao_de_embeddings, fazer_log)

        # Carregando modelo e tokenizador pre-treinados
        # optou-se por não usar pipeline, por ser mais lento que usar o modelo diretamente
        # self.modelo_bert_qa_pipeline = pipeline("question-answering", model=self.modelo_bert_qa, tokenizer=self.tokenizador_bert)
        if fazer_log: print('--- preparando modelo e tokenizador do Bert...')
        self.modelo_bert_qa = BertForQuestionAnswering.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE)
        self.tokenizador_bert = BertTokenizer.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE)

        self.interface_ollama = InterfaceOllama(url_llama=environment.URL_LLAMA, nome_modelo=environment.MODELO_LLAMA)

    def consultar_documentos_banco_vetores(self, pergunta: str, num_resultados:int=environment.NUM_DOCUMENTOS_RETORNADOS):
        return self.interface_chromadb.consultar_documentos(pergunta, num_resultados)
    
    def formatar_lista_documentos(self, documentos: dict):
        return [{'id': documentos['ids'][0][idx],
                 'distancia': documentos['distances'][0][idx],
                 'metadados': documentos['metadatas'][0][idx],
                 'conteudo': documentos['documents'][0][idx]} for idx in range(len(documentos['ids'][0]))]

    async def async_stream_wrapper(self, sync_generator: Generator):
        loop = asyncio.get_running_loop()
        for item in sync_generator:
            yield await loop.run_in_executor(self.executor, lambda x=item: x)

    def estimar_resposta(self, pergunta, texto_documento: str):
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

        # AFAZER: Avaliar se score ponderado faz sentido
        logits_inicio = outputs.start_logits.numpy()
        media_logits_inicio_positivos = average([logit for logit in logits_inicio[0] if logit > 0])
        indice_melhor_logit_inicio = argmax(logits_inicio[0])
        melhor_logit_inicio = logits_inicio[0][indice_melhor_logit_inicio]

        logits_fim = outputs.end_logits.numpy()
        media_logits_fim_positivos = average([logit for logit in logits_fim[0] if logit > 0])
        indice_melhor_logit_fim = argmax(logits_fim[0])
        melhor_logit_fim = logits_fim[0][indice_melhor_logit_fim]

        media_logits_positivos = average([media_logits_inicio_positivos, media_logits_fim_positivos])

        score = melhor_logit_inicio + melhor_logit_fim
        score_ponderado = score * media_logits_positivos

        tokens_resposta = inputs['input_ids'][0][indice_melhor_logit_inicio:indice_melhor_logit_fim + 1]
        resposta = self.tokenizador_bert.decode(tokens_resposta, skip_special_tokens=True)

        return {
            'score': float(score),
            'score_ponderado': float(score_ponderado),
            'logits': [float(melhor_logit_inicio), float(melhor_logit_fim)],
            'resposta': resposta
        }

    async def consultar(self, dados_chat: DadosChat, fazer_log:bool=True):
        contexto = dados_chat.contexto
        pergunta = dados_chat.pergunta

        if fazer_log: print(f'Gerador de respostas: realizando consulta para "{pergunta}"...')

        # Recuperando documentos usando o ChromaDB
        marcador_tempo_inicio = time()
        documentos = await asyncio.to_thread(self.consultar_documentos_banco_vetores, pergunta)
        lista_documentos = self.formatar_lista_documentos(documentos)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if fazer_log: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')

        # Atribuindo scores usando Bert
        if fazer_log: print(f'--- aplicando scores do Bert aos documentos recuperados...')
        marcador_tempo_inicio = time()
        for documento in lista_documentos:
            resposta_estimada = self.estimar_resposta(pergunta, documento['conteudo'])
            documento['score_bert'] = resposta_estimada['score']
            # documento['score_ponderado'] = resposta_estimada['score_ponderado']
            documento['logits'] = resposta_estimada['logits']
            documento['resposta_bert'] = resposta_estimada['resposta']
        marcador_tempo_fim = time()
        tempo_bert = marcador_tempo_fim - marcador_tempo_inicio
        if fazer_log: print(f'--- scores atribuídos ({tempo_bert} segundos)')
        
        # Gerando resposta utilizando o Llama
        if fazer_log: print(f'--- gerando resposta com o Llama')
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
                if fazer_log: print(f'----- iniciou retorno da resposta ({tempo_inicio_resposta} segundos)')

        item['response'] = texto_resposta_llama
        marcador_tempo_fim = time()
        tempo_llama = marcador_tempo_fim - marcador_tempo_inicio
        if fazer_log: print(f'--- resposta do Llama concluída ({tempo_llama} segundos)')

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

    