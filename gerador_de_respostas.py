import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import time

from utils import InterfaceChroma, InterfaceOllama, DadosChat
import environment
    

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self,
                url_banco_vetores=environment.URL_BANCO_VETORES,
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

    