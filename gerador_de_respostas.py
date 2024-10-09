from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
import json
import var_ambiente

import asyncio
from concurrent.futures import ThreadPoolExecutor

from time import time

class DadosChat(BaseModel):
    pergunta: str
    historico: list

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self, url_banco_vetores,
                    funcao_de_embeddings=None,
                    device=var_ambiente.DEVICE,
                    tipo_de_busca=var_ambiente.TIPO_DE_BUSCA,                   
                    url_llama=var_ambiente.URL_LLAMA,
                    papel_do_LLM=None,
                    numero_de_documentos_retornados=var_ambiente.NUM_DOCUMENTOS_RETORNADOS,
                    limiar_score_similaridade=var_ambiente.LIMIAR_SCORE_SIMILARIDADE,
                    verbose=True):
        
        self.executor = ThreadPoolExecutor(max_workers=var_ambiente.THREADPOOL_MAX_WORKERS)
        # If you're deploying this API using a web server like uvicorn or gunicorn, consider increasing the number of workers in production when deploying the API.
        # Run on bash: uvicorn main:app --workers 5
        
        if tipo_de_busca == 'mmr':
            argumentos_de_busca={'k':numero_de_documentos_retornados}
        elif tipo_de_busca == 'similarity':
            argumentos_de_busca={'k':numero_de_documentos_retornados}
        elif tipo_de_busca == 'similarity_score_threshold':
            argumentos_de_busca={'score_threshold':limiar_score_similaridade, 'k':numero_de_documentos_retornados}

        if not funcao_de_embeddings:
            print(f'Criando a função de embeddings com {var_ambiente.MODELO_DE_EMBEDDINGS}')
            from langchain_huggingface import HuggingFaceEmbeddings
            funcao_de_embeddings = HuggingFaceEmbeddings(
                model_name=var_ambiente.MODELO_DE_EMBEDDINGS,
                show_progress=True,
                model_kwargs={"device": var_ambiente.DEVICE},
            )

        
        if verbose: print('-- Gerador de respostas em inicialização...')
        if verbose: print('--- inicializando banco de vetores...')
        self.banco_de_vetores = Chroma(
            persist_directory=url_banco_vetores,
            embedding_function=funcao_de_embeddings
        )

        if verbose: print('--- gerando retriever (gerenciador de consultas)...')
        self.gerenciador_de_consulta = self.banco_de_vetores.as_retriever(search_type=tipo_de_busca, search_kwargs=argumentos_de_busca)

        if verbose: print('--- gerando a interface com o LLM...')
        self.interface_llama = ChatOllama(
            model="llama3.1",    # modelo llama a ser usado
            temperature=0,       # 'temperature' controla a aleatoriedade da saída do modelo, sendo 0 o valor para saída determinística
            base_url=url_llama,  # 'base_url' aponta para o end point da API do Llama
        )

        if not papel_do_LLM:
            papel_do_LLM = '''Você é um assistente de servidores que responde a dúvidas de servidores da Assembleia Legislativa do Rio Grande do Norte.
                            Você tem conhecimento sobre o regimento interno da ALRN, o regime jurídico dos servidores estaduais do RN, bem como resoluções da ALRN.
                            ALERN e ALRN significam Assembleia Legislativa do Estado do Rio Grande do Norte.
                            Use as informações do contexto fornecido para gerar uma resposta clara para a pergunta.
                            Na resposta, não mencione que foi fornecido um texto, agindo como se o contexto fornecido fosse parte do seu conhecimento próprio.
                            Quando adequado, pode citar os nomes dos documentos e números dos artigos em que a resposta se baseia.
                            A resposta não deve ter saudação, nem qualquer tipo de introdução que dê a entender que não houve interação anterior.
                            Assuma um tom formal, porém caloroso, com gentileza nas respostas.
                            Utilize palavras e termos que sejam claros, autoexplicativos e linguagem simples, próximo do que o cidadão comum utiliza.
                            Se você não souber a resposta, assuma um tom gentil e diga que não tem informações suficientes para responder.'''

        if verbose: print('--- gerando template de respostas...')
        self.template_do_prompt = ChatPromptTemplate.from_messages(
            [    # Estabelece o papel que o LLM vai assumir ao responder as perguntas. Pode incluir o "tom" das respostas
                ('system', papel_do_LLM),  

                # Placeholder para o histórico do chat manter o contexto. Durante a execução será substituído pelo histórico real do chat
                ("system", "HISTORICO (perguntas anteriores, não devem ser respondidas, só usadas para contexto) {historico_chat}"), 

                # Placeholder para o input a ser fornecido durante a execução
                # Será substituído pela pergunta do usuário e o contexto vindo do banco de vetores
                ("human", "CONTEXTO: {contexto} \n\nPERGUNTA: {pergunta}"),
            ]
        )


        # 'StrOutputParser' é usado para tratar a saída do chat, convertendo em um string
        self.formatador_saida = StrOutputParser()

        if verbose: print('--- inicialização completa!')

    def formatar_documentos_recuperados(self, docs):
            '''Função de formatação dos documentos. 'docs' é uma lista de objetos do tipo langchain_core.documents.Document'''
            return "\n\n\n\n".join([f'{doc.metadata["titulo"]}, {doc.page_content}' for doc in docs])
    
    async def async_stream_wrapper(self, sync_generator):
        """Run the synchronous generator in an executor to yield its items as they become available."""
        loop = asyncio.get_running_loop()
        for item in sync_generator:
            yield await loop.run_in_executor(self.executor, lambda x=item: x)
    
    async def gerar_resposta(self, dadosChat: DadosChat):
        # enviando a resposta por streaming ao usuário
        return StreamingResponse(self.consultar(dadosChat), media_type="text/plain")

    async def consultar(self, dadosChat: DadosChat, verbose=True):
        '''Recebe uma pergunta, em formato de string, realiza uma consulta no banco de vetores,
        passa os resultados para o LLM gerar uma resposta palatável e a retorna'''

        pergunta = dadosChat.pergunta
        historico_chat = [(("human", item[0]), ("ai", item[1])) for item in dadosChat.historico]


        for item in dadosChat.historico:
            historico_chat.append(("human", item[0]))
            historico_chat.append(("ai", item[1]))

        if verbose: print('Gerador de respostas: realizando consulta...')
        if verbose: print(f'--- Pergunta feita: {pergunta}')
        marcador_tempo_inicio = time()
        # documentos_retornados = self.gerenciador_de_consulta.invoke(pergunta)
        documentos_retornados = await asyncio.to_thread(self.gerenciador_de_consulta.invoke, pergunta)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')
        # contexto = self.formatar_documentos_recuperados(documentos_retornados)
        contexto = await asyncio.to_thread(self.formatar_documentos_recuperados, documentos_retornados)
        prompt_llama = self.template_do_prompt.invoke({"pergunta": pergunta, "contexto": contexto, 'historico_chat': historico_chat})
        
        if verbose: print(f'--- gerando resposta com o Llama')
        marcador_tempo_inicio = time()
        texto_resposta_llama = ''

        # Wrap the synchronous stream in an asynchronous generator
        async for item in self.async_stream_wrapper(self.interface_llama.stream(prompt_llama)):
            texto_resposta_llama += item.content
            yield item.content
        
        item.content = texto_resposta_llama
        resposta_llama = item
        marcador_tempo_fim = time()
        tempo_llama = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- resposta gerada em ({tempo_llama} segundos)')
        resposta_formatada = self.formatador_saida.invoke(resposta_llama)

        dadosChat.historico.append(("human", pergunta))
        dadosChat.historico.append(("ai", resposta_formatada))
        yield "CHEGOU_AO_FIM_DO_TEXTO_DA_RESPOSTA"
        yield json.dumps(
            {
                "pergunta": pergunta,
                "documentos": [item.model_dump_json() for item in documentos_retornados],
                "contexto": prompt_llama.messages[1].content,
                "resposta_llama": resposta_llama.model_dump_json(),
                "resposta": resposta_formatada.replace('\n\n', '\n'),
                "historico": dadosChat.historico,
                "tempo_consulta": tempo_consulta,
                "tempo_llama": tempo_llama
                }
            )