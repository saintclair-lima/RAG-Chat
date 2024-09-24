from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

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
                    device='cpu',
                    # tipo_de_busca='mmr',
                    # tipo_de_busca='similarity',
                    tipo_de_busca='similarity_score_threshold',
                    limiar_score_similaridade=.6,
                    numero_de_documentos_retornados=10,
                    url_llama='http://localhost:11434',
                    papel_do_LLM=None,
                    verbose=True):

        if not funcao_de_embeddings:
            from langchain_huggingface import HuggingFaceEmbeddings
            funcao_de_embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-xl",
                show_progress=True,
                model_kwargs={"device": device},
            )

        
        if verbose: print('-- Gerador de respostas em inicialização...')
        if verbose: print('--- inicializando banco de vetores...')
        self.banco_de_vetores = Chroma(
            persist_directory=url_banco_vetores,
            embedding_function=funcao_de_embeddings
        )

        if verbose: print('--- gerando retriever (gerenciador de consultas)...')
        self.gerenciador_de_consulta = self.banco_de_vetores.as_retriever(search_type=tipo_de_busca, search_kwargs={'score_threshold':limiar_score_similaridade, 'k':numero_de_documentos_retornados})

        if verbose: print('--- gerando a interface com o LLM...')
        self.interface_llama = ChatOllama(
            model="llama3.1",    # modelo llama a ser usado
            temperature=0,       # 'temperature' controla a aleatoriedade da saída do modelo, sendo 0 o valor para saída determinística
            base_url=url_llama,  # 'base_url' aponta para o end point da API do Llama
        )

        if not papel_do_LLM:
            papel_do_LLM = '''Você é um assistente de servidores que responde a dúvidas de servidores da Assembleia Legislativa do Rio Grande do Norte.
                            Você tem conhecimento apenas sobre 3 assuntos: Constituição Federal do Brasil, Constituição do Estado do Rio Grande do Norte e
                            Regimento interno da Assembleia Legislativa do Estado do Rio Grande do Norte.
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
            return "\n\n\n\n".join([f'{doc.metadata['titulo']}, {doc.page_content}' for doc in docs])

    def consultar(self, dadosChat: DadosChat, verbose=True):
        '''Recebe uma pergunta, em formato de string, realiza uma consulta no banco de vetores,
        passa os resultados para o LLM gerar uma resposta palatável e a retorna'''

        pergunta = dadosChat.pergunta
        historico_chat = []

        for item in dadosChat.historico:
            historico_chat.append(("human", item[0]))
            historico_chat.append(("ai", item[1]))

        if verbose: print('Gerador de respostas: realizando consulta...')
        if verbose: print(f'--- Pergunta feita: {pergunta}')
        marcador_tempo_inicio = time()
        documentos_retornados = self.gerenciador_de_consulta.invoke(pergunta)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')
        contexto = self.formatar_documentos_recuperados(documentos_retornados)
        prompt_llama = self.template_do_prompt.invoke({"pergunta": pergunta, "contexto": contexto, 'historico_chat': historico_chat})
        
        if verbose: print(f'--- gerando resposta com o Llama')
        marcador_tempo_inicio = time()
        resposta_llama = self.interface_llama.invoke(prompt_llama)
        marcador_tempo_fim = time()
        tempo_llama = marcador_tempo_fim - marcador_tempo_inicio
        if verbose: print(f'--- resposta gerada em ({tempo_llama} segundos)')
        resposta_formatada = self.formatador_saida.invoke(resposta_llama)

        dadosChat.historico.append(("human", pergunta))
        dadosChat.historico.append(("ai", resposta_formatada))

        return {
            'pergunta': pergunta,
            'documentos': documentos_retornados,
            'contexto': prompt_llama.messages[1].content,
            'resposta_llama': resposta_llama,
            'resposta': resposta_formatada,
            'historico': dadosChat.historico,
            'tempo_consulta': tempo_consulta,
            'tempo_llama': tempo_llama
            }