from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from time import time

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self, url_vector_store,
                    funcao_de_embeddings=None,
                    device='cpu',
                    tipo_de_busca='mmr',
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
            persist_directory=url_vector_store,
            embedding_function=funcao_de_embeddings
        )

        if verbose: print('--- gerando retriever (gerenciador de consultas)...')
        self.gerenciador_de_consulta = self.banco_de_vetores.as_retriever(search_type=tipo_de_busca)

        if verbose: print('--- gerando a interface com o LLM...')
        self.interface_llama = ChatOllama(
            model="llama3.1", # modelo llama a ser usado
            temperature=0,  # 'temperature' controla a aleatoriedade da saída do modelo, sendo 0 o valor para saída determinística
            base_url=url_llama,  # 'base_url' aponta para o end point da API do Llama
        )

        if not papel_do_LLM:
            papel_do_LLM = '''Você é um assistente de servidores que responde a dúvidas sobre a Assembleia legislativa do Rio Grande do Norte.
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
                ("placeholder", "{chat_history}"), 

                # Placeholder para o input a ser fornecido durante a execução
                # Será substituído pela pergunta do usuário e o contexto vindo do banco de vetores
                ("human", "CONTEXTO: {contexto} \n\nPERGUNTA: {pergunta}"),
            ]
        )
        
        # if verbose: print('--- gerando o runnable base e formatador de saída...')
        # # Esse runnable faz o tratamento da pergunta e do contexto para o prompt
        # self.base_runnable = (
        #     {
        #         # 'contexto' faz ouso do retrtiever/gerenciador de consulta para obter contexto do banco de vetores
        #         # 'gerenciador_de_consulta' faz a busca no banco de vetores com base na pergunta/query
        #         # 'formatar_documentos_recuperados' formata os resultados
        #         "contexto": self.gerenciador_de_consulta | self.formatar_documentos_recuperados,
        #         # 'pergunta' será passada para o retriever/gerenciador de consulta como query
        #         "pergunta": RunnablePassthrough(),
        #     }
        #     # combina o contexto e a pergunta no template do prompt
        #     | self.template_do_prompt
        # )


        # 'StrOutputParser' é usado para tratar a saída do chat, convertendo em um string
        self.formatador_saida = StrOutputParser()

        # if verbose: print('--- defiindo o pipeline...')
        # self.pipeline_rag = (
        #     self.base_runnable  # Usa the base runnable com os embeddings
        #     | self.interface_llama  # Envia os resultados para o LLama, para geração da resposta
        #     | self.formatador_saida  # Converte a saída do modelo em um formato de string
        # )

        if verbose: print('--- inicialização completa!')

    def formatar_documentos_recuperados(self, docs):
            '''Função de formatação dos documentos. 'docs' é uma lista de objetos do tipo langchain_core.documents.Document'''
            return "\n\n\n\n".join([doc.page_content for doc in docs])

    def consultar(self, pergunta, verbose=True):
        '''Recebe uma pergunta, em formato de string, realiza uma consulta no banco de vetores,
        passa os resultados para o LLM gerar uma resposta palatável e a retorna'''

        if verbose: print('Gerador de respostas: realizando consulta...')
        if verbose: print(f'--- Pergunta feita: {pergunta}')
        marcador_tempo_1 = time()
        consulta_result = self.gerenciador_de_consulta.invoke(pergunta)
        marcador_tempo_2 = time()
        tempo_consulta = marcador_tempo_2 - marcador_tempo_1
        if verbose: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')
        contexto = self.formatar_documentos_recuperados(consulta_result)
        runnable_output = self.template_do_prompt.invoke({"pergunta": pergunta, "contexto": contexto})
        
        if verbose: print(f'--- gerando resposta com o Llama')
        llama_response = self.interface_llama.invoke(runnable_output)
        marcador_tempo_3 = time()
        tempo_llama = marcador_tempo_3 - marcador_tempo_2
        if verbose: print(f'--- resposta gerada em ({tempo_llama} segundos)')
        formatted_output = self.formatador_saida.invoke(llama_response)

        return {
            'pergunta': pergunta,
            'documentos': consulta_result,
            'contexto': runnable_output.messages[1].content,
            'resposta_llama': llama_response,
            'saida': formatted_output,
            'tempo_consulta': tempo_consulta,
            'tempo_llama': tempo_llama
            }