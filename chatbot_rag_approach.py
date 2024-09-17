print(
"""Inicializando a aplicação...
Definindo os caminhos e lendo o arquivo..."""
)

URL_DADOS = "datasets/regimento_alrn.txt"
URL_CHROMADB = "chroma_db_instructor_xl"
CARREGAR_CHROMADB_EXISTENTE = True
# DEVICE = "cpu"
# DEVICE = 'cuda'

import torch_directml
dml = torch_directml.device()
DEVICE = str(dml)
print(dml)

# with open(URL_DADOS, "r", encoding="latin-1") as arq:
#     texto = arq.read()

# # Listagens até 9, em português, são marcadas por números ordinais. Isso faz com que
# # posteriormente os números ordinais recebam mais atenção do que deveriam, na
# # representação TF-IDF. O código abaixo remove a marcação de ordinais e coloca a
# # mesma notação utilizada nos demais itens

# for num in range(1, 10):
#     texto = texto.replace(f"Art. {num}º", f"Art. {num}.")
#     texto = texto.replace(f"art. {num}º", f"art. {num}.")
#     texto = texto.replace(f"§ {num}º", f"§ {num}.")
# texto = texto.split("\n")
# texto.remove("")
# artigos = []

# for art in texto:
#     item = art.split(" ")
#     qtd_palavras = len(item)
#     if qtd_palavras > 500:
#         item = (
#             art.replace(". §", ".\n§")
#             .replace("; §", ";\n§")
#             .replace(": §", ":\n§")
#             .replace(";", "\n")
#             .replace(":", "\n")
#             .replace("\n ", "\n")
#             .replace(" \n", "\n")
#             .split("\n")
#         )
#         caput = item[0]
#         fragmento_artigo = "" + caput
#         for i in range(1, len(item)):
#             if len(fragmento_artigo.split(" ")) + len(item[i]) <= 500:
#                 fragmento_artigo = fragmento_artigo + " " + item[i]
#             else:
#                 artigos.append(fragmento_artigo)
#                 fragmento_artigo = "" + caput + " " + item[i]
#         artigos.append(fragmento_artigo)
#     else:
#         artigos.append(art)

# print("""Criando os documentos com base nos artigos...""")

# from langchain_core.documents import Document

# documentos = []
# titulos = []
# for artigo in artigos:
#     tit = artigo.split(". ")[1]
#     titulos.append(tit)
#     doc = Document(
#         page_content=artigo,
#         metadata={
#             "title": f"Regimento Interno - Artigo {tit}_{titulos.count(tit)}",
#             "author": "ALERN",
#             "source": "https://www.google.com/url?q=https%3A%2F%2Fwww.al.rn.leg.br%2Fregimento-interno%2FRegimento_Interno_ALRN_junho_2024_DL.pdf",
#         },
#     )
#     documentos.append(doc)


# print("""Definindo o hkunlp/instructor-xl""")
# from langchain_huggingface import HuggingFaceEmbeddings

# embedding_function_instructor_xl = HuggingFaceEmbeddings(
#     model_name="hkunlp/instructor-xl",
#     show_progress=True,
#     model_kwargs={"device": DEVICE},
# )

# print("""Inicializando o vector store""")
# from langchain_chroma import Chroma

# if CARREGAR_CHROMADB_EXISTENTE:
#     # Carrega a instância ChromaDB existente a partir do diretório especificado
#     # 'persist_directory' especifica o diretório onde a instância ChromaDB será salva
#     db_instructor_xl = Chroma(
#         persist_directory=URL_CHROMADB,
#         embedding_function=embedding_function_instructor_xl,
#     )
# else:
#     # Inicializa a instância ChromaDB usando o método 'from_documents'
#     # 'documents' é uma lista de fragmentos de documentos que serão armazenados no banco de dados
#     # 'embedding' é a função de embedding usada para gerar embeddings dos documentos
#     # 'persist_directory' especifica o diretório onde a instância ChromaDB será salva
#     db_instructor_xl = Chroma.from_documents(
#         documents=documentos,
#         embedding=embedding_function_instructor_xl,
#         persist_directory=URL_CHROMADB,
#     )
# print("""Inicializando o retriever...""")
# retriever_instructor_xl = db_instructor_xl.as_retriever(search_type="mmr")
# # retriever_instructor_xl = db_instructor_xl.as_retriever(search_type='similarity')
# # retriever_instructor_xl = db_instructor_xl.as_retriever(search_type='similarity_score_threshold')

# # print(
# #     """Nesse ponto, é necessário que o Llama esteja funcionando na porta 11434 do localhost"""
# # )
# # print(
# #     """Preparando a estrutura de comunicação com o Llama, incluindo template das mensagens"""
# # )
# # from langchain_ollama import ChatOllama
# # model_llama = ChatOllama(
# #     model="llama3.1",  # 'model' specifies the model to use, in this case 'llama3.1'
# #     temperature=0,  # 'temperature' controls the randomness of the model's output, with 0 being deterministic
# #     base_url="http://localhost:11434",  # 'base_url' specifies the base URL for the model's API endpoint
# # )

# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.runnables import RunnablePassthrough

# # # Define the chat prompt template with a series of messages
# # # 'from_messages' method creates a ChatPromptTemplate from a list of message tuples
# # # Each tuple contains a message type and the message content
# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         # System message to establish the assistant's role
# #         # This message sets the context for the assistant, instructing it to answer questions about UFRN
# #         # If the assistant doesn't know the answer, it should say so
# #         (
# #             "system",
# #             "Você é um assistente de servidores que responde a dúvidas sobre a Assembleia legislativa do Roi Grande do Norte. Use as informações fornecidas para responder às perguntas. Assuma um tom sarcástico, com leve ironia nas respostas. Se você não souber a resposta, seja passivo agressivo e diga que não tem informações suficientes para responder.",
# #         ),
# #         # Placeholder for chat history to maintain context
# #         # This placeholder will be replaced with the actual chat history during execution
# #         ("placeholder", "{chat_history}"),
# #         # Human message placeholder for user input
# #         # This placeholder will be replaced with the user's question and context during execution
# #         ("human", "\nCONTEXTO: {context} \n\nPERGUNTA: {question}"),
# #     ]
# # )

# # # Define a function to format retrieved documents
# # # 'docs' is a list of document objects
# # # The function joins the page content of each document with four newline characters in between
# # def format_retrieved_documents(docs):
# #     return "\n\n\n\n".join([doc.page_content for doc in docs])

# # # Define a base runnable for the Sentence Transformers retriever
# # # This runnable will handle the context and question for the chat prompt
# # base_runnable_instructor_xl = (
# #     {
# #         # 'context' key will use the retriever to get relevant documents and format them
# #         # 'retriever_sentence_transformers' retrieves relevant documents based on the query
# #         # 'format_retrieved_documents' formats the retrieved documents for the chat prompt
# #         "context": retriever_instructor_xl | format_retrieved_documents,
# #         # 'question' key will pass the question directly to the retriever without modification
# #         "question": RunnablePassthrough(),
# #     }
# #     # Combine the context and question with the chat prompt template
# #     | prompt
# # )


# # # Initialize an output parser to parse the string output of the chat
# # # 'StrOutputParser' is used to parse the output of the chat into a string format
# # output_parser = StrOutputParser()

# # rag_chain_sentence_transformers_llama = (
# #     base_runnable_instructor_xl  # Use the base runnable with Sentence Transformers embeddings
# #     | model_llama  # Pass the result to the Llama 3.1 model for response generation
# #     | output_parser  # Parse the model's output into a string format
# # )


# # from time import time
# # pergunta = "Explique a funcionalidade que você está servindo"
# # while pergunta != '':
# #     # resposta = rag_chain_sentence_transformers_llama.invoke(pergunta)
# #     # print(resposta + '\n\n')
# #     start = time()
# #     resposta = retriever_instructor_xl.invoke(pergunta)
# #     print(resposta)
# #     end = time()
# #     print(f'Executou em {end - start} seundos')
# #     pergunta = input('>>> ')