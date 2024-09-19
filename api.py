print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from gerador_de_respostas import GeradorDeRespostas
from pydantic import BaseModel

print('Instanciando a api (FastAPI)')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

print('Criando a função de embeddings')
funcao_de_embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    show_progress=True,
    model_kwargs={"device": "cpu"}  # Or "cpu" if you don't have a GPU
)

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_vector_store='chroma_db_instructor_xl')


# Define the Pydantic model for the POST request
class PerguntaModel(BaseModel):
    pergunta: str

print('Definindo as rotas')
@app.post("/get_embedding/")
async def get_embedding(dadosRecebidos: PerguntaModel):
    resposta = gerador_de_respostas.consultar(dadosRecebidos.pergunta)
    return {"resposta": resposta}

# # Define a GET route that takes a query parameter 'text'
# @app.get("/get_embedding/")
# async def get_embedding(text: str = Query(..., description="Pergunta/Query a ser buscada no banco de vetores")):
#     resposta = gerador_de_respostas.consultar(text)
#     return {"resposta": resposta}

print('API inicializada')