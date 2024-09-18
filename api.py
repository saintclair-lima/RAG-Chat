print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI, Query
from langchain_huggingface import HuggingFaceEmbeddings
from gerador_de_respostas import GeradorDeRespostas

print('Instanciando a api (FastAPI)')
app = FastAPI()

print('Criando a função de embeddings')
funcao_de_embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    show_progress=True,
    model_kwargs={"device": "cpu"}  # Or "cpu" if you don't have a GPU
)

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_vector_store='chroma_db_instructor_xl')

print('Definindo as rotas')
@app.post("/get_embedding/")
async def get_embedding(pergunta: str):
    resposta = gerador_de_respostas.consultar(pergunta)
    return {"resposta": resposta}

# Define a GET route that takes a query parameter 'text'
@app.get("/get_embedding/")
async def get_embedding(text: str = Query(..., description="Pergunta/Query a ser buscada no banco de vetores")):
    resposta = gerador_de_respostas.consultar(text)
    return {"resposta": resposta}

print('API inicializada')