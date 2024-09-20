print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from gerador_de_respostas import GeradorDeRespostas, DadosChat

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
    model_kwargs={"device": "cpu"}
)

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_vector_store='chroma_db_instructor_xl')

print('Definindo as rotas')


@app.post("/hilda/enviar_pergunta/")
async def get_embedding(dadosRecebidos: DadosChat):
    dados_resposta = gerador_de_respostas.consultar(dadosRecebidos)
    return {"dados_resposta": dados_resposta}


@app.get("/hilda/")
async def pagina_chat():
    with open('chat.html', 'r', encoding='utf-8') as arquivo: conteudo_html = arquivo.read()
    return HTMLResponse(content=conteudo_html, status_code=200)

print('API inicializada')