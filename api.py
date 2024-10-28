print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
from starlette.middleware.cors import CORSMiddleware

import environment
from gerador_de_respostas import GeradorDeRespostas, DadosChat
from utils import FuncaoEmbeddings

print('Instanciando a api (FastAPI)')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allow all methods
    allow_headers=['*'],  # Allow all headers
)

print(f'Criando a função de embeddings com {environment.MODELO_DE_EMBEDDINGS}')
funcao_de_embeddings = FuncaoEmbeddings(model_name=environment.MODELO_DE_EMBEDDINGS, biblioteca=SentenceTransformer)

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_banco_vetores=environment.URL_BANCO_VETORES)

print('Definindo as rotas')

@app.post('/chat/enviar_pergunta/')
async def gerar_resposta(dadosRecebidos: DadosChat):
    return StreamingResponse(gerador_de_respostas.consultar(dadosRecebidos), media_type='text/plain')


@app.get('/chat/')
async def pagina_chat():
    with open('chat.html', 'r', encoding='utf-8') as arquivo: conteudo_html = arquivo.read()
    # substituindo as tags dentro do HTML, para maior controle
    for tag, valor in environment.TAGS_SUBSTITUICAO_HTML.items():
        conteudo_html = conteudo_html.replace(tag, valor)
    return HTMLResponse(content=conteudo_html, status_code=200)

@app.get('/assets/img/favicon/favicon.ico')
async def favicon(): return FileResponse('assets/img/favicon/favicon.ico')

@app.get('/assets/img/favicon/favicon.svg')
async def favicon(): return FileResponse('assets/img/favicon/favicon.svg')

@app.get('/assets/img/favicon/favicon-48x48.png')
async def favicon(): return FileResponse('assets/img/favicon/favicon-48x48.png')

@app.get('/assets/img/favicon/apple-touch-icon.png')
async def favicon(): return FileResponse('assets/img/favicon/apple-touch-icon.png')

@app.get('/assets/img/favicon/site.webmanifest')
async def favicon(): return FileResponse('assets/img/favicon/site.webmanifest')

@app.get('/assets/img/Legisberto.png')
async def legisberto(): return FileResponse('assets/img/Legisberto.png')

print('API inicializada')