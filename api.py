print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from gerador_de_respostas import GeradorDeRespostas, DadosChat

import configuracoes

print('Instanciando a api (FastAPI)')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allow all methods
    allow_headers=['*'],  # Allow all headers
)

print(f'Criando a função de embeddings com {configuracoes.MODELO_DE_EMBEDDINGS}')
funcao_de_embeddings = HuggingFaceEmbeddings(
    model_name=configuracoes.MODELO_DE_EMBEDDINGS,
    show_progress=True,
    model_kwargs={'device': configuracoes.DEVICE},
    cache_folder='modelos_ia'
)

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_banco_vetores=configuracoes.URL_BANCO_VETORES)

print('Definindo as rotas')

@app.post('/chat/enviar_pergunta/')
async def gerar_resposta(dadosRecebidos: DadosChat):
    # dados_resposta = await gerador_de_respostas.gerar_resposta(dadosRecebidos)
    # return {'dados_resposta': dados_resposta}
    return StreamingResponse(gerador_de_respostas.consultar(dadosRecebidos), media_type='text/plain')


@app.get('/chat/')
async def pagina_chat():
    with open('chat.html', 'r', encoding='utf-8') as arquivo: conteudo_html = arquivo.read()
    # substituindo as tags dentro do HTML, para maior controle
    for tag, valor in configuracoes.TAGS_SUBSTITUICAO_HTML.items():
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