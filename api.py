print('Inicializando a estrutura da API...\nImportando as bibliotecas...')
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from gerador_de_respostas import GeradorDeRespostas, DadosChat

URL_BANCO_VETORES = 'banco_vetores_alrn'

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

gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_banco_vetores=URL_BANCO_VETORES)

print('Definindo as rotas')


# @app.post("/chat/enviar_pergunta/")
# async def gerarResposta(dadosRecebidos: DadosChat):
#     dados_resposta = await gerador_de_respostas.consultar(dadosRecebidos)
#     return {"dados_resposta": dados_resposta}

@app.post("/chat/enviar_pergunta/")
async def gerar_resposta(dadosRecebidos: DadosChat):
    # dados_resposta = await gerador_de_respostas.gerar_resposta(dadosRecebidos)
    # return {"dados_resposta": dados_resposta}
    return StreamingResponse(gerador_de_respostas.consultar(dadosRecebidos), media_type="text/plain")


@app.get("/chat/")
async def pagina_chat():
    with open('chat.html', 'r', encoding='utf-8') as arquivo: conteudo_html = arquivo.read()
    return HTMLResponse(content=conteudo_html, status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("assets/img/favicon/favicon.ico")

print('API inicializada')