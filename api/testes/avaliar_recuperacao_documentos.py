## print('Para simplicidade, mover o arquivo para a pasta principal para executar')
print('Importando bibliotecas...')
import json
from sentence_transformers import SentenceTransformer
from ..environment.environment import environment
from ..gerador_de_respostas import GeradorDeRespostas
from ..utils.utils import FuncaoEmbeddings
#from documentos_perguntas import documentos_perguntas as docs
from time import time
import asyncio
import os
from torch import cuda

URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
URL_BANCO_VETORES=os.path.join(URL_LOCAL,"../conteudo/bancos_vetores/banco_vetores_regimento_resolucoes_rh")
NOME_COLECAO='regimento_resolucoes_rh'
DEVICE='cuda' if cuda.is_available() else 'cpu'

FAZER_LOG = False

async def avaliar_recuperacao_documentos():
    print(f'Criando GeradorDeRespostas (usando {EMBEDDING_INSTRUCTOR})...')
    funcao_de_embeddings = FuncaoEmbeddings(nome_modelo=EMBEDDING_INSTRUCTOR, tipo_modelo=SentenceTransformer, device=DEVICE)
    gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_banco_vetores=URL_BANCO_VETORES, colecao_de_documentos=NOME_COLECAO, device=DEVICE)

    with open(os.path.join(URL_LOCAL,'documentos_perguntas.json'), 'r') as arq:
        docs = json.load(arq)
    
    print(f'Gerando lista de perguntas sintéticas')
    perguntas = []
    for item in docs:
        for pergunta in item['perguntas']:
            try:
                if pergunta['resposta'] != '': perguntas.append({'id': item['id'], 'titulo': item['metadata']['titulo'], 'subtitulo': item['metadata']['subtitulo'], 'pergunta': pergunta['pergunta'], 'resposta': pergunta['resposta']})
            except:
                print(pergunta)

    qtd_perguntas = len(perguntas)
    for idx in range(qtd_perguntas):
        pergunta = perguntas[idx]
        print(f'\rPergunta {idx+1} de {qtd_perguntas}', end='')
        if FAZER_LOG: print(f'''-- realizando consulta para: "{pergunta['pergunta']}"...''')

        # Recuperando documentos usando o ChromaDB
        marcador_tempo_inicio = time()
        documentos = await gerador_de_respostas.consultar_documentos_banco_vetores(pergunta['pergunta'], num_resultados=10)
        lista_documentos = gerador_de_respostas.formatar_lista_documentos(documentos)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if FAZER_LOG: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')

        # Atribuindo scores usando Bert
        if FAZER_LOG: print(f'--- aplicando scores do Bert aos documentos recuperados...')
        marcador_tempo_inicio = time()
        for documento in lista_documentos:
            resposta_estimada = await gerador_de_respostas.estimar_resposta(pergunta['pergunta'], documento['conteudo'])
            documento['score_bert'] = resposta_estimada['score']
            documento['score_ponderado'] = resposta_estimada['score_ponderado']
            documento['resposta_bert'] = resposta_estimada['resposta']
        marcador_tempo_fim = time()
        tempo_bert = marcador_tempo_fim - marcador_tempo_inicio
        if FAZER_LOG: print(f'--- scores atribuídos ({tempo_bert} segundos)\n\n\n')
        pergunta.update({
            'documentos': [
                {'id': doc['id'],
                'titulo': doc['metadados']['titulo'],
                'subtitulo': doc['metadados']['subtitulo'],
                'score_bert': doc['score_bert'],
                'score_distancia': doc['score_distancia'],
                'score_ponderado': doc['score_ponderado'],
                'resposta_bert': doc['resposta_bert']
                } for doc in lista_documentos],
            'tempo_consulta': tempo_consulta,
            'tempo_bert': tempo_bert
            })


        with open('testes_avaliar_documentos.json', 'w', encoding='utf-8') as arq:
            arq.write(json.dumps(perguntas, indent=4, ensure_ascii=False))



# Run the `avaliar` function
if __name__ == "__main__":
    asyncio.run(avaliar_recuperacao_documentos())