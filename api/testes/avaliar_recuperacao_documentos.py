## print('Para simplicidade, mover o arquivo para a pasta principal para executar')
print('Importando bibliotecas...')
import json
from sentence_transformers import SentenceTransformer
import environment
from gerador_de_respostas import GeradorDeRespostas
from utils import FuncaoEmbeddings
from testes.docs_perguntas import documentos as docs
from time import time
import asyncio

FAZER_LOG = False

async def avaliar_recuperacao_documentos():
    print(f'Criando GeradorDeRespostas (usando {environment.MODELO_DE_EMBEDDINGS})...')
    funcao_de_embeddings = FuncaoEmbeddings(nome_modelo=environment.MODELO_DE_EMBEDDINGS, tipo_modelo=SentenceTransformer, device=environment.DEVICE)
    gerador_de_respostas = GeradorDeRespostas(funcao_de_embeddings=funcao_de_embeddings, url_banco_vetores=environment.URL_BANCO_VETORES, device=environment.DEVICE)

    print(f'Gerando lista de perguntas sintéticas')
    perguntas = []
    for item in docs:
        for pergunta in item['perguntas']:
            if pergunta['resposta'] != '': perguntas.append({'id': item['id'], 'titulo': item['metadata']['titulo'], 'subtitulo': item['metadata']['subtitulo'], 'pergunta': pergunta['pergunta'], 'resposta': pergunta['resposta']})

    qtd_perguntas = len(perguntas)
    for idx in range(qtd_perguntas):
        pergunta = perguntas[idx]
        print(f'\rPergunta {idx+1} de {qtd_perguntas}', end='')
        if FAZER_LOG: print(f'-- realizando consulta para: "{pergunta['pergunta']}"...')

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


    with open('testes.json', 'w', encoding='utf-8') as arq:
        arq.write(json.dumps(perguntas, indent=4, ensure_ascii=False))



# Run the `avaliar` function
if __name__ == "__main__":
    asyncio.run(avaliar_recuperacao_documentos())