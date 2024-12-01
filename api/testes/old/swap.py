import json
import pickle

with open('testes_10_docs.json', 'r', encoding='utf-8') as arq:
    dados = json.load(arq)

ordenacoes = []
for item in dados:
    docs = item['documentos']
    score_distancia = sorted(docs, key=lambda x: x['score_distancia'], reverse=True)
    score_ponderado = sorted(docs, key=lambda x: x['score_ponderado'], reverse=True)
    score_bert_soma = sorted(docs, key=lambda x: x['score_bert'][0], reverse=True)
    score_bert_multipl = sorted(docs, key=lambda x: x['score_bert'][2], reverse=True)

    ordenacoes.append({
        'score_distancia':    [docs.index(item) for item in score_distancia],
        'score_ponderado':    [docs.index(item) for item in score_ponderado],
        'score_bert_soma':    [docs.index(item) for item in score_bert_soma],
        'score_bert_multipl': [docs.index(item) for item in score_bert_multipl],
    })

with open('ordenacoes_10.pickle', 'wb') as arq:
    pickle.dump(ordenacoes, arq, pickle.HIGHEST_PROTOCOL)