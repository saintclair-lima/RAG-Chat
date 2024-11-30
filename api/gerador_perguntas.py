import requests
import json
import api.environment.environment as environment
url = 'http://localhost:11434/api/generate'
def gerar_perguntas(artigo, contexto):
    prompt = '''Considere o artigo abaixo. Crie pelo menos 5 perguntas que possam ser respondidas com fragmentos do artigo. A saída deve ser uma lista de objetos JSON, com os atributos {{"pergunta": "Texto da pergunta Gerada", "resposta": "fragmento do artigo que responde a pergunta"}}. Não adicione nada na resposta, exceto a lista de objetos JSON, sem qualquer comentário adicional. ARTIGO: {}'''.format(artigo)
    payload = {
        "model": 'llama3.2',
        "prompt": prompt,
        "temperature": 0.0,
        "context": contexto
    }
    resposta = requests.post(url, json=payload, stream=True)
    resposta.raise_for_status()
    texto_resposta = ''
    for fragmento in resposta.iter_content(chunk_size=None):
        if fragmento:
            dados = json.loads(fragmento.decode())
            texto_resposta += dados['response']
    return texto_resposta



documentos = []
titulos = []
id=1

for k, v in environment.DOCUMENTOS.items():
    URL_DADOS = './' + v['url']
    print(f'''Lendo o arquivo {URL_DADOS}...''')
    with open(URL_DADOS, 'r', encoding='UTF-8') as arq:
        texto = arq.read()
    # Listagens até 9, em português, são marcadas por números ordinais. Isso faz com que
    # posteriormente os números ordinais recebam mais atenção do que deveriam, na
    # representação TF-IDF. O código abaixo remove a marcação de ordinais e coloca a
    # mesma notação utilizada nos demais itens
    for num in range(1, 10):
        texto = texto.replace(f'Art. {num}º', f'Art. {num}.')
        texto = texto.replace(f'art. {num}º', f'art. {num}.')
        texto = texto.replace(f'§ {num}º', f'§ {num}.')
    texto = texto.split('\n')
    if '' in texto: texto.remove('')
    print(f'''Dividindo em artigos...''')
    artigos = []
    for art in texto:
        item = art.split(' ')
        qtd_palavras = len(item)
        if qtd_palavras > 500:
            item = (
                art.replace('. §', '.\n§')
                .replace('; §', ';\n§')
                .replace(': §', ':\n§')
                .replace(';', '\n')
                .replace(':', '\n')
                .replace('\n ', '\n')
                .replace(' \n', '\n')
                .split('\n')
            )
            caput = item[0]
            fragmento_artigo = '' + caput
            for i in range(1, len(item)):
                if len(fragmento_artigo.split(' ')) + len(item[i]) <= 500:
                    fragmento_artigo = fragmento_artigo + ' ' + item[i]
                else:
                    artigos.append(fragmento_artigo)
                    fragmento_artigo = '' + caput + ' ' + item[i]
            artigos.append(fragmento_artigo)
        else:
            artigos.append(art)
    print('''Criando os documentos com base nos artigos...''')
    for artigo in artigos:
        tit = artigo.split('. ')[1]
        titulos.append(tit)
        doc = {
            'id': id,
            'page_content': artigo,
            'metadata': {
                'titulo': f'{v["titulo"]}',
                'subtitulo': f'Art. {tit} - {titulos.count(tit)}',
                'autor': f'{v["autor"]}',
                'fonte': f'{v["fonte"]}',
            },
        }
        documentos.append(doc)
        id += 1

qtd_docs = len(documentos)
for idx in range(qtd_docs):
    print(f'Processando documento {idx+1} de {qtd_docs}')
    doc = documentos[idx]    
    perguntas = gerar_perguntas(artigo=doc['page_content'], contexto=[])
    doc['perguntas'] = perguntas

print('Salvando arquivo com perguntas...')
import pickle
with open('documentos.pickle', 'wb') as f:
    pickle.dump(documentos, f, pickle.HIGHEST_PROTOCOL)