import os
from ..environment.environment import environment
from ..utils.utils import FuncaoEmbeddings

from sentence_transformers import SentenceTransformer
from chromadb import chromadb

URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
COMPRIMENTO_MAX_FRAGMENTO = 300
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
URL_BANCO_VETORES=os.path.join(URL_LOCAL,"bancos_vetores/bancos_vetores_regimento_resolucoes_rh_300")
NOME_COLECAO='regimento_resolucoes_rh'
DEVICE='cuda'

class GeradorBancoVetores:
    def run(self):
        documentos = []
        titulos = []
        id=1

        for k, v in environment.DOCUMENTOS.items():
            URL_DADOS = os.path.join(URL_LOCAL, v['url'])
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
                if qtd_palavras > COMPRIMENTO_MAX_FRAGMENTO:
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
                        if len(fragmento_artigo.split(' ')) + len(item[i]) <= COMPRIMENTO_MAX_FRAGMENTO:
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

        # Utilizando o ChromaDb diretamente
        persist_directory = URL_BANCO_VETORES
        client = chromadb.PersistentClient(path=persist_directory)
        funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(nome_modelo=EMBEDDING_INSTRUCTOR, tipo_modelo=SentenceTransformer, device=DEVICE)
        collection = client.create_collection(name=NOME_COLECAO, embedding_function=funcao_de_embeddings_sentence_tranformer, metadata={'hnsw:space': 'cosine'})

        qtd_docs = len(documentos)
        for idx in range(qtd_docs):
            print(f'\rIncluindo documento {idx+1} de {qtd_docs}', end='')
            doc = documentos[idx]
            collection.add(
                documents=[doc['page_content']],
                ids=[str(doc['id'])],
                metadatas=[doc['metadata']],
            )
        client._system.stop()

        #query_result = collection.query(query_texts=["O que é uma legislatura?"], n_results=5)
        #print(query_result)

        # client.get_collection(name='legisberto', embedding_function=funcao_de_embeddings_sentence_tranformer)

if __name__ == "__main__":
    gerador_banco_vetores = GeradorBancoVetores()
    gerador_banco_vetores.run()