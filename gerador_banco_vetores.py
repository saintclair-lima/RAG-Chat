import configuracoes

print(
"""Inicializando a aplicação...
Definindo os caminhos e lendo arquivos..."""
)

#pip install langchain-core langchain_chroma chromadb langchain-huggingface

for k, v in configuracoes.DOCUMENTOS.items(): print(v['url'])

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

documentos = []
titulos = []
for k, v in configuracoes.DOCUMENTOS.items():
    URL_DADOS = '/content/drive/MyDrive/ALRN-Docs/Chatbot/' + v['url']
    print(f"""Lendo o arquivo {URL_DADOS}...""")

    with open(URL_DADOS, "r", encoding="UTF-8") as arq:
        texto = arq.read()

    # Listagens até 9, em português, são marcadas por números ordinais. Isso faz com que
    # posteriormente os números ordinais recebam mais atenção do que deveriam, na
    # representação TF-IDF. O código abaixo remove a marcação de ordinais e coloca a
    # mesma notação utilizada nos demais itens

    for num in range(1, 10):
        texto = texto.replace(f"Art. {num}º", f"Art. {num}.")
        texto = texto.replace(f"art. {num}º", f"art. {num}.")
        texto = texto.replace(f"§ {num}º", f"§ {num}.")
    texto = texto.split("\n")
    if '' in texto: texto.remove('')

    print(f"""Dividindo em artigos...""")
    artigos = []
    for art in texto:
        item = art.split(" ")
        qtd_palavras = len(item)
        if qtd_palavras > 500:
            item = (
                art.replace(". §", ".\n§")
                .replace("; §", ";\n§")
                .replace(": §", ":\n§")
                .replace(";", "\n")
                .replace(":", "\n")
                .replace("\n ", "\n")
                .replace(" \n", "\n")
                .split("\n")
            )
            caput = item[0]
            fragmento_artigo = "" + caput
            for i in range(1, len(item)):
                if len(fragmento_artigo.split(" ")) + len(item[i]) <= 500:
                    fragmento_artigo = fragmento_artigo + " " + item[i]
                else:
                    artigos.append(fragmento_artigo)
                    fragmento_artigo = "" + caput + " " + item[i]
            artigos.append(fragmento_artigo)
        else:
            artigos.append(art)

    print("""Criando os documentos com base nos artigos...""")

    for artigo in artigos:
        tit = artigo.split(". ")[1]
        titulos.append(tit)
        doc = Document(
            page_content=artigo,
            metadata={
                "titulo": f"{v['titulo']}",
                "subtitulo": f"Art. {tit} - {titulos.count(tit)}",
                "autor": f"{v['autor']}",
                "fonte": f"{v['fonte']}",
            },
        )
        documentos.append(doc)


print(f'Criando a função de embeddings com {configuracoes.MODELO_DE_EMBEDDINGS}')
funcao_de_embeddings = HuggingFaceEmbeddings(
    model_name= configuracoes.MODELO_DE_EMBEDDINGS,
    show_progress=True,
    model_kwargs={"device": configuracoes.DEVICE},
)

print("""Inicializando o banco de vetores com persistência""")

# Inicializa a instância ChromaDB usando o método 'from_documents'
# 'documents' é uma lista de fragmentos de documentos que serão armazenados no banco de dados
# 'embedding' é a função de embedding usada para gerar embeddings dos documentos
# 'persist_directory' especifica o diretório onde a instância ChromaDB será salva
db_instructor_xl = Chroma.from_documents(
    documents=documentos,
    embedding=funcao_de_embeddings,
    persist_directory=configuracoes.URL_BANCO_VETORES,
)