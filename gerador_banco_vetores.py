print(
"""Inicializando a aplicação...
Definindo os caminhos e lendo o arquivo..."""
)

URL_DADOS = "datasets/regimento_alrn.txt"
URL_CHROMADB = "chroma_db_instructor_xl"
DEVICE = "cpu"
# DEVICE = 'cuda'

import torch_directml
dml = torch_directml.device()
DEVICE = dml
print(dml)

with open(URL_DADOS, "r", encoding="latin-1") as arq:
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
texto.remove("")
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

from langchain_core.documents import Document

documentos = []
titulos = []
for artigo in artigos:
    tit = artigo.split(". ")[1]
    titulos.append(tit)
    doc = Document(
        page_content=artigo,
        metadata={
            "title": f"Regimento Interno - Artigo {tit}_{titulos.count(tit)}",
            "author": "ALERN",
            "source": "https://www.google.com/url?q=https%3A%2F%2Fwww.al.rn.leg.br%2Fregimento-interno%2FRegimento_Interno_ALRN_junho_2024_DL.pdf",
        },
    )
    documentos.append(doc)


print("""Definindo o hkunlp/instructor-xl""")
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function_instructor_xl = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    show_progress=True,
    model_kwargs={"device": DEVICE},
)

print("""Inicializando o vector store""")
from langchain_chroma import Chroma

# Inicializa a instância ChromaDB usando o método 'from_documents'
# 'documents' é uma lista de fragmentos de documentos que serão armazenados no banco de dados
# 'embedding' é a função de embedding usada para gerar embeddings dos documentos
# 'persist_directory' especifica o diretório onde a instância ChromaDB será salva
db_instructor_xl = Chroma.from_documents(
    documents=documentos,
    embedding=embedding_function_instructor_xl,
    persist_directory=URL_CHROMADB,
)