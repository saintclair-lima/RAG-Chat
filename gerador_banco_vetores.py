print(
"""Inicializando a aplicação...
Definindo os caminhos e lendo o arquivo..."""
)

DOCUMENTOS = {
    'regimento_alrn': {
        'url': 'datasets/regimento_al.txt',
        'titulo': 'Regimento Interno - ALRN',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'https://www.google.com/url?q=https%3A%2F%2Fwww.al.rn.leg.br%2Fregimento-interno%2FRegimento_Interno_ALRN_junho_2024_DL.pdf'
    },
    'res_auxilio_saude_alrn' : {
        'url': 'datasets/res_auxilio_saude_alrn.txt',
        'titulo': 'Resolução Nº 78, de 10 de julho de 2024 - auxílio de assistência à saúde no âmbito da Assembleia Legislativa do Rio Grande do Norte',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'https://www.al.rn.leg.br/legislacao/leis-complementares'
    },
    'res_base_calculo_ferias_decimo_alrn' : {
        'url': 'datasets/res_base_calculo_ferias_decimo_alrn.txt',
        'titulo': 'Resolução Nº 77, de 10 de julho de 2024 - base de cálculo do terço de férias e da gratificação natalina (13º salário)',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'https://www.al.rn.leg.br/legislacao/leis-complementares'
    },
    'res_avaliacao_desempenho_estagio_probatorio_alrn' : {
        'url': 'datasets/res_avaliacao_desempenho_estagio_probatorio_alrn.txt',
        'titulo': 'Resolução Nº 106/2018 - Avaliação do servidor em estágio probatório de trabalho',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'https://www.al.rn.leg.br/legislacao/leis-complementares'
    },
    'res_plano_de_cargos_carreiras_vencimento_alrn' : {
        'url': 'datasets/res_plano_de_cargos_carreiras_vencimento_alrn.txt',
        'titulo': 'Resolução Nº 089/2017 - Plano de Cargos, Carreiras e Vencimentos dos servidores efetivos',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'https://www.al.rn.leg.br/legislacao/leis-complementares'
    },
    'res_alt_plano_de_cargos_carreiras_vencimento_alrn' : {
        'url': 'datasets/res_alt_plano_de_cargos_carreiras_vencimento_alrn.txt',
        'titulo': 'Resolução Nº 75, de 27 de junho de 2024 - Alteração no Plano de Cargos, Carreiras e Vencimentos dos servidores efetivos',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'http://diario.al.rn.leg.br/diarios/2B2F147C885C4945A9F6FCE08EF1832A/arquivo'
    },
    'res_substituicao_cargo_funcao_alrn' : {
        'url': 'datasets/res_substituicao_cargo_funcao_alrn.txt',
        'titulo': 'Resolução Nº 64, de 19 de dezembro de 2022 - Substituição de cargos e funções',
        'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
        'fonte': 'http://diario.al.rn.leg.br/diarios/C72EA4B458B34333BCCC2F72B6A0EB5E/arquivo'
    },
    'regime_juridico_servidores_rn' : {
        'url': 'datasets/regime_juridico_servidores_rn.txt',
        'titulo': 'Regime Jurídico dos Servidores do Rio Grande do Norte, LEI COMPLEMENTAR ESTADUAL Nº 122, DE 30 DE JUNHO DE 1994',
        'autor': 'Governo do Rio Grande do Norte',
        'fonte': 'https://www.al.rn.leg.br/storage/legislacao/2019/07/17/da631d970bd52174d7fa82be9d3e23e9.pdf'
    }
    #'constituicao_rn': {
    #    'url': 'datasets/constituicao_rn.txt',
    #    'titulo': 'Constituição Estadual do Rio Grande do Norte',
    #    'autor': 'Governo do Rio Grande do Norte',
    #    'fonte': 'https://www.al.rn.leg.br/documentos/Constituicao_Estadual_versao_final_2023.pdf'
    #},
    #'constituicao_federal': {
    #    'url': 'datasets/constituicao_federal.txt',
    #    'titulo': 'Constituição da República Federativa do Brasil de 1988',
    #    'autor': 'Governo Federal',
    #    'fonte': 'https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm'
    #},
}

URL_CHROMADB = "/content/drive/MyDrive/ALRN-Docs/Chatbot/banco_vetores_alrn_adicional"
#URL_CHROMADB = "/content/drive/MyDrive/ALRN-Docs/Chatbot/banco_vetores_alrn_const"
DEVICE = "cpu"
# DEVICE = 'cuda'

#pip install langchain-core langchain_chroma chromadb langchain-huggingface

for k, v in DOCUMENTOS.items(): print(v['url'])

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

documentos = []
titulos = []
for k, v in DOCUMENTOS.items():
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


print("""Definindo o hkunlp/instructor-xl""")

embedding_function_instructor_xl = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl",
    show_progress=True,
    model_kwargs={"device": DEVICE},
)

print("""Inicializando o vector store""")

# Inicializa a instância ChromaDB usando o método 'from_documents'
# 'documents' é uma lista de fragmentos de documentos que serão armazenados no banco de dados
# 'embedding' é a função de embedding usada para gerar embeddings dos documentos
# 'persist_directory' especifica o diretório onde a instância ChromaDB será salva
db_instructor_xl = Chroma.from_documents(
    documents=documentos,
    embedding=embedding_function_instructor_xl,
    persist_directory=URL_CHROMADB,
)