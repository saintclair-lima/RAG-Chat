import os
from dotenv import load_dotenv

url_raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
url_dotenv = os.path.join(url_raiz_projeto, ".env")
load_dotenv(url_dotenv)

class Environment:
    def __init__(self):
        self.URL_BANCO_VETORES = os.getenv('URL_BANCO_VETORES')
        self.URL_LLAMA=os.getenv('URL_LLAMA')
        self.URL_HOST=os.getenv('URL_HOST')
        self.TAGS_SUBSTITUICAO_HTML={
            'TAG_INSERCAO_URL_HOST': self.URL_HOST,
            'TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM': os.getenv('TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM')
            }

        self.THREADPOOL_MAX_WORKERS=int(os.getenv('THREADPOOL_MAX_WORKERS'))
        self.NOME_COLECAO_DE_DOCUMENTOS=os.getenv('COLECAO_DE_DOCUMENTOS')
        self.EMBEDDING_INSTRUCTOR=os.getenv('EMBEDDING_INSTRUCTOR')
        self.EMBEDDING_SQUAD_PORTUGUESE=os.getenv('EMBEDDING_SQUAD_PORTUGUESE')
        self.MODELO_LLAMA=os.getenv('MODELO_LLAMA')
        self.DEVICE=os.getenv('DEVICE') # ['cpu', cuda']
        self.NUM_DOCUMENTOS_RETORNADOS=int(os.getenv('NUM_DOCUMENTOS_RETORNADOS'))

        self.MODELO_DE_EMBEDDINGS = self.EMBEDDING_INSTRUCTOR

        self.CONTEXTO_BASE = []

        self.DOCUMENTOS =  {
            'regimento_alrn': {
                'url': 'datasets/regimento_al.txt',
                'titulo': 'Regimento Interno - ALRN',
                'autor': 'Assembleia Legislativa do Rio Grande do Norte - ALERN',
                'fonte': 'https://www.al.rn.leg.br/regimento-interno/Regimento_Interno_ALRN_junho_2024_DL.pdf'
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
        }

environment = Environment()