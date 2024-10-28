import os
from dotenv import load_dotenv
load_dotenv()

URL_BANCO_VETORES = os.getenv('URL_BANCO_VETORES')
URL_LLAMA=os.getenv('URL_LLAMA')
URL_HOST=os.getenv('URL_HOST')
TAGS_SUBSTITUICAO_HTML={
    'TAG_INSERCAO_URL_HOST': URL_HOST,
    'TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM': os.getenv('TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM')
    }

THREADPOOL_MAX_WORKERS=int(os.getenv('THREADPOOL_MAX_WORKERS'))
NOME_COLECAO_DE_DOCUMENTOS=os.getenv('COLECAO_DE_DOCUMENTOS')
EMBEDDING_INSTRUCTOR=os.getenv('EMBEDDING_INSTRUCTOR')
EMBEDDING_SQUAD_PORTUGUESE=os.getenv('EMBEDDING_SQUAD_PORTUGUESE')
MODELO_LLAMA=os.getenv('MODELO_LLAMA')
DEVICE=os.getenv('DEVICE') # ['cpu', cuda']
NUM_DOCUMENTOS_RETORNADOS=int(os.getenv('NUM_DOCUMENTOS_RETORNADOS'))

MODELO_DE_EMBEDDINGS = EMBEDDING_INSTRUCTOR

DOCUMENTOS =  {
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
}