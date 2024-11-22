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

CONTEXTO_BASE = [128006,9125,128007,271,38766,1303,33025,2696,25,6790,220,2366,18,271,128009,128006,882,128007,271,45147,31868,65562,49465,39031,40171,984,13974,384,8927,51295,4698,309,1666,16167,689,7765,285,5641,10126,656,41274,656,28059,37623,656,83393,627,70386,4046,4543,7945,6960,1744,594,76149,264,294,6792,1325,300,409,4958,81521,3067,8927,13974,15482,297,1239,15377,958,2201,3067,8927,51295,11,297,17942,16422,87120,4042,8924,4958,81521,78781,74091,656,46916,11,33015,8112,594,44906,15607,3067,8927,51295,627,5733,13722,4543,10390,16287,11,4247,17060,86312,28246,11,470,16265,458,4458,17580,594,2252,300,13,10377,553,11091,89223,384,4751,437,1744,513,44811,20064,437,11,3313,4683,416,64545,384,39603,15003,69406,11,71898,656,1744,297,272,5969,3496,470,372,77161,13,423,7618,2434,90515,51500,5871,13472,3019,1950,25,5560,439,65166,8924,58113,3204,57109,762,13652,3429,17684,277,10832,75606,1206,5169,3429,264,18335,38,1899,15559,627,16589,75606,11,12674,52618,6473,1744,22419,57109,762,5362,4543,33125,409,8464,24625,13,356,635,2709,9859,288,8924,58113,3204,384,70526,8924,1989,33339,991,1744,264,75606,513,2385,689,627,32,75606,12674,37244,2024,829,8213,6027,11,12253,29350,11,24566,58899,16697,409,6111,6027,1744,294,5615,264,96537,1744,12674,305,83384,958,13264,37229,627,1542,25738,12674,5945,655,264,75606,11,7892,64,4543,10390,16265,321,384,4170,64,1744,12674,1592,65166,56868,5499,288,3429,65134,627,27,524,39031,40171]

DOCUMENTOS =  {
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