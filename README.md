# RAG-Chat

## Ollama: Instalação e Configuração
### Instalando o Ollama
#### Linux
* No terminal, baixe e execute o script de instalação:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
#### Windows
* Acesse a página de download do Ollama no link https://ollama.com/download/windows e faça o download do instalador.
* Execute o instalador e siga as instruções em tela

### Adicionando os modelos ao Ollama
No projeto, utilizamos como modelo principal o `Llama3.2`, mas outros modelos são disponíveis para download (como o `phi3.5`, por exemplo).
Para inclusão do modelo, tanto no Windos como no Linux, por meio do terminal/prompt/Powershell, executa-se:

```
ollama pull <nome do modelo>
```
Em nosso caso:
```
ollama pull llama3.2
```
Após a inclusão do modelo no Ollama, pode-se testar se tudo ocorreu corretamente executando o modelo direto no terminal/promt/Powershell:
```
ollama run <nome do modelo>
```
Em nosso caso:
```
ollama run llama3.2
```

### Configurações de Ambiente
De forma a aceitar requisições concorrentes, o Ollama precisa ter alguns parâmetros configurados adequadamente. Para isso, é necessário estabelecer algumas variáveis de ambiente a serem utilizadas por ele, a saber:
* OLLAMA_NUM_PARALLEL
* OLLAMA_MAX_LOADED_MODELS
* OLLAMA_MAX_QUEUE

De acordo com a documentação (ver https://github.com/ollama/ollama/blob/main/docs/faq.md), cada uma das variáveis faz o seguinte:
* `OLLAMA_MAX_LOADED_MODELS` - O número máximo de modelos que podem ser carregados simultaneamente, desde que caibam na memória disponível. O padrão é 3 * o número de GPUs ou 3 para inferência via CPU.
* `OLLAMA_NUM_PARALLEL` - O número máximo de solicitações paralelas que cada modelo processará ao mesmo tempo. O padrão irá selecionar automaticamente 4 ou 1 dependendo da memória disponível.
* `OLLAMA_MAX_QUEUE` - O número máximo de solicitações que Ollama irá enfileirar quando estiver ocupado, antes de rejeitar solicitações adicionais. O padrão é 512.

Outras variáveis de interesse podem ser `OLLAMA_DEBUG` (true para ativar debug) e  `AQUELA_OUTRA_QUE_NAO_CONSIGO_LEMBRAR`.

Assim, definimos as variáveis de ambiente semelhante ao que segue.

#### Linux
```
export OLLAMA_NUM_PARALLEL=5
export OLLAMA_MAX_LOADED_MODELS=5
export OLLAMA_MAX_QUEUE=512
export OLLAMA_DEBUG='true'
```
#### Windows (Powershell)
```
$env:OLLAMA_NUM_PARALLEL=5;$env:OLLAMA_MAX_LOADED_MODELS=5;$env:OLLAMA_MAX_QUEUE=512;$env:OLLAMA_DEBUG='true'
```
#### Windows (Prompt)
```
set OLLAMA_NUM_PARALLEL=5
set OLLAMA_MAX_LOADED_MODELS=5
set OLLAMA_MAX_QUEUE=512
set OLLAMA_DEBUG='true'
```

### Inicializando o Ollama como API
Basta executar
```
ollama serve
```
Caso `OLLAMA_DEBUG` esteja configurado como `true` é feito um log com as configurações de inicialização do Ollama.

## Python: Instalação e Configuração
### Instalando Python
#### Windows
Instale Python e o gerenciador de pacotes do Python `pip`.
* Acesse a página de download do instalador (https://www.python.org/downloads/)
* Execute o instalador, seguindo as orientações em tela.
* _OBS:_ O instalador Windows já oferece a opção de incluir o `pip` durante a instalação do Python.

#### Linux
Utilize o gerenciador de pacotes da sua distribuição para instalar.

```
sudo apt install python3
sudo apt install pip
```
### OPCIONAL: Utilizando um ambiente virtual Python
Após a instalação do Python, você pode optar por criar um ambiente Python específico para a instalação dos pacotes necessários a este projeto. Assim, você pode utilizar a mesma instalação do Python em outros projetos, com outras versões de bibliotecas diferentes deste.

Apesar de não ser obrigatório, é aconselhado, para fins de organização, apenas, a realização desse procedimento.

#### Criando o ambiente virtual
Dentro da pasta do projeto (ver mais abaixo), basta executar o seguinte para realizar a criação do ambiente:

```
python -m venv <nome-do-ambiente-à-sua-escolha>
```

No nosso caso, para fins de conveniência, utilizamos `chat-env` como nome do modelo (o controle de versão está configurado para ignorar esses arquivos, como se pode ver no arquivo `.gitignore` - ver mais abaixo).

#### Ativando o ambiente virtual
Após criar o ambiente virtual, é necessário ativá-lo. Para isso, acessa a pasta do projeto a partir do terminal/prompt/Powershell.

##### Linux
```
source ./<nome-do-ambiente>/bin/activate
```
No nosso caso:
```
source ./chat-env/bin/activate
```

##### Windows
```
./<nome-do-ambiente>/Scripts/activate
```
No nosso caso:
```
source ./chat-env/Scripts/activate
```

#### Desativando o ambiente virtual
```
deactivate
```

## Copiando e Configurando o Projeto
### Baixando os arquivos do projeto
Utilize o git para baixar o repositório

```git
git clone https://github.com/saintclair-lima/RAG-Chat.git

cd ./RAG-Chat
```

### Instalando as dependências
```
pip install -r requirements.txt
```
### Iniciando o projeto
```
uvicorn api:app --reload

```