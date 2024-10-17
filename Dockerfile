FROM python:3.12.6-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git \
    bash-completion \
    && apt-get remove --purge -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Modifica terminal padrão dos usuários para bash em vez do sh
RUN sed -i 's#/bin/sh#/bin/bash#g' /etc/passwd && \
    # Adiciona cores no prefixo do terminal
    echo 'force_color_prompt=yes' >> ~/.bashrc && \
    echo 'if [ "$force_color_prompt" = yes ]; then' >> ~/.bashrc && \
    echo 'PS1="\\[\\e[01;32m\\]\\u@\\h\\[\\e[00m\\]:\\[\\e[01;34m\\]\\w\\[\\e[00m\\]\\$ "' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc

WORKDIR /var/www

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get remove --purge -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PORT=8000
EXPOSE ${PORT}

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--reload", "--workers", "2"]