# ==================== BASE (Python) ====================
FROM python:3.11-slim-bookworm AS base

# ==================== METADADOS ====================
LABEL maintainer="Nedir França<nfranca@ageprev.ms.gov.br>"
LABEL description="Ambiente completo de desenvolvimento com múltiplos runtimes"

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo \
    NVM_DIR=/root/.nvm \
    PYTHONUNBUFFERED=1

# ==================== ATUALIZAÇÃO E PACOTES BÁSICOS ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    wget \
    unzip \
    vim \
    ssh \
    sudo \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==================== INSTALAÇÃO DO NODE.JS via NVM ====================
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install --lts \
    && nvm use --lts \
    && nvm alias default lts/* \
    && npm install -g typescript ts-node \
    && echo 'export NVM_DIR="$HOME/.nvm"' >> /root/.bashrc \
    && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /root/.bashrc \
    && echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> /root/.bashrc

# ==================== PYTHON + DJANGO ====================
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir django djangorestframework django-cors-headers

# ==================== RUST + CARGO ====================
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && echo 'source "$HOME/.cargo/env"' >> /root/.bashrc

# ==================== JAVA (Temurin 17 + 21) ====================
RUN mkdir -p /etc/apt/keyrings \
    && wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | \
    gpg --no-tty --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME=/{print$2}' /etc/os-release) main" > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    temurin-17-jdk temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# ==================== C# / .NET SDK ====================
RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update && apt-get install -y dotnet-sdk-8.0 \
    && rm -rf /var/lib/apt/lists/*

# ==================== ERLANG + ELIXIR ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    erlang elixir \
    && rm -rf /var/lib/apt/lists/*

# ==================== BANCO DE DADOS ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-contrib postgresql-client sqlite3 \
    && rm -rf /var/lib/apt/lists/*
# docker pull postgres:latest

# ==================== SERVIDOR FTP (vsftpd) ====================
RUN apt-get update && apt-get install -y vsftpd \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/vsftpd/empty

# ==================== SSH SERVER ====================
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config \
    && echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config \
    && echo 'root:@Dirgin@202604' | chpasswd

# ==================== USUÁRIO DE DESENVOLVIMENTO ====================
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo 'dev:@Dirgin@202604' | chpasswd



# ==================== JULIA (adicionada na mesma stage) ====================
# Continuamos da stage base (que tem Python, Node, Rust, Java, etc.)
FROM base AS final 
# Instalar dependências do sistema necessárias para Julia (gráficos, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-dev \
    imagemagick \
    ffmpeg \
    gettext \
    && rm -rf /var/lib/apt/lists/*

# ==================== INSTALAÇÃO DO JULIA ====================
# Baixa e instala a versão oficial do Julia (recomendado)
RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz -O julia.tar.gz \
    && tar -xzf julia.tar.gz -C /usr/local --strip-components=1 \
    && rm julia.tar.gz \
    && ln -s /usr/local/bin/julia /usr/bin/julia 
# Instalar pacotes Julia
# Instalar Julia e alguns pacotes
RUN julia -e 'using Pkg; Pkg.add(["FileIO"]); Pkg.precompile()'




# ==================== DIRETÓRIO DE TRABALHO ====================
WORKDIR /app
COPY . /app

# ==================== ENTRYPOINT ====================
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]