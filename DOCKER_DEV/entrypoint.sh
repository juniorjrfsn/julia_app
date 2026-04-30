#!/bin/bash
set -e

echo "================================================"
echo "  Iniciando servicos do container..."
echo "================================================"

mkdir -p /var/run/sshd

if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "[SSH] Gerando chaves do host..."
    ssh-keygen -A
fi

echo "[Django] Iniciando o servidor web..."
cd /srv/chatbot_ageprev_py
if [ -f manage.py ]; then
    python manage.py migrate
    python manage.py runserver 0.0.0.0:8000 &
else
    echo "[AVISO] manage.py não encontrado, serviço Django não iniciado."
fi

echo "[SSH] Iniciando sshd..."
echo "================================================"
echo "  Container pronto!"
echo "  ssh root@localhost -p 2222"
echo "  Web: http://localhost:8000"
echo "  Senha: senha_forte_aqui"
echo "================================================"

exec /usr/sbin/sshd -D