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

echo "[SSH] Iniciando sshd..."
echo "================================================"
echo "  Container pronto!"
echo "  ssh root@localhost -p 2222"
echo "  Senha: senha_forte_aqui"
echo "================================================"

exec /usr/sbin/sshd -D