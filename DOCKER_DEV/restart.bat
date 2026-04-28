@echo off
chcp 65001 >nul

echo =====================================================
echo     Reiniciando Ambiente Docker Dev Full-Stack
echo =====================================================
echo.

echo Parando serviços atuais...
call stop.bat

echo.
echo Aguardando 3 segundos antes de reiniciar...
timeout /t 3 >nul

echo.
echo Iniciando serviços novamente...
call start.bat

echo.
echo Reinício concluído.