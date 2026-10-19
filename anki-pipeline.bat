@echo off
cd /d "%~dp0"
start "Anki-pipeline server" /b py -m src.gui
if errorlevel 1 pause

