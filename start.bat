@echo off
cd /d "%~dp0"
py -m src.gui
if errorlevel 1 pause

