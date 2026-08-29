@echo off
rem Запуск ORB-бота для Task Scheduler (див. README.md).
cd /d %~dp0
call .venv\Scripts\activate
tradingbot run >> logs\stdout.log 2>&1
