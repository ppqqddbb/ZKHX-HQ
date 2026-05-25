@echo off
cd /d D:\InstallFiles\File\Project

start "" "D:\InstallFiles\Anaconda\envs\py310\python.exe" -m api.api_server
start "" "D:\InstallFiles\Anaconda\envs\py310\python.exe" -m api.api_database

pause
