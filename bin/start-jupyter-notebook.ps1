#!/usr/bin/env -S powershell.exe -ExecutionPolicy Bypass

$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition

cd $scriptPath
cd ..

jupyter-notebook --no-browser