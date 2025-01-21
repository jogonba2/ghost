#!/usr/bin/env bash

set -e
set -x

mypy "ghost"
flake8 "ghost" --ignore=E501,W503,E203,E402
black "ghost" --check -l 80
