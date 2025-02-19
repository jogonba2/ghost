#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "ghost" --exclude=__init__.py
isort "ghost"
black "ghost" -l 80
