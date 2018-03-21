#!/bin/bash
pip freeze > requirements.txt
sed -i "s/torch/#torch/g" requirements.txt
