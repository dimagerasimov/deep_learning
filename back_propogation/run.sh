#!/bin/bash

git clone https://github.com/sorki/python-mnist
sh ./get_data.sh
cd ./code
make -B
make run

exit 0
