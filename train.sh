#!/bin/bash

if [ "$#" -ne 4 ]; then
	echo "Usage: ./train.sh <model_name> <exp_number> <exp_name> <date(mm-dd-yyy)>"
fi

cd src/models/ || exit
cp train_model.py temp.py
echo "from $1 import $1" >> temp.py
echo "trainer($1(), \"$2\", $3, \"$4\").train()" >> temp.py

cd ../../
python src/models/temp.py &> "$1-EXP$2"
rm src/models/temp.py
