#!/bin/bash

if [ "$#" -ne 4 ]; then
	echo "Usage: ./train.sh <model_name> <exp_number> <exp_name> <date(mm-dd-yyy)>"
	exit
fi

cd src/models/
cp train_model.py temp.py
echo "from $1 import $1" >> temp.py
echo "trainer($1(), \"$3\", $2, \"$4\").train()" >> temp.py

cd ../../
python src/models/temp.py &> "$1-EXP$2.out"
rm src/models/temp.py
