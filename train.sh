#!/bin/bash

if [ "$#" -ne 4 ]; then
	echo "Usage: ./train.sh <model_name> <exp_number> <exp_name> <date(mm-dd-yyy)>"
	exit
fi

cd src/models/ || exit
temp_file="temp-training-$1-EXP$2.py"
cp train_model.py $temp_file
echo >> $temp_file
echo "from $1 import $1" >> $temp_file
echo "trainer($1(), \"$3\", $2, \"$4\").train()" >> $temp_file

cd ../../
python src/models/$temp_file &> "./models/$1/EXP$2-$3-$4/output"
rm src/models/$temp_file
