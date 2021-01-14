#!/bin/bash
# execute main.py

IFS="="
read -ra ADDR <<< "$(grep "mode" gen_args.txt)"
MODE="${ADDR[1]}"

if [ $MODE == "train" ]; then
    python main.py @gen_args.txt train @train_args.txt

elif [ $MODE == "predict" ]; then
    python main.py @gen_args.txt predict @predict_args.txt

fi