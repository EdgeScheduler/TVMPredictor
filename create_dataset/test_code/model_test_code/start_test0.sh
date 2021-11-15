#!/bin/bash

SCRIPTFOLD="$( cd "$( dirname "$0"  )" && pwd )"
cd $SCRIPTFOLD/../../../
echo "run in fold: $PWD"

echo "---start to test all models---"

index=1
COUNT=`find $SCRIPTFOLD/run0/ -type f | wc -l`
for data in $SCRIPTFOLD/run0/*
do
    echo "start to run($index/$COUNT): python3 -u \"$data\""
    python3 $data
    
    echo -e "done($index/$COUNT): python3 -u \"$data\".\n"
    index=`expr $index + 1`
done 
echo "---finished---"