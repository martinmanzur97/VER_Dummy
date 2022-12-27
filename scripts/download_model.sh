#!/bin/bash -e
MODEL=$1

REPO_FOLDER=$(pwd)
MODEL_REPO_LINK="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/"
MODEL_FOLDER=$REPO_FOLDER/model

if [ ! -f "$MODEL_FOLDER/$MODEL.xml"  ]; then
    echo "Downloading Model.. "
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.xml \
     --create-dirs -o $MODEL_FOLDER/$MODEL.xml
fi

if [ ! -f "$MODEL_FOLDER/$MODEL.bin"  ]; then
    curl $MODEL_REPO_LINK/$MODEL/FP32/$MODEL.bin \
     --create-dirs -o $MODEL_FOLDER/$MODEL.bin
    echo "Model Downloaded"
fi
