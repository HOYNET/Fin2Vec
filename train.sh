#! /bin/bash

DATAFILE="./data/data_0.csv"
CODEFILE="./data/NASDAQ_FC_STK_IEM_IFO.csv"
BATCH="25"
EPOCHS="800"
LR="0.0001"
EMBEDDINGSIZE="10,10"
TERM="800"
PT_MODEL="./checkpoints_10/hoynet_799.pth"
TESTFILE="./data/NASDAQ_DT_FC_STK_QUT.csv"
DEVICE="cuda:1"

python training.py -d $DEVICE -p $DATAFILE -c $CODEFILE\
 -b $BATCH -e $EPOCHS -l $LR -s $EMBEDDINGSIZE\
 -t $TERM -m $PT_MODEL --testFile $TESTFILE
