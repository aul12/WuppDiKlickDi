#!/bin/bash

PLAYLIST_URL=https://www.youtube.com/watch\?v\=iYp3vVq2jNM\&list\=PLkMeFiwcGZmMsbtPb2KRXAThhlTiITCKL
ROOT=$(pwd)

mkdir -p $ROOT/data/raw
cd $ROOT/data/raw
#youtube-dl -i -f bestaudio $(PLAYLIST_URL)
mkdir -p $ROOT/data/wav
parallel -j$(nproc) ffmpeg -i $ROOT/data/raw/{} -ac 1 -ar 20000 $ROOT/data/wav/{.}.wav ::: *
cd $ROOT
