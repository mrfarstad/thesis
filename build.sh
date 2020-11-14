#!/bin/bash
if [ $1 = "smem" ]; then
make ID="$2" BUILD="$2" BLOCK_X="$3" BLOCK_Y="$4" HOST="$5" SMEM=true #BUILD=debug
elif [ $1 = "coop_smem" ]; then
make ID="$2" BUILD="$2" BLOCK_X="$3" BLOCK_Y="$4" HOST="$5" SMEM=true COOP=true #BUILD=debug
elif [ $1 = "coop" ]; then
make ID="$2" BUILD="$2" BLOCK_X="$3" BLOCK_Y="$4" HOST="$5" COOP=true #BUILD=debug
else
make ID="$2" BUILD="$2" BLOCK_X="$3" BLOCK_Y="$4" HOST="$5" #BUILD=debug
fi
