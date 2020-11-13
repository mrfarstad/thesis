#!/bin/bash
if [ $1 = "smem" ]; then
make ID=prod BLOCK_X="$2" BLOCK_Y="$3" HOST="$4" SMEM=true #BUILD=debug
elif [ $1 = "coop_smem" ]; then
make ID=prod BLOCK_X="$2" BLOCK_Y="$3" HOST="$4" SMEM=true COOP=true #BUILD=debug
elif [ $1 = "coop" ]; then
make ID=prod BLOCK_X="$2" BLOCK_Y="$3" HOST="$4" COOP=true #BUILD=debug
else
make ID=prod BLOCK_X="$2" BLOCK_Y="$3" HOST="$4" #BUILD=debug
fi
