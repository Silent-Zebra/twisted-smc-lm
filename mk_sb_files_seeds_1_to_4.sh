#!/bin/bash
# First arg should be the full filename of the sb file
for i in `seq 1 4`
do 
	cp $1 $1_s$i
	sed -i "s@seed 0@seed $i@" $1_s$i
	sed -i "s@-J s0@-J s$i@" $1_s$i
	sed -i "s@0.txt@$i.txt@" $1_s$i
done
