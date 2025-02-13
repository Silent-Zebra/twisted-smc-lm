#!/bin/bash
# First arg should be the full filename of the sb file
for i in `seq 2 5`
do 
	cp $1 $1_s$i
	sed -i "s@seed 1@seed $i@" $1_s$i
	sed -i "s@-J s1@-J s$i@" $1_s$i
	sed -i "s@1.txt@$i.txt@" $1_s$i
done
