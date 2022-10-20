#!/bin/bash

read -p "Test or Full? Answering Full will result in a very large directory being downloaded on your machine " type

if [ "$type" == "Full" ]
then
	aws s3 cp s3://pmc-oa-opendata/oa_comm/txt/all/ ./data \
        --no-sign-request \
        --recursive \

else
        RANDOM=$$
	a=$(( ( RANDOM % 9 ) + 1 ))
	b=$(( ( RANDOM % 9 ) + 1 ))
	c=$(( ( RANDOM % 9 ) + 1 ))
	d=$(( ( RANDOM % 9 ) + 1 ))
	e=$(( ( RANDOM % 9 ) + 1 ))
	echo "Downloading files which match last five digits $a$b$c$d$e"
	aws s3 cp s3://pmc-oa-opendata/oa_comm/txt/all/ ./data \
	--no-sign-request \
	--recursive \
	--exclude "*" \
	--include "PMC**$a$b$c$d$e.txt" 
fi

