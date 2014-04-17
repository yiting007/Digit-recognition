filename=$1

awk '!x[$0]++' $filename
