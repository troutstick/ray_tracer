#!/bin/bash
# convert program output into PNGs to be exported

output="./images/output"

echo "Exporting images..."

shopt -s nullglob
for filename in $output/*.ppm
do
  filename=$(basename -- "$filename")
  filename="${filename%.*}"
  echo "Converting $filename.ppm to .png"
  convert $output/$filename.ppm $output/$filename.png
  rm $output/$filename.ppm
done