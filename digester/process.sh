files="twts.json twts1.json twts2.json twts3.json twts4.json twts5.json"

for file in $files
do
	echo "Processing ${file}..."

	python3 ingester.py $file "digested_${file}"
done
exit 0
