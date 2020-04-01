dbs="twts twts1 twts2"

for db in $dbs
do
	sudo mongoexport --db $db --collection collection1 --out "/mnt/disks/disk2/shared2/${db}.json"
done
exit 0
