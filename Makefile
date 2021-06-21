build : src
	cargo r --release
	./scripts/export.sh
	eog ./images/output/
