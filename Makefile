OUTPUT=images/output/
INPUT=images/input/

build : src $(INPUT)
	cargo r --release
	./scripts/export.sh
	eog ./$(OUTPUT)
display : $(OUTPUT)
	./scripts/export.sh
	eog ./$(OUTPUT)
clean : $(OUTPUT)
	rm -r ./$(OUTPUT)