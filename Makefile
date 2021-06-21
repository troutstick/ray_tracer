OUTPUT=images/output/
INPUT=images/input/

run : src $(INPUT)
	cargo r --release

all : run
	./scripts/export.sh
	eog ./$(OUTPUT)

display : $(OUTPUT)
	./scripts/export.sh
	eog ./$(OUTPUT)

clean : $(OUTPUT)
	rm -r ./$(OUTPUT)