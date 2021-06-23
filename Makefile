OUTPUT=images/output/
INPUT=images/input/
IMAGE_NAME=teapot
NUM_ITER=1

run : src $(INPUT)
	cargo r --release $(IMAGE_NAME) $(NUM_ITER)

all : run
	./scripts/export.sh
	eog ./$(OUTPUT)

display : $(OUTPUT)
	./scripts/export.sh
	eog ./$(OUTPUT)

clean : $(OUTPUT)
	rm -r ./$(OUTPUT)