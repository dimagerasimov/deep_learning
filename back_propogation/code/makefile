EXEC=deep_learning_lab1
.PHONY: run clean

$(EXEC): $(FILES.OBJ)
	g++ -std=c++11 -O3 network.cpp utils.cpp main.cpp -o $@
run:
	@ ./$(EXEC)
clean:
	@ rm -rv *.o
