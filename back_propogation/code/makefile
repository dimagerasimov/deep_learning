CC=g++
CFLAGS=-std=c++11 -O3
EXEC=deep_learning_lab1
FILES.C:=$(wildcard *.cpp)
FILES.OBJ:=$(patsubst %.cpp, %.o, $(FILES.C))

.PHONY: run clean

%.o: %.c
	@ echo "Compiling: $<"
	@ $(CC) $(CFLAGS) -c $< -o $@
$(EXEC): $(FILES.OBJ)
	@ echo "Linking: $^"
	@ $(CC) $^ -o $@
run:
	@ ./$(EXEC)
clean:
	@ rm -rv *.o
