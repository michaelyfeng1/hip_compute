CC=hipcc
CFLGS=-std=c++11 
LIBS=-lm -lncurses
INCLUDES=-Iopt/rocm/include -Iinc

SOURCE=$(shell find . -name "*.cpp")

OBJ=$(patsubst %.cpp,%.o, $(SOURCE)) 

all: $(OBJ)
	$(CC) $(INCLUDES) $(LIBS) $(CFLGS) -o $@ $^ 
	
%.o: %.cpp 
	$(CC) $(INCLUDES) $(LIBS) -c -o $@ $< $(CFLAGS) 

.PHONY: clean 

clean:
	rm *.o

run: all
	./all
