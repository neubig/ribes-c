CC=g++
CXXFLAGS=-g -O0 -Wall
LDFLAGS=-lboost_program_options

ribes-c: ribes-c.o
	$(CC) ribes-c.o -o ribes-c $(LDFLAGS)

clean:
	rm -f ribes-c ribes-c.o
