CC      ?= cc
CFLAGS  ?= -O3 -std=c99

.PHONY: all test lib clean

all: test

test:
	go test ./...

# Standalone static library, for size reporting or non-Go consumers.
lib: libcvmatch.a
	@ls -l libcvmatch.a

libcvmatch.a: cvmatch.c cvmatch.h
	$(CC) $(CFLAGS) -c cvmatch.c -o cvmatch.o
	ar rcs $@ cvmatch.o

clean:
	rm -f cvmatch.o libcvmatch.a
