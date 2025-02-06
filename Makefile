CC = gcc
CFLAGS = -Wall -Wextra -I.
DEPS = autodiff.h
OBJ = main.o autodiff.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o main