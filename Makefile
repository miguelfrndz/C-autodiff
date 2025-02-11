.PHONY: all cpp clean

# Switch between C and C++ mode
ifeq ($(LANG),C++)
    # C++ mode
    CC      = g++
    SRC     = main.cpp autodiff.cpp
    OBJ     = $(SRC:.cpp=.o)
    DEPS    = autodiff.hpp
    CFLAGS  = -Wall -Wextra -I.
    # Remove .c from the suffixes so implicit rules won’t use it.
    .SUFFIXES: .o .cpp
else
    # Default (C) mode
    CC      = gcc
    SRC     = main.c autodiff.c
    OBJ     = $(SRC:.c=.o)
    DEPS    = autodiff.h
    CFLAGS  = -Wall -Wextra -I.
    .SUFFIXES: .o .c
endif

all: main

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

ifeq ($(LANG),C++)
%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
else
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
endif

# The "cpp" target re‑invokes make with LANG=C++ so that only the C++ files are used.
cpp:
	$(MAKE) LANG=C++ all

clean:
	rm -f *.o main
