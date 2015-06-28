CC=g++
STD=-std=c++11
CFLAGS+=-g -Wall `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

PROG=nrdc

CXX_SRCS := $(wildcard *.cpp)
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
OBJS := $(CXX_OBJS)

.PHONY: all clean

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(STD) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(STD) -c $(CFLAGS) $<

%-o: %.cpp
	$(CC) $(STD) $(CFLAGS) $< -o $@

clean:
	@- $(RM) $(PROG)
	@- $(RM) $(OBJS)
