CC=g++
STD=-std=c++11
CFLAGS+=-Wall `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

PROG=nrdc

OBJS=$(PROG).o

.PHONY: all clean

$(PROG): $(OBJS)
	$(CC) $(STD) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(STD) -c $(CFLAGS) $<

%-o: %.cpp
	$(CC) $(STD) $(CFLAGS) $< -o $@

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG) $(OBJS2) $(MTB)
