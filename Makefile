CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic

all: simulator

simulator: simulator.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

clean:
	rm -f simulator

