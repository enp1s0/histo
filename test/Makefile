CXX=g++
CXXFLAGS=-std=c++17 -I../include -fopenmp

TARGET=histo.test

$(TARGET):main.cpp ../include/histo/histo.hpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
