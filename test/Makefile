CXX=g++
CXXFLAGS=-std=c++17 -I../include

TARGET=histo.test

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
