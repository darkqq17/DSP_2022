SRIPATH ?= /root/srilm-1.5.10
MACHINE_TYPE ?= i686-m64
INC_PATH ?= inc
SRC_PATH ?= src

CXX = g++ 
CXXFLAGS = -O2 -I$(SRIPATH)/include -I$(INC_PATH) -w --std=c++11
vpath lib%.a $(SRIPATH)/lib/$(MACHINE_TYPE)
vpath %.c $(SRC_PATH)
vpath %.cpp $(SRC_PATH)

TARGET = mydisambig
TARGET2 = mydisambig_trigram
SRC = mydisambig.cpp
OBJ = $(SRC:.cpp=.o)
OBJ2 = mydisambig_trigram.o
FROM ?= Big5-ZhuYin.map
TO ?= ZhuYin-Big5.map
.PHONY: all clean map

all: $(TARGET) $(TARGET2)

$(TARGET): $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

$(TARGET2): $(OBJ2) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

map:
	python3 $(SRC_PATH)/transfer.py $(FROM) $(TO)

clean:
	$(RM) $(OBJ) $(TARGET)