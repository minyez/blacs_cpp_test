#
.SUFFIXES = .cpp

sources = $(wildcard *.cpp)
objects = $(patsubst %.cpp,%.o,$(sources))

PROG = test.exe
CXX = mpiicpc
INC = -I$(MKLROOT)/include

ifneq (,$(findstring mpi,$(CXX)))
	CXXFLAGS = -O0 -g -D__MPI__
	LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else
	CXXFLAGS = -O0 -g
	LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl
endif

all: $(PROG)

$(PROG): $(objects)
	$(CXX) -o $@ $(INC) $(CXXFLAGS) $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(INC) $(CXXFLAGS) -c $<

.PHONY: all clean mods

clean:
	rm -f *.o $(PROG)
