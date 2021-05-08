CC := g++
SRCDIR := src
BUILDDIR := build
TARGET := automata
 
SRCEXT := cpp
CUEXT := cu
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
CUSOURCES := $(shell find $(SRCDIR) -type f -name *.$(CUEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CUOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUSOURCES:.$(CUEXT)=.o))
CFLAGS := -Wall -std=c++17 -fopenmp # -g
LIB := -lGL -lGLU -lglut -lGLEW -lboost_program_options -lcudart -fopenmp
INC := -I include -I/usr/local/cuda/include

run: $(TARGET)
	@echo "\033[1;37mRunning" $(TARGET) "\033[0m"; 
	./$(TARGET) --render

profile: $(TARGET)
	@echo "\033[1;37mProfiling" $(TARGET) "\033[0m"; 
	nsys profile --stats=true -o report ./$(TARGET) -m 5 -d 0 -x 1000 -y 1000

$(TARGET): $(OBJECTS) $(CUOBJECTS)
	@echo "\033[1;37mLinking" $(TARGET) "\033[0m"
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(CUEXT)
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	nvcc -ccbin $(CC) $(INC) --machine=64 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -c -o $@ $<

clean:
	@echo "\033[1;37mCleaning...\033[0m"; 
	$(RM) -r $(BUILDDIR) $(TARGET) *.qdrep *.sqlite callgrind.out*


.PHONY: clean