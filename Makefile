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
CFLAGS := -Wall -std=c++17 # -g
CUFLAGS := -m64 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
LIB := -lGL -lGLU -lglut -lGLEW -lboost_program_options -lcudart
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
	nvcc -ccbin $(CC) $(INC) $(CUFLAGS) -c -o $@ $<

clean:
	@echo "\033[1;37mCleaning...\033[0m"; 
	$(RM) -r $(BUILDDIR) $(TARGET) *.qdrep *.sqlite callgrind.out*


.PHONY: clean