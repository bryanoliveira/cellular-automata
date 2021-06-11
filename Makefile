CC := g++
SRCDIR := src
BUILDDIR := build
TARGET := automata

EXTRA_FLAGS := $(EXTRA_FLAGS)
ifdef CPU_ONLY
EXTRA_FLAGS := $(EXTRA_FLAGS) -DCPU_ONLY
endif
ifdef HEADLESS_ONLY
EXTRA_FLAGS := $(EXTRA_FLAGS) -DHEADLESS_ONLY
endif

CFLAGS := -Wall -std=c++17 $(EXTRA_FLAGS) # -g
ifdef OLDER_CUDA
CUFLAGS := $(EXTRA_FLAGS) -m64 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
else
CUFLAGS := $(EXTRA_FLAGS) -m64 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
endif

SOURCES := $(shell find $(SRCDIR) -type f -name *.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))

ifdef CPU_ONLY
CUSOURCES := $()
CUOBJECTS := $()
else
CUSOURCES := $(shell find $(SRCDIR) -type f -name *.cu)
CUOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUSOURCES:.cu=.o))
endif

LIB := -lboost_program_options
ifndef HEADLESS_ONLY
LIB := $(LIB) -lGL -lGLU -lglut -lGLEW
endif
ifndef CPU_ONLY
LIB := $(LIB) -lcudart
endif

INC := -I include
ifndef CPU_ONLY
INC := $(INC) -I/usr/local/cuda/include
endif

run: $(TARGET)
	@echo "\033[1;37mRunning" $(TARGET) "\033[0m"; 
	./$(TARGET) --render

profile: $(TARGET)
	@echo "\033[1;37mProfiling" $(TARGET) "\033[0m"; 
	nsys profile --stats=true -o report ./$(TARGET) -m 5 -d 0 -x 1000 -y 1000

$(TARGET): $(OBJECTS) $(CUOBJECTS)
	@echo "\033[1;37mLinking" $(TARGET) "\033[0m"
	$(CC) $^ -o $(TARGET) $(LIB)
	@echo "\033[1;37mCompiled successfully\033[0m"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	@mkdir -p $(BUILDDIR)
	nvcc -ccbin $(CC) $(INC) $(CUFLAGS) -c -o $@ $<

clean:
	@echo "\033[1;37mCleaning...\033[0m"; 
	$(RM) -r $(BUILDDIR) $(TARGET) *.qdrep *.sqlite callgrind.out*


.PHONY: clean