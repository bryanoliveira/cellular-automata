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
CFLAGS := -g -Wall
LIB := -lGL -lGLU -lglut -lcudart
INC := -I include

$(TARGET): $(OBJECTS) $(CUOBJECTS)
	@echo " Linking..."
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(CUEXT)
	@mkdir -p $(BUILDDIR)
	nvcc -ccbin $(CC) $(INC) --machine=64 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	$(RM) -r $(BUILDDIR) $(TARGET)

run:
	@echo " Running..."; 
	./$(TARGET)

.PHONY: clean