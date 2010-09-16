EXT := cu 

OBJS := rpud.o rpudist.o

#compiler/preprocessor options
INCS := -I/usr/local/cuda/include
PARAMS := -Xcompiler "-I/usr/share/R/include -fpic"

#linker options
LD_PARAMS := -Xlinker "-L/usr/lib/R/lib -lR -Wl,-rpath,/usr/local/cuda/lib"
LIBS :=  -L/usr/local/cuda/lib -lcublas -lcuda

TARGETS := rpud.so

CC := /usr/local/cuda/bin/nvcc

all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(CC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(CC) -c $(INCS) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
