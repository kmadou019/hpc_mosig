# Makefile qui genere l'executable distanceEdition et fait des tests de verification
#
#
CC=gcc
NVCC=nvcc
LATEXC=pdflatex
DOCC=doxygen
CFLAGS=-g -Wall 

REFDIR=.
SRCDIR=$(REFDIR)/src
BINDIR=$(REFDIR)/bin
DOCDIR=$(REFDIR)/doc
TESTDIR=$(REFDIR)/tests
REPORTDIR=$(REFDIR)/report

LATEXSOURCE=$(wildcard $(REPORTDIR)/*.tex)
CSOURCE := $(wildcard $(SRCDIR)/*.c) $(wildcard $(SRCDIR)/*.cu)
PDF=$(LATEXSOURCE:.tex=.pdf)

#all: binary report doc binary_perf
all: binary report binary_perf

binary: $(BINDIR)/distanceEdition

binary_perf: $(BINDIR)/distanceEdition-perf

$(BINDIR)/distanceEdition-perf: $(SRCDIR)/distanceEdition.cu $(BINDIR)/Needleman-Wunsch-recmemo.o
	$(CC) $(OPT) -D__PERF_MESURE__ -I$(SRCDIR) -o $(BINDIR)/distanceEdition-perf $(BINDIR)/Needleman-Wunsch-recmemo.o $(SRCDIR)/distanceEdition.cu 

report: $(PDF) 

#doc: $(DOCDIR)/index.html

# Compile utils.cu
$(BINDIR)/utils.o: $(SRCDIR)/utils.cu $(SRCDIR)/utils.h
	$(NVCC) $(OPT) -I$(SRCDIR) -c -o $@ $<

# Compile Needleman-Wunsch-recmemo.cu
$(BINDIR)/Needleman-Wunsch-recmemo.o: $(SRCDIR)/Needleman-Wunsch-recmemo.cu $(SRCDIR)/Needleman-Wunsch-recmemo.h $(SRCDIR)/characters_to_base.h $(BINDIR)/utils.o
	$(NVCC) $(OPT) -I$(SRCDIR) -c -o $@ $<

# Compile distanceEdition.cu
$(BINDIR)/distanceEdition.o: $(SRCDIR)/distanceEdition.cu
	$(NVCC) $(OPT) -I$(SRCDIR) -c -o $@ $<

# Link all object files into the final executable
$(BINDIR)/distanceEdition: $(BINDIR)/utils.o $(BINDIR)/Needleman-Wunsch-recmemo.o $(BINDIR)/distanceEdition.o
	$(NVCC) $(OPT) -o $@ $^
	
$(BINDIR)/extract-fasta-sequences-size: $(SRCDIR)/extract-fasta-sequences-size.c
	$(CC) $(OPT) -I$(SRCDIR) -o $(BINDIR)/extract-fasta-sequences-size $(SRCDIR)/extract-fasta-sequences-size.c

clean:
	rm -rf $(DOCDIR) $(BINDIR)/* $(REPORTDIR)/*.aux $(REPORTDIR)/*.log  $(REPORTDIR)/rapport.pdf $(REFDIR)/valgrind4perf* $(REFDIR)/cachegrind.out.*

#$(BINDIR)/distanceEdition: $(CSOURCE)
#	$(CC) $(CFLAGS)  $^ -o $@ 

$(BINDIR)/distanceEditiondebug: $(CSOURCE)
	$(CC) $(CFLAGS)  $^ -o $@ -DDEBUG

%.pdf: $(LATEXSOURCE)
	$(LATEXC) -output-directory $(REPORTDIR) $^ 

#$(DOCDIR)/index.html: $(SRCDIR)/Doxyfile $(CSOURCE)
#	$(DOCC) $(SRCDIR)/Doxyfile


test: $(BINDIR)/distanceEdition $(TESTDIR)/Makefile-test
	cd $(TESTDIR) ; make -f Makefile-test all 
	
test-valgrind: $(BINDIR)/distanceEdition $(TESTDIR)/Makefile-test
	make -f $(TESTDIR)/Makefile-test all-valgrind
	
.PHONY: all doc bin report 

