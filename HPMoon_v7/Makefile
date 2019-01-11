## @file Makefile
## @author Juan José Escobar Pérez
## @date 15/06/2015
## @brief File with the necessary rules for compile the project. Also, documentation is generated

SRC = src
INC = include
OBJ = obj
BIN = bin
DOC = doc
GNUPLOT = gnuplot
OPENCL = $(AMDAPPSDKROOT)/include

COMP ?= mpic++
CPPFLAGS = -std=c++0x -c -Iinclude
OPT = -O2 -funroll-loops
OMP = -fopenmp

NFEATURES = -D N_FEATURES=$(N_FEATURES)

OBJECTS = $(OBJ)/tinyxml2.o $(OBJ)/cmdParser.o $(OBJ)/config.o $(OBJ)/clUtils.o $(OBJ)/bd.o $(OBJ)/ag.o $(OBJ)/evaluation.o $(OBJ)/individual.o $(OBJ)/zitzler.o $(OBJ)/main.o

# ************ Targets ************

all: $(BIN)/hpmoon

# ************ Documentation ************

documentation:
	@mkdir -p $(DOC)/html
	@doxygen $(DOC)/Doxyfile

# ************ Compilation of modules ************

$(OBJ)/tinyxml2.o: $(SRC)/tinyxml2.cpp $(INC)/tinyxml2.h
	@mkdir -p $(OBJ)
	$(COMP) $(CPPFLAGS) $(OPT) $(SRC)/tinyxml2.cpp -o $(OBJ)/tinyxml2.o

$(OBJ)/cmdParser.o: $(SRC)/cmdParser.cpp $(INC)/cmdParser.h
	$(COMP) $(CPPFLAGS) $(OPT) $(SRC)/cmdParser.cpp -o $(OBJ)/cmdParser.o
$(OBJ)/config.o: $(SRC)/config.cpp $(INC)/config.h $(OPENCL)
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) -I$(OPENCL) $(SRC)/config.cpp -o $(OBJ)/config.o
$(OBJ)/clUtils.o: $(SRC)/clUtils.cpp $(INC)/clUtils.h $(OPENCL)
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) -I$(OPENCL) $(SRC)/clUtils.cpp -o $(OBJ)/clUtils.o
$(OBJ)/bd.o: $(SRC)/bd.cpp $(INC)/bd.h
	$(COMP) $(CPPFLAGS) $(OPT) $(SRC)/bd.cpp -o $(OBJ)/bd.o
$(OBJ)/ag.o: $(SRC)/ag.cpp $(INC)/ag.h $(OPENCL)
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) $(OMP) -I$(OPENCL) $(SRC)/ag.cpp -o $(OBJ)/ag.o
$(OBJ)/evaluation.o: $(SRC)/evaluation.cpp $(INC)/evaluation.h $(OPENCL)
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) $(OMP) -I$(OPENCL) $(SRC)/evaluation.cpp -o $(OBJ)/evaluation.o
$(OBJ)/individual.o: $(SRC)/individual.cpp $(INC)/individual.h
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) $(SRC)/individual.cpp -o $(OBJ)/individual.o
$(OBJ)/zitzler.o: $(SRC)/zitzler.cpp $(INC)/zitzler.h
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) $(SRC)/zitzler.cpp -o $(OBJ)/zitzler.o

$(OBJ)/main.o: $(SRC)/main.cpp $(OPENCL)
	$(COMP) $(CPPFLAGS) $(NFEATURES) $(OPT) -I$(OPENCL) $(SRC)/main.cpp -o $(OBJ)/main.o

# ************ Linking and creating executable ************

$(BIN)/hpmoon: $(OBJECTS)
	@mkdir -p $(BIN) $(GNUPLOT)
	$(COMP) $(OBJECTS) -o $(BIN)/hpmoon -lOpenCL $(OMP)

# ************ Cleaning ***************

clean:
	@echo "Partial cleaning..."
	@printf "\t- Binary files\n"
	@printf "\t- .o files\n"
	@printf "\t- ~ files\n"
	@\rm -rf $(OBJ) $(BIN) *~

eraseAll: clean
	@echo "Additionally..."
	@printf "\t- gnuplot files\n"
	@printf "\t- Documentation files\n"
	@\rm -rf $(GNUPLOT) $(DOC)/html
