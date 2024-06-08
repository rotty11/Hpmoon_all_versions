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
FONSECA = hv-1.3-Fonseca
AMDAPPSDK = /opt/AMDAPPSDK-2.9-1

COMP = g++
CXXFLAGS = -c -Iinclude -Wno-unused-result
OPT = -O3 -funroll-loops -ftree-vectorize -march=native
OMP = -fopenmp

POPSIZE = -D POPULATION_SIZE=$(POPULATION_SIZE)
NFEATURES = -D N_FEATURES=$(N_FEATURES)
NOBJECTIVES = -D N_OBJECTIVES=$(N_OBJECTIVES)
NINSTANCES = -D N_INSTANCES=$(N_INSTANCES)
VARS = $(POPSIZE) $(NFEATURES) $(NOBJECTIVES) $(NINSTANCES)

OBJECTS = $(OBJ)/tinyxml2.o $(OBJ)/config.o $(OBJ)/clUtils.o $(OBJ)/bd.o $(OBJ)/ag.o $(OBJ)/evaluation.o $(OBJ)/individual.o $(OBJ)/main.o

# ************ Targets ************

all: $(BIN)/hpmoon

# ******** Fonseca library ********

fonseca:
	@cd $(FONSECA); $(MAKE) -j -s

# ************ Documentation ************

documentation:
	@mkdir -p $(DOC)/html
	@doxygen $(DOC)/Doxyfile

# ************ Compilation of modules ************

$(OBJ)/tinyxml2.o: $(SRC)/tinyxml2.cpp $(INC)/tinyxml2.h
	@mkdir -p $(OBJ)
	$(COMP) $(CXXFLAGS) $(OPT) $(SRC)/tinyxml2.cpp -o $(OBJ)/tinyxml2.o

$(OBJ)/config.o: $(SRC)/config.cpp $(INC)/config.h $(AMDAPPSDK)/include
	$(COMP) $(CXXFLAGS) $(VARS) $(OPT) -I$(AMDAPPSDK)/include $(SRC)/config.cpp -o $(OBJ)/config.o
$(OBJ)/clUtils.o: $(SRC)/clUtils.cpp $(INC)/clUtils.h $(AMDAPPSDK)/include
	$(COMP) $(CXXFLAGS) $(NFEATURES) $(NOBJECTIVES) $(OPT) -I$(AMDAPPSDK)/include $(SRC)/clUtils.cpp -o $(OBJ)/clUtils.o
$(OBJ)/bd.o: $(SRC)/bd.cpp $(INC)/bd.h
	$(COMP) $(CXXFLAGS) $(OPT) $(SRC)/bd.cpp -o $(OBJ)/bd.o
$(OBJ)/ag.o: $(SRC)/ag.cpp $(INC)/ag.h $(AMDAPPSDK)/include
	$(COMP) $(CXXFLAGS) $(NFEATURES) $(NOBJECTIVES) $(OPT) -I$(AMDAPPSDK)/include $(SRC)/ag.cpp -o $(OBJ)/ag.o
$(OBJ)/evaluation.o: $(SRC)/evaluation.cpp $(INC)/evaluation.h $(FONSECA)/hv.h $(AMDAPPSDK)/include
	$(COMP) $(CXXFLAGS) $(NFEATURES) $(NOBJECTIVES) $(NINSTANCES) $(OPT) -I$(AMDAPPSDK)/include $(OMP) -I$(FONSECA) $(SRC)/evaluation.cpp -o $(OBJ)/evaluation.o
$(OBJ)/individual.o: $(SRC)/individual.cpp $(INC)/individual.h
	$(COMP) $(CXXFLAGS) $(NFEATURES) $(NOBJECTIVES) $(OPT) $(SRC)/individual.cpp -o $(OBJ)/individual.o

$(OBJ)/main.o: $(SRC)/main.cpp $(AMDAPPSDK)/include
	$(COMP) $(CXXFLAGS) $(VARS) $(OPT) -I$(AMDAPPSDK)/include $(SRC)/main.cpp -o $(OBJ)/main.o	

# ************ Linking and creating executable ************

$(BIN)/hpmoon: fonseca $(OBJECTS)
	@mkdir -p $(BIN) $(GNUPLOT)
	$(COMP) $(OBJECTS) -o $(BIN)/hpmoon -lOpenCL $(OMP) $(FONSECA)/fpli_hv.a

# ************ Cleaning ***************

clean:
	@echo "Partial cleaning..."
	@printf "\t- Binary files\n"
	@printf "\t- .o files\n"
	@printf "\t- ~ files\n"
	@printf "\t- Hypervolume project of Fonseca\n"
	@\rm -rf $(OBJ) $(BIN) *~
	@cd $(FONSECA); $(MAKE) -s clean

eraseAll: clean
	@echo "Additionally..."
	@printf "\t- gnuplot files\n"
	@printf "\t- Documentation files\n"
	@\rm -rf $(GNUPLOT) $(DOC)/html