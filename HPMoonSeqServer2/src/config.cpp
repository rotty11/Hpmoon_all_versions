/**
 * @file config.cpp
 * @author Juan José Escobar Pérez
 * @date 14/02/2016
 * @brief File with the necessary implementation for extracting the program configuration parameters from command-line or from the XML configuration
 *
 */

/********************************* Includes *******************************/

#include "config.h"

using namespace std;

/********************************* Methods ********************************/

Config::Config(const char **argv, const int argc) {


	/************ Init the parser ***********/

	ezParser parser;
	parser.overview = "High computing: Genetic algorithm in C++";
	parser.syntax = "./bin/hpmoonSeq [OPTIONS]";
	parser.example = "./bin/hpmoonSeq -h\n";
	parser.example += "./bin/hpmoonSeq -conf \"config.xml\" -db \"datos.txt\"\n";
	parser.example += "./bin/hpmoonSeq -conf \"config.xml\" -ts 4 -maxf 85 -plotimg \"imgPareto\" -db \"datos.txt\"\n\n";
	parser.footer = "Cluster HPMoon (C) 2016\n";

	// Options
	parser.add("", 0, 0, 0, "Display usage instructions.", "-h"); // Display help
	parser.add("", 1, 1, 0, "Name of the file containing the XML configuration file.", "-conf"); // XML
	parser.add("", 0, 1, 0, "Name of the file containing the database.", "-db"); // DataBase
	parser.add("", 0, 1, 0, "Number of generations.", "-g"); // Number of generations
	parser.add("", 0, 1, 0, "Maximum number of features initially set to \"1\".", "-maxf"); // Max. features
	parser.add("", 0, 1, 0, "Number of individuals competing in the tournament.", "-ts"); // Tournament size
	parser.add("", 0, 1, 0, "Name of the file containing the fitness of the individuals in the first Pareto front.", "-plotdata"); // Gnuplot data
	parser.add("", 0, 1, 0, "Name of the file containing the gnuplot code for data display.", "-plotsrc"); // Gnuplot code
	parser.add("", 0, 1, 0, "Name of the file containing the image with the data (graphic).", "-plotimg"); // Gnuplot image

	parser.parse(argc, argv);
	string option;


	/************ Check important parameters ***********/

	// Help option
	if (parser.isSet("-h")) {
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(0);
	}

	// Missing options
	vector<string> badOptions;
	if (!parser.gotRequired(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "ERROR: Missing required option %s\n", badOptions[i].c_str());
		}
		exit(-1);
	}

	// Missing arguments
	if (!parser.gotExpected(badOptions)) {
		for (int i = 0; i < badOptions.size(); ++i) {
			fprintf(stderr, "Got unexpected number of arguments for option %s\n\n", badOptions[i].c_str());
		}
		parser.getUsage(option);
		fprintf(stdout, "%s", option.c_str());
		exit(-1);
	}


	/************ Get the parameters ***********/

	////////////////////// -conf value
	parser.get("-conf") -> getString(option);
	XMLDocument configDoc;
	configDoc.LoadFile(option.c_str());


	////////////////////// -db value
	if (parser.isSet("-db")) {
		parser.get("-db") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") -> FirstChildElement("DataBaseFileName") -> GetText();
	}
	this -> dataBaseFileName = new char[option.length() + 1];
	strcpy(this -> dataBaseFileName, option.c_str());


	////////////////////// -g value
	if (parser.isSet("-g")) {
		parser.get("-g") -> getInt(this -> nGenerations);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("NGenerations") -> QueryIntText(&(this -> nGenerations));
	}


	////////////////////// -maxf value
	if (parser.isSet("-maxf")) {
		parser.get("-maxf") -> getInt(this -> maxFeatures);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("MaxFeatures") -> QueryIntText(&(this -> maxFeatures));
	}


	////////////////////// -ts value
	if (parser.isSet("-ts")) {
		parser.get("-ts") -> getInt(this -> tourSize);
	}
	else {
		configDoc.FirstChildElement("Config") ->  FirstChildElement("TournamentSize") -> QueryIntText(&(this -> tourSize));
	}


	////////////////////// -plotdata value
	if (parser.isSet("-plotdata")) {
		parser.get("-plotdata") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("DataFileName") -> GetText();
	}
	this -> dataFileName = new char[option.length() + 1];
	strcpy(this -> dataFileName, option.c_str());


	////////////////////// -plotsrc value
	if (parser.isSet("-plotsrc")) {
		parser.get("-plotsrc") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("PlotFileName") -> GetText();
	}
	this -> plotFileName = new char[option.length() + 1];
	strcpy(this -> plotFileName, option.c_str());


	////////////////////// -plotimg value
	if (parser.isSet("-plotimg")) {
		parser.get("-plotimg") -> getString(option);
	}
	else {
		option = configDoc.FirstChildElement("Config") ->  FirstChildElement("ImageFileName") -> GetText();
	}
	this -> imageFileName = new char[option.length() + 1];
	strcpy(this -> imageFileName, option.c_str());
}