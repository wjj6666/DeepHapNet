// This script takes a text file containing a whitespace-separated matrix of integers and
// stores the matrix in a sparse format. The sparse format is a text file with one line per
// row of the matrix. Each line contains non-zero indices separated by whitespace.

#include <string>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <unistd.h>
#include <vector>

int main(int argc, char* argv[]){

const char* const opts_str = "m:o:";
	int option;
    std::string matrix_file, output_file;

	while((option = getopt(argc, argv, opts_str)) != -1){
		// cout<<"OPTION: "<<option<<endl;
		switch(option){
            case 'm':
                matrix_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case '?':
				break;
		}
    }

    std::string line;
    bool first = true;        
    int index, val;
    std::vector<int> indices;
    std::vector<int> vals;
    std::ofstream wfile(output_file, std::ios::out); \
    std::ifstream SNVmat_file(matrix_file, std::ios::in);
    while(std::getline(SNVmat_file, line)){
        std::istringstream iss(line);
        indices.clear();
        vals.clear();
        index = 0;
        while(iss >> val){
            if(val != 0){  // only store non-zero values
                indices.push_back(index);
                vals.push_back(val);
            }
            index++;
        }
        if(first){
            wfile << index << std::endl;  // number of columns in the matrix
            first = false;
        }

        for(int i = 0; i < indices.size(); i++){
            wfile << indices[i] << "," << vals[i] << " ";
        }
        wfile << std::endl;
    }
}