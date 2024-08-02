// File to process SAM file with known SNPs and generate read-SNV matrix

#include <algorithm>
#include <random>
#include <iomanip> // std::setprecision
#include <vector> // for use of vector
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream> 
#include <numeric>
#include <time.h>
#include <unistd.h>
#include <climits>
#include <sys/stat.h>
#include "progressbar.hpp"

using namespace std;

//检查给定文件名的文件是否存在
bool fileExists(const std::string& filename){
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1){
        return true;
    }
    return false;
}

//递归函数，将整数转换为二进制字符串
string binary(int number, stringstream& strs) 
{
	int remainder;
	if(number <= 1) 
	{
		strs << number;
	  	return strs.str() ;
	}
	remainder = number%2;
	binary(number >> 1, strs);    
	strs << remainder;
	
	return  strs.str();
}

//计算SAM文件中一行（一个read的对齐）是否有多个区段对齐到参考序列的情况，并返回对齐的数量。
int num_chimeric_aln(const vector<string>& tokens){
	int res = 1;
	for(string tok : tokens){
		if(tok.rfind("SA:Z:", 0) == 0){  // if token starts with "SA:Z:"
			res = count(tok.begin(), tok.end(), ';') + 1;
		}
	}
	return res;
}

//将字符串按给定的分隔符分割成多个子字符串（token）
vector<string> tokenize(const string& str,const string& delimiters)
{
	vector<string> tokens;	
	string::size_type lastPos = 0, pos = 0;  
  	int count = 0;
  	if(str.length()<1)  return tokens;
   
  	lastPos = str.find_first_not_of(delimiters, 0); // skip delimiters at beginning.     
  	if((str.substr(0, lastPos-pos).length()) > 0)
  	{	  	
  		count = str.substr(0, lastPos-pos).length();  	
  		for(int i=0; i < count; i++)  	
  	 		tokens.push_back("");
  		if(string::npos == lastPos)
	  		tokens.push_back("");
	}

  	pos = str.find_first_of(delimiters, lastPos); // find first \"non-delimiter\".
  	while (string::npos != pos || string::npos != lastPos)
 	 {  	      	    
     		tokens.push_back( str.substr(lastPos, pos - lastPos)); // found a token, add it to the vector.				
     		lastPos = str.find_first_not_of(delimiters, pos); // skip delimiters.  Note the \"not_of\"   	   	    
		
		if((string::npos != pos) && (str.substr(pos, lastPos-pos).length() > 1))  		
  		{
  			count = str.substr(pos, lastPos-pos).length();
  			for(int i=0; i < count-1; i++)
  	 			tokens.push_back("");
		}	
  		pos = str.find_first_of(delimiters, lastPos);
	}
	return tokens;
}

//解析SAM文件的一行，提取出有用的对齐信息，如对齐开始位置、AS得分（对齐得分）、插入删除的情况等
void parse_sam_line(const vector<string>& tokens, vector<int>& seq_b, double& a_score, char gap_quality, int& indels, int& al_start)
{  
	seq_b.clear();
	indels = 0;
	
  	bool foundAscore = false;
  	for(int j = 11; j< tokens.size();j++)
	{
  		vector<string> tokens_as = tokenize( tokens[j].c_str(),":");
    		if( tokens_as[0] == "AS")
		{
      			a_score = atof(tokens_as[2].c_str());
      			foundAscore = true;
      			break;
    		}
 	} 
	if (!foundAscore)
		cout << foundAscore << " no a_score "<< a_score << endl;
 
	al_start = atoi( tokens[3].c_str()) ;
 
	vector<int> isAlpha(tokens[5].size(),1);
  	for(int i =0; i< tokens[5].size();i++)
	{
    		if(!isalpha(tokens[5][i]))
      			isAlpha[i] = 0;
	}
  
  	vector<int> sub_length_vec;
  	vector<char> symbols;
  	int sub_length =0;
  	for(int i =0; i< tokens[5].size();i++)
	{
    		if(isAlpha[i] == 1)
		{
      			sub_length_vec.push_back( atoi(tokens[5].substr(i-sub_length,sub_length).c_str()));
      			symbols.push_back(tokens[5][i]);
      			sub_length =0;
    		}
    		if(isAlpha[i] == 0)
     			sub_length++;	
	}
  
  	int c =0;
  	for(int i =0; i< sub_length_vec.size();i++)
	{  
    		if(symbols[i] == 'S')
      			for(int j =0; j< sub_length_vec[i];j++)
				c++;
    		else if(symbols[i] == 'M')
			for(int j =0; j< sub_length_vec[i];j++)
			{
				int k = 0;
				if(tokens[9][c] == 'A' || tokens[9][c] == 'a')
			  		k=1;
				else if(tokens[9][c] == 'C' || tokens[9][c] == 'c')
			  		k=2;
				else if(tokens[9][c] == 'G' || tokens[9][c] == 'g')
			  		k=3;
				else if(tokens[9][c] == 'T' || tokens[9][c] == 't')
			  		k=4;
				seq_b.push_back(k);
				c++;
		      }
		else if(symbols[i] == 'I')
      			for(int j =0; j< sub_length_vec[i];j++)
			{	
				c++;
      			}
    		else if(symbols[i] == 'D')
      			for(int j =0; j< sub_length_vec[i];j++)
			{
				seq_b.push_back(0);
				indels++;	
      			}
	}  
}

//处理分割读数的情况，合并多个对齐片段的信息。
void parse_sam_splitread(vector<vector<string>>& tokens_vec, vector<int>& seq_b, double& a_score, char gap_quality, int& indels, int& al_start, int& qual)
{
	seq_b.clear();
	indels = 0;

	if(tokens_vec.size() == 1)
	{
		parse_sam_line(tokens_vec[0], seq_b, a_score,  gap_quality, indels, al_start);
		qual = atoi(tokens_vec[0][4].c_str());
		return; 
	}


	std::sort(tokens_vec.begin(), tokens_vec.end(), [](const vector<string>& t1, const vector<string>& t2) {
		return atoi(t1[3].c_str()) < atoi(t2[3].c_str());
		}
	);  // sort split alignments by starting aligned position

	// std::sort(tokens_vec.begin(), tokens_vec.end(), comp_tokens);
	vector<string> seq_b_combined;

	vector<int> seq_b_tmp;
	double a_score_tmp;
	int al_start_tmp;
	int qual_tmp;
	int Ngap = 0;

	parse_sam_line(tokens_vec[0], seq_b, a_score,  gap_quality, indels, al_start);

	for(int i=1; i < tokens_vec.size(); i++){ // Skip first element
		vector<string> tok = tokens_vec[i];
		parse_sam_line(tok, seq_b_tmp,  a_score_tmp,  gap_quality, indels, al_start_tmp);
		a_score = a_score + a_score_tmp;
		qual_tmp = atoi(tok[4].c_str());
		if(qual_tmp < qual) {qual = qual_tmp;}  // Store lowest mapping quality

		if(al_start_tmp - (al_start +   (int)seq_b.size())>0)  // Gap between split alignments
		{
			Ngap = al_start_tmp - (al_start + (int)seq_b.size());
			vector<int> Ns(Ngap,0);
			seq_b.insert(seq_b.end(), Ns.begin(), Ns.end());
	      	seq_b.insert(seq_b.end(), seq_b_tmp.begin(), seq_b_tmp.end());
		}
		else{  // No gap between alignments
			seq_b.erase(seq_b.begin() + (al_start_tmp - al_start), seq_b.end());
			seq_b.insert(seq_b.end(), seq_b_tmp.begin(), seq_b_tmp.end());
		}
	}
}

//是主要的函数，负责解析整个SAM文件，提取读数和变异信息，并生成读数-变异矩阵。它同时记录和输出统计信息，如映射率、过滤掉的读段数等。
int parseSAM( string al, double min_qual, char gap_quality, double& mean_length, vector<vector<pair<int, int>>>& SNV_matrix, vector<int> SNV_pos,
		int reconstruction_start, int reconstruction_end, int& total_count, int& read_count, int gene_length, string zonename, int nFrag)
{
	string line;
  	vector<string> tokens;
	vector<vector<string>> tokens_vec;
	int pair_counter = 0, seq_counter =0, singleton_counter = 0;
  	vector<int> seq_b;
  	double a_score;
	int indels, qual;
  	int al_start;
  	bool is_read = false;
	int rcount = 0;
  	string id;
  	int RC;
  
	int filtered_counter1 = 0, filtered_counter2 = 0;
	int mapped_counter = 0, unmapped_counter = 0; 
	int filtered_cond2 = 0;

	vector<int> ReadInd;
	ReadInd.clear();
	SNV_matrix.clear();  // Empty SNV_matrix 

	std::string name = zonename;
	std::ofstream writefile;
	writefile.open(name+"_SNV_matrix.txt");
	writefile << SNV_pos.size() << std::endl;  // number of columns in the matrix
	// if(fileExists(name+"_SNV_matrix.txt")){
	// 	cout<< name + "_SNV_matrix.txt" + " opened." << endl;
	// }
	ifstream inf6(al.c_str(),ios::in);
	int read_i = 0;  // read count


	progressbar bar(nFrag);
	bar.set_todo_char(" ");
	bar.set_done_char("█");

  	while( getline(inf6,line,'\n') )
	{	
		if(nFrag > 0) bar.update();
    	if(line[0] == '@')
      		continue;
	    
		tokens = tokenize(line,"\t");
		if(tokens.size() <5)
		{
			cout << "Problem with sam file..." << endl;
			return 1;
		}
		id =  tokens[0];
	    total_count++;
				
	    RC =  atoi( tokens[1].c_str());
	    stringstream strs;
	    string sRC = binary(RC,  strs);
		int sz = sRC.size();
		if(sz > 8 && RC < 2048) // bit9-12 should be 0 if not chimeric alignment
	      	{ filtered_counter1++;	continue;}
	    if(sRC[sz-3] == '1' ||  sRC[sz-4]  == '1'  ) // bit 3-4 should be 0
	      	{ filtered_counter2++;	continue;}
		
		if(rcount == 0){  // primary alignment of read
			is_read = false;
			tokens_vec.clear();
			rcount = num_chimeric_aln(tokens);
			singleton_counter++;
		}
		else{
			rcount--;
		}
		tokens_vec.push_back(tokens);

		if(rcount == 1){  // read all alignments for this read
			is_read = true;
			rcount = 0;
		}
		
	    if(is_read){
			parse_sam_splitread(tokens_vec, seq_b,  a_score,  gap_quality, indels, al_start, qual);
			read_i++;
			// if(read_i % 10000 == 0)
				// cout << "Read" << " " <<  read_i << " " << SNV_matrix.size() << endl;

	      	if(qual >= min_qual){	 
				mapped_counter++;
				int StartPos = al_start;
				vector<int> SEQ_combined = seq_b;
				bool is_gap = false;
				int Nlength = 0;
	
				filtered_cond2++;
				int EndPos = StartPos + SEQ_combined.size()-1; //range = reconstruction_end-reconstruction_start+1;
				int snv_start_idx = 0, snv_end_idx = 0;
				if ( StartPos <= reconstruction_end && EndPos >= reconstruction_start){
					// vector<int> SEQ_range;
					if (StartPos < reconstruction_start){
						if (EndPos <= reconstruction_end){
							// cout << "CASE 1" << endl;
							// reconstruction_start = SNV_pos[0], so do not need to search
							// while(SNV_pos[snv_idx] < reconstruction_start) snv_idx++;
							snv_end_idx = snv_start_idx;
							while(SNV_pos[snv_end_idx] <= EndPos - reconstruction_start) snv_end_idx++;

							// vector<int> SEQ_inrange(SEQ_combined.begin() + (reconstruction_start-StartPos),SEQ_combined.end());
							// vector<int> Ns(gene_length-SEQ_inrange.size(),0);
							// SEQ_inrange.insert(SEQ_inrange.end(),Ns.begin(),Ns.end());
							// SEQ_range = SEQ_inrange;
						}
						else{
							// cout << "CASE 2" << endl;
							snv_end_idx = SNV_pos.size();  // reconstruction_end = last SNP position

							// vector<int> SEQ_inrange(SEQ_combined.begin()+(reconstruction_start-StartPos),SEQ_combined.begin()+(reconstruction_end-StartPos+1));
							// SEQ_range = SEQ_inrange;
						}
					}
					else{
						if (EndPos <= reconstruction_end){
							// search for first SNP index
							// cout << "CASE 3" << endl;
							while(SNV_pos[snv_start_idx] < StartPos - reconstruction_start) snv_start_idx++;
							snv_end_idx = snv_start_idx;
							while(SNV_pos[snv_end_idx] <= EndPos - reconstruction_start) snv_end_idx++;

							// vector<int> SEQ_inrange = SEQ_combined;
							// vector<int> Ns1(StartPos-reconstruction_start,0);
							// SEQ_inrange.insert(SEQ_inrange.begin(),Ns1.begin(),Ns1.end());
							// vector<int> Ns2(reconstruction_end-EndPos,0);
							// SEQ_inrange.insert(SEQ_inrange.end(),Ns2.begin(),Ns2.end());	
							// SEQ_range = SEQ_inrange;						
						}
						else{
							// cout << "CASE 4" << endl;
							while(SNV_pos[snv_start_idx] < StartPos - reconstruction_start) snv_start_idx++;
							snv_end_idx = SNV_pos.size();

							// vector<int> SEQ_inrange(SEQ_combined.begin(),SEQ_combined.begin()+(reconstruction_end-StartPos+1));
							// vector<int> Ns(StartPos-reconstruction_start,0);
							// SEQ_inrange.insert(SEQ_inrange.begin(),Ns.begin(),Ns.end());
							// SEQ_range = SEQ_inrange;
						}
					}

					vector<pair<int,int>> read;
					int seq_base = 0;
					// cout << SEQ_combined.size()	<< endl;
					// cout << "SNV indices: " << snv_start_idx << " " << snv_end_idx << endl;
					for(int j=snv_start_idx; j<snv_end_idx; j++){
						seq_base = SEQ_combined[SNV_pos[j] + reconstruction_start - StartPos];
						// cout << j << " " << SNV_pos[j] << endl;
						// cout << "Base: " << seq_base << endl;
						if(seq_base > 0){
							read.push_back(make_pair(j, seq_base));
						}
					}
					// cout << "Pushing read" << endl;
					if(read.size() > 1){  // Read covers more than 1 SNV
						// cout << "Number of reads: " << SNV_matrix.size() << endl;
						SNV_matrix.push_back(read);
						ReadInd.push_back(read_i);
					}
					else{
						// deleted_reads_list.push_back(i);  // Delete reads covering 0 or 1 SNVs
						;
					}

					// vector<int> read;
					// int SNV_count = 0;  // number of SNVs covered by read
					// for(int snv : SNV_pos){
					// 	read.push_back(SEQ_range[snv]);
					// 	if(SEQ_range[snv] > 0) SNV_count++;
					// }

					// vector<pair<int,int>> read;
					// int SNV_count = 0;
					// int snv;
					// for(int j=0; j<SNV_pos.size(); j++){
					// 	snv = SNV_pos[j];
					// 	if(SEQ_range[snv] > 0){
					// 		read.push_back(make_pair(j, SEQ_range[snv]));
					// 		SNV_count++;
					// 	}
					// }
					// if(SNV_count > 1){
					// 	SNV_matrix.push_back(read);
					// 	ReadInd.push_back(read_i);  // Read covers more than 1 SNV
					// }
					// else{
					// 	// deleted_reads_list.push_back(i);  // Delete reads covering 0 or 1 SNVs
					// 	;
					// }

					mean_length += SEQ_combined.size();
					seq_counter++;
				}				  
	    	}
			else{ // unmapped or low qual/short seqlen etc
				unmapped_counter++;
				if (qual < min_qual) // || seq_b.size() < min_length)
				{
					vector<int> tag(8,0);
					tag[0] = pair_counter;
					tag[2] = qual;
					tag[3] = al_start;
					tag[4] = seq_b.size();
				}
			}
		}

		if(SNV_matrix.size() > 0 && SNV_matrix.size() % 100 == 0){
			// cin.get();
			// cout << "Printing into file" << endl;
			for(int i=0; i<SNV_matrix.size(); i++){
				for(auto snv_base : SNV_matrix[i]){
					writefile << snv_base.first<<"," << snv_base.second << " ";
				}
				// for (int j = 0; j<SNV_matrix[i].size(); j++){
				// 	cout << i << " " << j << endl;
				// 	writefile << SNV_matrix[i][j].first<<"," << SNV_matrix[i][j].second << " ";
				// 	// cout << i << " " << j << " " << SNV_matrix[i][j] << endl;
				// } 
				writefile << "\n";
			}
			SNV_matrix.clear();
		}

	}
	
	if(SNV_matrix.empty() == false){
		for(int i=0; i<SNV_matrix.size(); i++){
			for(auto snv_base : SNV_matrix[i]){
				writefile << snv_base.first<<"," << snv_base.second << " ";
			}
			writefile << "\n";
		}
		SNV_matrix.clear();
	}
	writefile.close();
	read_count = ReadInd.size();

	cout << "mapped_counter:"<<mapped_counter<< " unmapped_counter:"<<unmapped_counter<< endl;
	cout << "filtered_counter1:"<< filtered_counter1 <<" filtered_counter2:"<<filtered_counter2<<endl;
        cout << " filtered_cond2:"<<filtered_cond2<<endl;
	cout << "pair_counter:" << pair_counter <<" singleton_counter:"<<singleton_counter<< " total_counter:"<<total_count<<endl;
  	return 0;
}

//程序的入口点，处理命令行参数，读取输入文件和参数，调用parseSAM函数处理SAM文件，并输出最终的读-变异矩阵到文件中。
int main(int argc, char* argv[]) {
 
  	srand(time(NULL));
 
	// string cons;
	vector<string> FASTAreads;
	string SNV_pos_file;
	int  reconstruction_start = 0,  reconstruction_end = INT_MAX;
	double  min_qual;
	int nFrag = -1;
	
	string zonename;
	
	const char* const opts_str = ":s:p:b:e:q:z:n:";
	int option;

	while((option = getopt(argc, argv, opts_str)) != -1){
		// cout<<"OPTION: "<<option<<endl;
		switch(option){
		//    case 'f':
		//            cons = optarg;
		//            break;
			case 's':
				FASTAreads.push_back(optarg);
				break;
			case 'p':
				SNV_pos_file = optarg;
				break;
			case 'b':
				reconstruction_start = atoi(optarg);
				break;
			case 'e':
				reconstruction_end = atoi(optarg);
				break;
			case 'q':
				min_qual = atoi(optarg);
				break;
			case 'z':
				zonename = optarg;
				break;
			case 'n':
				nFrag = atoi(optarg);
				cout << "Number of read alignments: " << nFrag << endl;
				break;
			case ':':
				switch(optopt){
					case 'b':
						reconstruction_start = 0;
						break;
					case 'e':
						reconstruction_end = INT_MAX;
						break;
					default: 
						cerr << "Option needs a value." << endl;
				}
			case '?':
				break;
		}

	}
	

	vector<int> SNV_pos;
	string pos;
	ifstream SNVpos_file(SNV_pos_file, ios::in);
	int SNV_count = 0;
    while(SNVpos_file >> pos){
		if(SNV_count == 0) {reconstruction_start = atoi(pos.c_str());}  // first SNV position
		SNV_count++;
        SNV_pos.push_back(atoi(pos.c_str()) - reconstruction_start);  // offset by start of region
	}
	reconstruction_end = atoi(pos.c_str());;  // last SNV position

	clock_t start, end;
	double cpu_time_used;
	start = clock();

	// vector<vector<int> > Read_matrix;
	vector<vector<pair<int, int>>> Read_matrix; 
	int total_count = 0;
	int read_count = 0;
	char gap_quality = '*'; //'I'; 	 
	double mean_length = 0.;
	int gene_length  = reconstruction_end-reconstruction_start+1;
	// cout << reconstruction_start << " " << reconstruction_end << endl;
	// cout << "Gene length:" << gene_length << endl;

	int error_flag = 0;
	cout << "Parsing SAM file." << endl;
  	error_flag = parseSAM(FASTAreads[0], min_qual, gap_quality, mean_length, Read_matrix, SNV_pos,
				reconstruction_start, reconstruction_end, total_count, read_count, gene_length, zonename, nFrag);

	if(error_flag>0)
    		return 1;

	for (int i = 0; i< Read_matrix.size(); i++)
	{
	        if (Read_matrix[i].size() != gene_length)
			cout << "error!!!"<<endl;
	}

	int nReads = read_count;
	cout <<  endl << "After parsing " << total_count << " reads in file " <<FASTAreads[0]<< ", there are "<<nReads<< " reads(mean lengths "<< mean_length/nReads << ") covering regions "<< reconstruction_start << "-" << reconstruction_end <<"."<< endl;
	
	end = clock();	
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout << "CPU time for SAM parsing: "  << cpu_time_used << endl<<endl;

	start = clock();		
	vector<vector<pair<int, int>>> SNV_matrix; // sparse representation of nFrag x nSNV matrix
	int nSNV = SNV_pos.size();
	// vector<int> deleted_reads_list;	
	// int nFrag = SNV_matrix.size();	
	cout << "After calling SNVs from " << gene_length << " bases in regions between " << reconstruction_start << " and " << reconstruction_end << ", " << nSNV<< " SNVs are detected." << endl;
	cout << "After correcting error, "<< nFrag <<" fragments are used for haplotype assembly." << endl;
	// cout << "reduced number of fragment:" << deleted_reads_list.size() << endl;
	end = clock();	
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	cout << "CPU time for SNV calling: "  << cpu_time_used << endl << endl;
} 