#include <stdio.h>
#include "Ngram.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include<math.h>
using namespace std;

Vocab voc;
Ngram lm( voc, 2 );


// Get P(W2 | W1) -- bigram
double getBigramProb(const char *w1, const char *w2)
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid1, Vocab_None };
    return lm.wordProb( wid2, context);
}

int main(int argc, char *argv[]){
    if(argc<5){
        cout << "input data not enough: "<<argc << endl;
		return 0;
	}
    
    {
        const char* lm_filename[] = {argv[3]};
        File lmFile( *lm_filename, "r" );
        lm.read(lmFile);
        lmFile.close();
    }
    // Ū��
    ifstream ex_file(argv[1]);
    ifstream map_file(argv[2]);
    ofstream outfile(argv[4]);

    // �Nmap����key��value�s�bvector���ΨӰ����᪺����
    // map_key = �C�ӥi��X�{����r�Ϊ`�� (ex:�t �u �v...)
    // map_values = map_key�����쪺�Ҧ��i���r(ex:[�K, �P, �R...], [��, ��, ��...])
    string map_line = "";
    vector<string> map_key;
    vector< vector<string> > map_values;
    while(getline(map_file, map_line)){
        map_key.push_back(map_line.substr(0,2));

        vector<string> map_value;
        for (int i = 2; i < map_line.length(); i++) {
            if (map_line[i] == ' ' || map_line[i] == '\t') continue;
            map_value.push_back(map_line.substr(i++,2));
        }
        // map_value.pop_back();
        map_values.push_back(map_value);
    }
    //=================================
    string ex_line = "";
    while(getline(ex_file, ex_line)){
        // example�C��r���ର�x�}
        vector<string> word_vec;
        for (int i = 0; i < ex_line.length(); i++) { 
            if (ex_line[i] == ' ') continue;
            word_vec.push_back(ex_line.substr(i++,2));
        }

        // ��bexample���C�Ӧr�i�Hmap�쪺����r
        // all_word_maps = �Ӧr�ꤤ�C�Ӧr�����쪺map_value (ex:�Y�ĤG�Ӧr�O�t �hall_word_maps[1]=[�K, �P, �R...])
        vector< vector<string> > all_word_maps;
        for(int i=0; i<word_vec.size(); i++){
            for(int j=0; j<map_key.size(); j++){
                if(map_key[j].compare(word_vec[i])==0){            // ���Ӧr������map_key
                    all_word_maps.push_back(map_values[j]);
                    break;
                }
            }
        }
        vector<string> end;
        end.push_back("</s>");
        all_word_maps.push_back(end);
        
        // Viterbi
        // �ʺA�إ߯x�}
        int row = all_word_maps.size();
        int col = 0;                            //�o�̧ڭ̨������r�̦h���`����@col����
        for(int i=0; i<all_word_maps.size(); i++){
            if(all_word_maps[i].size()>col)
                col = all_word_maps[i].size();
        }

        double **prob;                  // �o�ӥΨӰO�����v
        prob=new double *[row];
        for(int i=0;i<row;i++){
            prob[i]=new double [col];
            memset(prob[i],0,col*sizeof(double));
        }
        int **last_pos;                 //�o�ӥΨӰO���W�Ӧr����m
        last_pos=new int *[row];
        for(int i=0;i<row;i++){
            last_pos[i]=new int [col];
            memset(last_pos[i],0,col*sizeof(int));
        }
        
        //=======�����}�l��==========
        for(int j=0; j<all_word_maps[0].size(); j++){
            prob[0][j] = getBigramProb("<s>", all_word_maps[0][j].c_str());
        }

        for(int k=1; k<row; k++){                                       // ���X�@�椤���C�Ӧr
            for(int i=0; i<all_word_maps[k].size(); i++){             // ���X�o�Ӧr���Ҧ��i��
                prob[k][i] = prob[k-1][0] + getBigramProb(all_word_maps[k-1][0].c_str(), all_word_maps[k][i].c_str());
                last_pos[k][i] = 0;
                for(int j=1; j<all_word_maps[k-1].size(); j++){           // ���X�W�Ӧr���Ҧ��i��
                    double temp_p = prob[k-1][j] + getBigramProb(all_word_maps[k-1][j].c_str(), all_word_maps[k][i].c_str());
                    if(temp_p > prob[k][i]){
                        prob[k][i] = temp_p;
                        last_pos[k][i] = j;
                    }
                }
            }
        }

        // backtrack
        stack<int> result_st;
        int last = 0;
        for(int i=row-1; i>0; i--){
            result_st.push(last_pos[i][last]);
            last = last_pos[i][last];
        }
        
        // �g�J��r��
        int i=0;
        outfile << "<s> ";
        while(!result_st.empty()){
            outfile << all_word_maps[i++][result_st.top()] << " " ;
            result_st.pop();
        }
        outfile << "</s>" <<endl;
        //==========================
        // �M�P�x�}
        for (int i = 0; i < row; i ++) {
            delete[] prob[i];
            prob[i] = NULL;
        }
        delete [] prob;
        prob=NULL;

        for (int i = 0; i < row; i ++) {
            delete[] last_pos[i];
            last_pos[i] = NULL;
        }
        delete [] last_pos;
        last_pos=NULL;
    }

    ex_file.close();
    map_file.close();
    outfile.close();
    return 0;
}