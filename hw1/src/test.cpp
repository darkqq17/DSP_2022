#include "hmm.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[])
{
    HMM hmms[5];
    ofstream ofs;
    ifstream ifs;

    load_models( argv[1], hmms, 5);
    
    int T=50;
    int N=6;
    int x=0;
    int model_max = 0;
    int maxdelta_node = 0;
    double delta[N][T] = {0};
    double maxdelta = 0;
    double P[5] = {0};
    double max_P = 0;

    ifstream fin(argv[2]);
    string vocab;  
    while( getline(fin,vocab) )
    {
        for(int i=0; i<5; i++)
        { 
            P[i]=0; 
        }
        model_max = 0;
        max_P = 0;
        for(int model=0; model<5; model++)
        {
            for(int i=0; i<N; i++)
            {
                for(int j=0; j<T; j++)
                {
                    delta[i][j] = 0;
                }
            }
            /* Viterbi */
            // initialization
            for(int i=0; i<hmms[model].state_num; i++)
            {
                x=0;
                if (vocab[0] == 'A'){x = 0;}
                else if (vocab[0] == 'B'){x=1;}
                else if (vocab[0] == 'C'){x=2;}
                else if (vocab[0] =='D'){x=3;}
                else if (vocab[0] == 'E'){x=4;}
                else {x=5;}
                delta[i][0] = hmms[model].initial[i] * hmms[model].observation[x][i]; //delta第一行初始值
            }

            //Recursion
            for(int t=1; t<T; t++)
            {
                for(int j=0; j<hmms[model].state_num; j++)
                {
                    maxdelta = 0;
                    for(int i=0; i<hmms[model].state_num; i++)
                    {
                        if(maxdelta < delta[i][t-1] * hmms[model].transition[i][j])
                        {
                            maxdelta = delta[i][t-1] * hmms[model].transition[i][j];
                        }
                    }
                    x=0;
                    if (vocab[t] == 'A'){x = 0;}
                    else if (vocab[t] == 'B'){x=1;}
                    else if (vocab[t] == 'C'){x=2;}
                    else if (vocab[t] == 'D'){x=3;}
                    else if (vocab[t] == 'E'){x=4;}
                    else {x=5;}

                    delta[j][t] = maxdelta * hmms[model].observation[x][j];
                }
            }

            //Termination
            for(int i = 0; i < hmms[model].state_num; i++)
            {
                if (P[model] < delta[i][T - 1])
                {
                    P[model] = delta[i][T - 1];
                }
            }

            if (max_P < P[model])
            {
                max_P = P[model];
                model_max = model;
            }
            
        }
        ofs.open(argv[3], ios::app);
        if (!ofs.is_open()) 
        {
            cout << "Failed to open file.\n";
        } else 
        {
            ofs << hmms[model_max].model_name << " "  << max_P << endl;
        }
        ofs.close();

        ofs.open("accuracy.txt", ios::app);
        if (!ofs.is_open()) 
        {
            cout << "Failed to open file.\n";
        } else 
        {
            ofs << hmms[model_max].model_name << endl;
        }
        ofs.close();
    }

    char buffer[1000000] = {0};
    ifs.open("accuracy.txt");
    if (!ifs.is_open()) {
        cout << "Failed to open file.\n";
    } else {
        ifs.read(buffer, sizeof(buffer));
        ifs.close();
    }

    // counting accuracy
    char buffer_2[1000000] = {0};
    ifs.open("./data/test_lbl.txt");
    if (!ifs.is_open()) {
        cout << "Failed to open file.\n";
    } else {
        ifs.read(buffer_2, sizeof(buffer));
        ifs.close();
    }

    double count = 0;
    for(int i=7; i<32495 ;i+=13)
    {
        if(buffer[i] == buffer_2[i])
        {
            count++;
        }
    }
    cout << "Accuracy : "<<count/25 << "%" << endl;

	return 0;
}