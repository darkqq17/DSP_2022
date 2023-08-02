#include "hmm.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;

//Input : iterations, initial model(model_init), observation sequences(train_seq)
//Output : five models
int main(int argc, char *argv[])
{
    HMM hmm_initial;
    /* read train_seq data */
    ifstream ifs;

    char *p;
    int num;

    errno = 0;
    long conv = strtol(argv[1], &p, 10); //10 : base

    // Check for errors: e.g., the string does not represent an integer
    // or the integer is larger than int
    if (errno != 0 || *p != '\0' || conv > INT16_MAX || conv < INT16_MIN) {
        // Put here the handling of the error, like exiting the program with
        // an error message
    } else {
        // No error
        num = conv;
        num = num*1.6;
    }

    loadHMM( &hmm_initial, argv[2]);
	//dumpHMM( stderr, &hmm_initial );

    int x=0;
    int T=50;
    int N=6;
    double sum_initialize[N] = {0};
    double alpha[N][T] = {0};
    double beta[N][T] = {0};
    double gamma[N][T] = {0};
    double sum_transition[N][N] = {0};
    double sum_observation[N][N] = {0};
    double cache1[N][N] = {0};
    double cache2[N][N] = {0};
    double cache3[N][N] = {0};
    double cache4[N][N] = {0};
    double xi[T][N][N] = {0};
    double sum_alpha_init = 0;
    double sum_aplha = 0;
    double sum_beta = 0;
    double sum_gamma = 0;
    double sum_xi = 0;
    double numerator = 0;
    double denominator = 0;
    double numerator_2 = 0;
    double denominator_2 = 0;

    for(int iter=0; iter<num; iter++) //int iter=0; iter<100; iter++
    {
        //initialize transition/obervation of sum
        for(int i=0; i<6; i++)
        {
            sum_initialize[i] = 0;
        }
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
            {
                sum_transition[i][j] = 0;
                sum_observation[i][j] = 0;
            }
        }
        ifstream fin(argv[3]); 
        string vocab;  
        while( getline(fin,vocab) )
        {
            //int T = vocab.length();
            //cout << vocab.length() << endl;
            for(int i=0; i<N; i++)
            {
                for(int j=0; j<T; j++)
                {
                    alpha[i][j] = 0;
                    beta[i][j] = 0;
                    gamma[i][j] = 0;
                    for(int z=0; z<T; z++)
                    {
                        xi[i][j][z] = 0;
                    }
                }
            }
            /* forward */
            //initialization (ok)
            for(int i=0; i<hmm_initial.state_num; i++)
            {
                if (vocab[0] == 'A'){x=0;}
                else if (vocab[0] == 'B'){x=1;}
                else if (vocab[0] == 'C'){x=2;}
                else if (vocab[0] == 'D'){x=3;}
                else if (vocab[0] == 'E'){x=4;}
                else {x=5;}
                alpha[i][0] = hmm_initial.initial[i] * hmm_initial.observation[i][x]; //陣列O 資料怎麼取 (ok)
            }

            //Induction (ok)
            for(int t=0; t<T-1; t++) //T為何? 先設為50 (ok)
            {
                for(int j=0; j<hmm_initial.state_num; j++)
                {
                    sum_alpha_init = 0;
                    for(int i=0; i<hmm_initial.state_num; i++)
                    {
                        sum_alpha_init += alpha[i][t] * hmm_initial.transition[i][j];
                    }
                    if (vocab[t+1] == 'A'){x=0;}
                    else if (vocab[t+1] == 'B'){x=1;}
                    else if (vocab[t+1] == 'C'){x=2;}
                    else if (vocab[t+1] == 'D'){x=3;}
                    else if (vocab[t+1] == 'E'){x=4;}
                    else {x=5;}
                    alpha[j][t+1] = sum_alpha_init*hmm_initial.observation[j][x]; //bj(o(t+1))
                }
            }

            //testing alpha是否正確 (ok)
            /*for(int i=0; i<6; i++)
            {
                for(int j=0;j<T-1;j++)
                {
                    printf("%.5lf ", alpha[i][j]);
                }
                printf("%.5lf\n", alpha[i][49]);
            }*/

            //Termination . probability
            sum_aplha = 0;
            for(int i=0; i<hmm_initial.state_num; i++)
            {
                sum_aplha += alpha[i][T-1];
            }

            /* backward */
            //initialization
            for(int i=0; i<hmm_initial.state_num; i++)
            {
                beta[i][T-1] = 1; //initailize最後一行
            }

            //induction
            for(int t=T-2; t>=0; t--) //T-2為倒數第二行
            {
                for(int i=0; i<hmm_initial.state_num; i++)
                {
                    sum_beta = 0;
                    for(int j=0; j<hmm_initial.state_num; j++)
                    {
                        if (vocab[t+1] == 'A'){x=0;}
                        else if (vocab[t+1] == 'B'){x=1;}
                        else if (vocab[t+1] == 'C'){x=2;}
                        else if (vocab[t+1] == 'D'){x=3;}
                        else if (vocab[t+1] == 'E'){x=4;}
                        else {x=5;}
                        sum_beta += hmm_initial.transition[i][j] * hmm_initial.observation[j][x] * beta[j][t+1]; //changed -> hmm_initial.observation[j][x]
                    }
                    beta[i][t] = sum_beta;
                }
            }
            
            //testing beta是否正確 (ok)
            /*for(int i=0; i<6; i++)
            {
                for(int j=0;j<T-1;j++)
                {
                    printf("%.5f ", beta[i][j]);
                }
                printf("%.5f\n", beta[i][49]);
            }*/

            /* Gamma */
            for(int t=0; t<T; t++)
            {
                sum_gamma = 0;
                for(int i=0; i<hmm_initial.state_num; i++)
                {
                    sum_gamma += alpha[i][t] * beta[i][t];
                }
                for(int i=0; i<hmm_initial.state_num; i++)
                {
                    gamma[i][t] = (alpha[i][t] * beta[i][t]) / sum_gamma;
                }
            } 

            /* Xi */
            for(int t=0; t<T; t++)
            {
                sum_xi = 0;
                for(int i=0; i<hmm_initial.state_num; i++)
                {
                    for(int j=0; j<hmm_initial.state_num; j++)
                    {
                        if (vocab[t+1] == 'A'){x=0;}
                        else if (vocab[t+1] == 'B'){x=1;}
                        else if (vocab[t+1] == 'C'){x=2;}
                        else if (vocab[t+1] == 'D'){x=3;}
                        else if (vocab[t+1] == 'E'){x=4;}
                        else {x=5;}
                        xi[t][i][j] = alpha[i][t] * beta[j][t+1] * hmm_initial.transition[i][j] * hmm_initial.observation[j][x];
                        sum_xi += xi[t][i][j];
                    }
                }
                for(int i=0; i<hmm_initial.state_num; i++)
                {
                    for(int j=0; j<hmm_initial.state_num; j++)
                    {
                        xi[t][i][j] /= sum_xi;
                    }
                }
            }

            /* baumWelch */
            // update initial prob.
            for (int i = 0; i < hmm_initial.state_num; i++)
            {
                //hmm_initial.initial[i] = gamma[i][1];
                sum_initialize[i] += gamma[i][1];
            }
            
            // update transition prob. (ok)
            for (int i = 0; i < hmm_initial.state_num; i++)
            {
                for (int j = 0; j < hmm_initial.state_num; j++)
                {
                    numerator = 0;
                    denominator = 0;
                    for (int t = 0; t < T - 1; t++)
                    {
                        numerator += xi[t][i][j];
                        denominator += gamma[i][t];
                    }
                    //hmm_initial.transition[i][j] = numerator / denominator;
                    //sum_transition[i][j] += hmm_initial.transition[i][j];
                    cache1[i][j] += numerator;
                    cache2[i][j] += denominator;
                }
            }

            // update observation prob.
            for (int i = 0; i < hmm_initial.state_num; i++)
            {
                for (int k = 0; k < hmm_initial.observ_num; k++)
                {
                    numerator_2 = 0;
                    denominator_2 = 0;
                    for (int t = 0; t < T; t++)
                    {
                        if (vocab[t] == 'A'){x=0;}
                        else if (vocab[t] == 'B'){x=1;}
                        else if (vocab[t] == 'C'){x=2;}
                        else if (vocab[t] == 'D'){x=3;}
                        else if (vocab[t] == 'E'){x=4;}
                        else if (vocab[t] == 'F'){x=5;}
                        //printf("%d\n",x);
                        if (x == k)
                        {
                            numerator_2 += gamma[i][t];
                        }
                        denominator_2 += gamma[i][t];
                    }
                    //printf("%.5lf ", numerator);
                    //printf("%.5lf \n", denominator);
                    //hmm_initial.observation[k][i] = numerator_2 / denominator_2; //[k][i]or[i][k]?
                    //sum_observation[k][i] += hmm_initial.observation[k][i];
                    cache3[i][k] += numerator_2;
                    cache4[i][k] += denominator_2;

                }
            }
        }
        for(int i=0; i<hmm_initial.state_num; i++)
        {
            hmm_initial.initial[i] = sum_initialize[i] / 10000;
            for(int j=0; j<hmm_initial.state_num; j++)
            {
                hmm_initial.transition[i][j] = cache1[i][j] / cache2[i][j];
            }
        }
        for(int i=0; i<hmm_initial.state_num; i++)
        {
            for(int k=0; k<hmm_initial.observ_num; k++)
            {
                hmm_initial.observation[k][i] = cache3[i][k] / cache4[i][k];
            }
        }
    }


    ofstream ofs;

    ofs.open(argv[4]);
    if (!ofs.is_open()) {
        cout << "Failed to open file.\n";
    } else {
        ofs 
        << "initial:" << " " << hmm_initial.state_num << endl
        << hmm_initial.initial[0] << " " << hmm_initial.initial[1] << " " << hmm_initial.initial[2] << " " << hmm_initial.initial[3] << " " << hmm_initial.initial[4] << " " << hmm_initial.initial[5] << endl
        << endl
        << "transition:" << " " << hmm_initial.state_num << endl
        << hmm_initial.transition[0][0] << " "<< hmm_initial.transition[0][1] << " "<< hmm_initial.transition[0][2] << " "<< hmm_initial.transition[0][3] << " "  << hmm_initial.transition[0][4] << " " << hmm_initial.transition[0][5] << endl
        << hmm_initial.transition[1][0] << " "<< hmm_initial.transition[1][1] << " "<< hmm_initial.transition[1][2] << " "<< hmm_initial.transition[1][3] << " "  << hmm_initial.transition[1][4] << " " << hmm_initial.transition[1][5] << endl
        << hmm_initial.transition[2][0] << " "<< hmm_initial.transition[2][1] << " "<< hmm_initial.transition[2][2] << " "<< hmm_initial.transition[2][3] << " "  << hmm_initial.transition[2][4] << " " << hmm_initial.transition[2][5] << endl
        << hmm_initial.transition[3][0] << " "<< hmm_initial.transition[3][1] << " "<< hmm_initial.transition[3][2] << " "<< hmm_initial.transition[3][3] << " "  << hmm_initial.transition[3][4] << " " << hmm_initial.transition[3][5] << endl
        << hmm_initial.transition[4][0] << " "<< hmm_initial.transition[4][1] << " "<< hmm_initial.transition[4][2] << " "<< hmm_initial.transition[4][3] << " "  << hmm_initial.transition[4][4] << " " << hmm_initial.transition[4][5] << endl
        << hmm_initial.transition[5][0] << " "<< hmm_initial.transition[5][1] << " "<< hmm_initial.transition[5][2] << " "<< hmm_initial.transition[5][3] << " "  << hmm_initial.transition[5][4] << " " << hmm_initial.transition[5][5] << endl
        << endl
        << "observation:" << " " << hmm_initial.observ_num << endl
        << hmm_initial.observation[0][0] << " "<< hmm_initial.observation[0][1] << " "<< hmm_initial.observation[0][2] << " "<< hmm_initial.observation[0][3] << " "  << hmm_initial.observation[0][4] << " " << hmm_initial.observation[0][5] << endl
        << hmm_initial.observation[1][0] << " "<< hmm_initial.observation[1][1] << " "<< hmm_initial.observation[1][2] << " "<< hmm_initial.observation[1][3] << " "  << hmm_initial.observation[1][4] << " " << hmm_initial.observation[1][5] << endl
        << hmm_initial.observation[2][0] << " "<< hmm_initial.observation[2][1] << " "<< hmm_initial.observation[2][2] << " "<< hmm_initial.observation[2][3] << " "  << hmm_initial.observation[2][4] << " " << hmm_initial.observation[2][5] << endl
        << hmm_initial.observation[3][0] << " "<< hmm_initial.observation[3][1] << " "<< hmm_initial.observation[3][2] << " "<< hmm_initial.observation[3][3] << " "  << hmm_initial.observation[3][4] << " " << hmm_initial.observation[3][5] << endl
        << hmm_initial.observation[4][0] << " "<< hmm_initial.observation[4][1] << " "<< hmm_initial.observation[4][2] << " "<< hmm_initial.observation[4][3] << " "  << hmm_initial.observation[4][4] << " " << hmm_initial.observation[4][5] << endl
        << hmm_initial.observation[5][0] << " "<< hmm_initial.observation[5][1] << " "<< hmm_initial.observation[5][2] << " "<< hmm_initial.observation[5][3] << " "  << hmm_initial.observation[5][4] << " " << hmm_initial.observation[5][5] << endl;
        ofs.close();
    }

    return 0;
}