#ifndef _SPARSE_AUTOENCODER_H
#define _SPARSE_AUTOENCODER_H

#include <iostream>
#include <math.h>
#include <algorithm>
#include </mfs/user/zhaojunbo/Eigen/Dense>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

using std::vector;
using std::string;
using namespace Eigen;

// global
VectorXd g_rho;


// parameters initializer
std::map<string, MatrixXd> initialize_parameter(int visible_size_lyr, int hidden_size_lyr);

// sigmoid activation
VectorXd sigmoid(VectorXd x);

// compute cost function
double compute_cost(const std::map<string, MatrixXd> WgtBias, 
                const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                double lamb, double sparsity_param, double beta);

// back-propogation batch gradient compute
std::map<string, MatrixXd> compute_batch_grad(const std::map<string, MatrixXd> WgtBias, 
                const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, double lamb,
                double sparsity_param, double beta);

// back-propogation stochastic gradient compute
std::map<string, MatrixXd> compute_stoc_grad(const std::map<string, MatrixXd> WgtBias, 
                  const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, int index);

std::map<string, MatrixXd> compute_mibt_stoc_grad(const std::map<string, MatrixXd> WgtBias, 
                  const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, vector<int> index_data);


#endif

