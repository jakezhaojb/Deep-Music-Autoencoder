#ifndef _SPARSE_AUTOENCODER_H
#define _SPARSE_AUTOENCODER_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <cassert>
#include <unordered_map>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// global
VectorXd g_rho;


// parameters initializer
unordered_map<string, MatrixXd> initialize_parameter(const int visible_size_lyr, const int hidden_size_lyr);

// sigmoid activation
VectorXd sigmoid(const VectorXd x);

// compute cost function
double compute_cost(const unordered_map<string, MatrixXd> WgtBias, 
                const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                double lamb, double sparsity_param, double beta);

// back-propogation batch gradient compute
unordered_map<string, MatrixXd> compute_batch_grad(const unordered_map<string, MatrixXd> WgtBias, 
                const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, double lamb,
                double sparsity_param, double beta);

// back-propogation stochastic gradient compute
unordered_map<string, MatrixXd> compute_stoc_grad(const unordered_map<string, MatrixXd> WgtBias, 
                  const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, int index);

unordered_map<string, MatrixXd> compute_mibt_stoc_grad(const unordered_map<string, MatrixXd> WgtBias, 
                  const MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, vector<int> index_data);


#endif

