#ifndef _SPARSE_AUTOENCODER_H
#define _SPARSE_AUTOENCODER_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include </mfs/user/zhaojunbo/Eigen/Dense>
#include <vector>
#include <string>
#include <cassert>
#include <map>

// global
Eigen::VectorXd g_rho;


// parameters initializer
std::map<std::string, Eigen::MatrixXd> initialize_parameter(int visible_size_lyr, int hidden_size_lyr);

// sigmoid activation
Eigen::VectorXd sigmoid(Eigen::VectorXd x);

// compute cost function
double compute_cost(const std::map<std::string, Eigen::MatrixXd> WgtBias, 
                const Eigen::MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                double lamb, double sparsity_param, double beta);

// back-propogation batch gradient compute
std::map<std::string, Eigen::MatrixXd> compute_batch_grad(const std::map<std::string, Eigen::MatrixXd> WgtBias, 
                const Eigen::MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, double lamb,
                double sparsity_param, double beta);

// back-propogation stochastic gradient compute
std::map<std::string, Eigen::MatrixXd> compute_stoc_grad(const std::map<std::string, Eigen::MatrixXd> WgtBias, 
                  const Eigen::MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, int index);

std::map<std::string, Eigen::MatrixXd> compute_mibt_stoc_grad(const std::map<std::string, Eigen::MatrixXd> WgtBias, 
                  const Eigen::MatrixXd& data, int hidden_size_lyr, int visible_size_lyr, 
                  double lamb, double sparsity_param, double beta, std::vector<int> index_data);


#endif

