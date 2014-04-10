#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <string>

using std::vector;
using std::string;
using namespace Eigen;

// global
VectorXd g_rho;

std::map<string, MatrixXd> initialize_parameter(int hidden_size, int visible_size){
  // According to Ng's initialization
  double r = sqrt(6);
  std::map<string, MatrixXd> InitWgtBias;

  MatrixXd W1 = (MatrixXd::Random(hidden_size, visible_size).array() * 2 * r - r).matrix();
  MatrixXd W2 = (MatrixXd::Random(visible_size, hidden_size).array() * 2 * r - r).matrix();
  VectorXd b1 = VectorXd::Random(hidden_size);
  VectorXd b2 = VectorXd::Random(visible_size);

  InitWgtBias["W1"] = W1;
  InitWgtBias["W2"] = W2;
  InitWgtBias["b1"] = b1;
  InitWgtBias["b2"] = b2;

  return InitWgtBias;
}


VectorXd sigmoid(VectorXd x){
  VectorXd sigm;
  sigm = (1.0 / (1 + exp(-x.array()))).matrix();
  return sigm;
}


double compute_cost(std::map<string, MatrixXd> WgtBias, MatrixXd data, 
                    int hidden_size, int visible_size, double lamb,
                    double sparsity_param, double beta){
  //MatrixXd W1, W2;
  //VectorXd b1, b2;
  int i, j;
  double cost;
  VectorXd sparse_kl;  // sparse penalty
  auto W1 = WgtBias["W1"];
  auto W2 = WgtBias["W2"];
  auto b1 = WgtBias["b1"];
  auto b2 = WgtBias["b2"];
  g_rho = VectorXd::Zero(b1.rows(), b1.cols());

  std::map<int, VectorXd> a;
  std::map<int, VectorXd> z;
  for (i = 0; i < data.cols(); i++) {
    a[1] = data.col(i);
    z[2] = W1 * a[1] + b1;
    a[2] = sigmoid(z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = sigmoid(z[3]);
    cost += ((a[1]-a[3]).array().pow(2)).sum();
    g_rho += a[2];
  }
  // rho post-process
  g_rho = (g_rho.array() / data.cols()).matrix();

  // cost post-process
  sparse_kl = sparsity_param * log(sparsity_param/g_rho.array()) +\
              (1-sparsity_param) * log((1-sparsity_param)/(1-g_rho.array()));
  cost /= data.cols();
  cost += W1.array().pow(2).sum() + W2.array().pow(2).sum() +\
          beta*sparse_kl.sum();

  return cost;
}


std::map<string, MatrixXd> compute_batch_grad(std::map<string, MatrixXd> WgtBias, int hidden_size, 
                    int visible_size, double lamb,
                    double sparsity_param, double beta){
  int i, j;
  auto W1 = WgtBias["W1"];
  auto W2 = WgtBias["W2"];
  auto b1 = WgtBias["b1"];
  auto b2 = WgtBias["b2"];

  MatrixXd W1_delta = MatrixXd::Zeros(W1.rows(), W1.cols());
  MatrixXd W2_delta = MatrixXd::Zeros(W2.rows(), W2.cols());
  VectorXd b1_delta = VectorXd::Zeros(b1.size());
  VectorXd b2_delta = VectorXd::Zeros(b2.size());

  // Use ArrayXd may be better! Take into consideration.
  std::map<int, VectorXd> a;
  std::map<int, VectorXd> z;
  std::map<int, VectorXd> sigma;
  for (i = 0; i < data.cols(); i++) {
    a[1] = data.col(i);
    z[2] = W1 * a[1] + b1;
    a[2] = sigmoid(z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = sigmoid(z[3]);
    sigma[3] = -(a[1]-a[3]).array() * (a[3].array()*(1-a[3].array()));
    auto sparsity_sigma = -sparsity_param/g_rho.array() +\
                          (1-sparsity_param)*(1-g_rho.array());
    sigma[2] = ((W2*sigma[3]).array() + beta*sparsity_sigma.array())*\
               a[2].array()*(1-a[2].array());

    W1_delta += sigma[2] * a[1].transpose();
    W2_delta += sigma[3] * a[2].transpose();
    b1_delta += sigma[2];
    b2_delta += sigma[3];
  }

  auto W1_grad = W1_delta.array() / data.cols() + lamb * W1.array();
  auto W2_grad = W2_delta.array() / data.cols() + lamb * W2.array();
  auto b1_grad = b1_delta.array() / data.cols();
  auto b2_grad = b2_delta.array() / data.cols();

  W1_grad = W1_grad.matrix();
  W2_grad = W2_grad.matrix();
  b1_grad = b1_grad.matrix();
  b2_grad = b2_grad.matrix();

  // return the gradients
  std::map<string, MatrixXd> WgtBiasGrad;
  WgtBiasGrad["W1"] = W1_grad;
  WgtBiasGrad["W2"] = W2_grad;
  WgtBiasGrad["b1"] = b1_grad;
  WgtBiasGrad["b2"] = b2_grad;

  return WgtBiasGrad;
}
