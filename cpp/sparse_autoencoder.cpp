#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <assert.h>
#include <map>

using std::vector;
using std::string;
using namespace Eigen;

// global
VectorXd g_rho;

// Initialize weights and bias in a key-value struct
std::map<string, MatrixXd> initialize_parameter(int visible_size, int hidden_size){
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


// sigmoid func
VectorXd sigmoid(VectorXd x){
  VectorXd sigm;
  sigm = (1.0 / (1 + exp(-x.array()))).matrix();
  return sigm;
}


// compute the cost of AE reconstruction, and combine the weight decay and sparse penalty
double compute_cost(std::map<string, MatrixXd> WgtBias, MatrixXd& data, 
                    int hidden_size, int visible_size, double lamb,
                    double sparsity_param, double beta){
  //MatrixXd W1, W2;
  //VectorXd b1, b2;
  int i;
  double cost = 0;
  VectorXd sparse_kl;  // sparse penalty
  MatrixXd W1 = WgtBias["W1"];
  MatrixXd W2 = WgtBias["W2"];
  VectorXd b1 = WgtBias["b1"];
  VectorXd b2 = WgtBias["b2"];
  g_rho = VectorXd::Zero(b1.rows(), b1.cols());

  std::map<int, VectorXd> a;
  std::map<int, VectorXd> z;
  for (i = 0; i < data.cols(); i++) {
    a[1] = data.col(i);
    z[2] = W1 * a[1] + b1;
    a[2] = sigmoid(z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = sigmoid(z[3]);
    cost += ((a[1]-a[3]).array().pow(2)/2).sum();
    g_rho += a[2];
  }
  // rho post-process
  g_rho = (g_rho.array() / data.cols()).matrix();

  // cost post-process
  sparse_kl = sparsity_param * log(sparsity_param/g_rho.array()) +\
              (1-sparsity_param) * log((1-sparsity_param)/(1-g_rho.array()));
  cost /= data.cols();
  cost += lamb/2 * (W1.array().pow(2).sum() + W2.array().pow(2).sum()) +\
          beta*sparse_kl.sum();

  return cost;
}


// compute the batch-gradient of the cost over each weight elem and bias elem
std::map<string, MatrixXd> compute_batch_grad(std::map<string, MatrixXd> WgtBias, MatrixXd& data, 
                                              int hidden_size, int visible_size, double lamb,
                                              double sparsity_param, double beta){
  int i;
  MatrixXd W1 = WgtBias["W1"];
  MatrixXd W2 = WgtBias["W2"];
  VectorXd b1 = WgtBias["b1"];
  VectorXd b2 = WgtBias["b2"];

  MatrixXd W1_delta(W1.rows(), W1.cols());
  MatrixXd W2_delta(W2.rows(), W2.cols());
  VectorXd b1_delta(b1.size());
  VectorXd b2_delta(b2.size());

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
    sigma[3] = (-(a[1]-a[3]).array() * (a[3].array()*(1-a[3].array()))).matrix();
    VectorXd sparsity_sigma = -sparsity_param/g_rho.array() +\
                          (1-sparsity_param)*(1-g_rho.array());
    sigma[2] = (((W2.transpose()*sigma[3]).array() + beta*sparsity_sigma.array())*\
               a[2].array()*(1-a[2].array())).matrix();

    W1_delta += sigma[2] * a[1].transpose();
    W2_delta += sigma[3] * a[2].transpose();
    b1_delta += sigma[2];
    b2_delta += sigma[3];
  }

  MatrixXd W1_grad = (W1_delta.array() / data.cols() + lamb * W1.array()).matrix();
  MatrixXd W2_grad = (W2_delta.array() / data.cols() + lamb * W2.array()).matrix();
  VectorXd b1_grad = (b1_delta.array() / data.cols()).matrix();
  VectorXd b2_grad = (b2_delta.array() / data.cols()).matrix();

  // return the gradients
  std::map<string, MatrixXd> WgtBiasGrad;
  WgtBiasGrad["W1"] = W1_grad;
  WgtBiasGrad["W2"] = W2_grad;
  WgtBiasGrad["b1"] = b1_grad;
  WgtBiasGrad["b2"] = b2_grad;

  return WgtBiasGrad;
}


// compute the stochastic gradient of the cost over each weight elem and bias elem
// mini-batch is involved. If you don't need the mini-batch, simply assign a 1-D std::vector<int> into index_data
std::map<string, MatrixXd> compute_stoc_grad(std::map<string, MatrixXd> WgtBias, MatrixXd& data,
                                             int hidden_size, int visible_size, double lamb, double sparsity_param, 
                                             double beta, vector<int> index_data){
  size_t mini_batch_size = index_data.size();
  int i;
  std::map<string, MatrixXd> WgtBiasGrad;
  MatrixXd W1 = WgtBias["W1"];
  MatrixXd W2 = WgtBias["W2"];
  VectorXd b1 = WgtBias["b1"];
  VectorXd b2 = WgtBias["b2"];
  
  if (!(mini_batch_size-1)) {
    // means no mini-batch
    std::map<int, VectorXd> a;
    std::map<int, VectorXd> z;
    std::map<int, VectorXd> sigma;
    a[1] = data.col(index_data[0]);
    a[2] = sigmoid(z[2]);
    VectorXd rho = a[2];  // Get rho first
    z[3] = W2 * a[2] + b2;
    a[3] = sigmoid(z[3]);
    sigma[3] = (-(a[1]-a[3]).array() * (a[3].array()*(1-a[3].array()))).matrix();
    VectorXd sparsity_sigma = -sparsity_param/g_rho.array() +\
                          (1-sparsity_param)*(1-g_rho.array());
    sigma[2] = (((W2.transpose()*sigma[3]).array() + beta*sparsity_sigma.array())*\
               a[2].array()*(1-a[2].array())).matrix();
    // gradient of that sample
    MatrixXd W1_grad = sigma[2] * a[1].transpose();  
    MatrixXd W2_grad = sigma[3] * a[2].transpose();  
    VectorXd b1_grad = sigma[2];  
    VectorXd b2_grad = sigma[3];  

    WgtBiasGrad["W1"] = W1_grad;
    WgtBiasGrad["W2"] = W2_grad;
    WgtBiasGrad["b1"] = b1_grad;
    WgtBiasGrad["b2"] = b2_grad;
  }else{
    // Got a mini-batch SGD
    //MatrixXd W1_delta = MatrixXd::Zeros(W1.rows(), W1.cols());
    //MatrixXd W2_delta = MatrixXd::Zeros(W2.rows(), W2.cols());
    //VectorXd b1_delta = VectorXd::Zeros(b1.size());
    //VectorXd b2_delta = VectorXd::Zeros(b2.size());
    MatrixXd W1_delta(W1.rows(), W1.cols());
    MatrixXd W2_delta(W2.rows(), W2.cols());
    VectorXd b1_delta(b1.size());
    VectorXd b2_delta(b2.size());

    std::map<int, VectorXd> a;
    std::map<int, VectorXd> z;
    std::map<int, VectorXd> sigma;

    // Get rho first
    VectorXd rho(b1.size());
    for (i = 0; i < mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = sigmoid(z[2]);
      rho += a[2];
    }
    rho = (rho.array() / mini_batch_size).matrix();

    // BP
    for (i = 0; i < mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = sigmoid(z[2]);
      z[3] = W2 * a[2] + b2;
      a[3] = sigmoid(z[3]);
      sigma[3] = (-(a[1]-a[3]).array() * (a[3].array()*(1-a[3].array()))).matrix();
      VectorXd sparsity_sigma = -sparsity_param/g_rho.array() +\
                            (1-sparsity_param)*(1-g_rho.array());
      sigma[2] = (((W2.transpose()*sigma[3]).array() + beta*sparsity_sigma.array())*\
                 a[2].array()*(1-a[2].array())).matrix();

      W1_delta += sigma[2] * a[1].transpose();
      W2_delta += sigma[3] * a[2].transpose();
      b1_delta += sigma[2];
      b2_delta += sigma[3];
    }

    MatrixXd W1_grad = (W1_delta.array() / mini_batch_size + lamb * W1.array()).matrix();
    MatrixXd W2_grad = (W2_delta.array() / mini_batch_size + lamb * W2.array()).matrix();
    VectorXd b1_grad = (b1_delta.array() / mini_batch_size).matrix();
    VectorXd b2_grad = (b2_delta.array() / mini_batch_size).matrix();

    // return the gradients
    WgtBiasGrad["W1"] = W1_grad;
    WgtBiasGrad["W2"] = W2_grad;
    WgtBiasGrad["b1"] = b1_grad;
    WgtBiasGrad["b2"] = b2_grad;

  }
  return WgtBiasGrad;
}

int main(int argc, const char *argv[])
{
  auto wb = initialize_parameter(64, 25);
  //std::cout << wb["W1"] << std::endl;
  //std::cout << wb["W2"] << std::endl;
  //std::cout << wb["b1"] << std::endl;
  //std::cout << wb["b2"] << std::endl;

  auto sigm = sigmoid(VectorXd::Random(20));
  MatrixXd data = MatrixXd::Random(64, 100);

  double cost = compute_cost(wb, data, 25, 64, 0.0001, 0.01, 3);
  std::cout << cost << std::endl;

  // BGD testing
  auto bgd = compute_batch_grad(wb, data, 25, 64, 0.0001, 0.01, 3);
  //std::cout << bgd["W1"] << std::endl;
  //std::cout << bgd["W2"] << std::endl;
  //std::cout << bgd["b1"] << std::endl;
  //std::cout << bgd["b2"] << std::endl;

  // SGD and mini-batch testing
  vector<int> in;
  for (int i = 0; i < data.cols(); i++) {
    in.push_back(i);
  }
  std::random_shuffle(in.begin(), in.end());
  in.erase(in.begin()+10, in.end());
  auto sgd = compute_stoc_grad(wb, data, 25, 64, 0.0001, 0.01, 3, in);
  //std::cout << sgd["W1"] << std::endl;
  //std::cout << sgd["W2"] << std::endl;
  //std::cout << sgd["b1"] << std::endl;
  //std::cout << sgd["b2"] << std::endl;
  //std::cout << "Mean: " << std::endl;
  //std::cout << (sgd["W1"].mean()+sgd["W2"].mean()+ sgd["b1"].mean()+ sgd["b2"].mean())/4.0f << std::endl;
  return 0;
}
