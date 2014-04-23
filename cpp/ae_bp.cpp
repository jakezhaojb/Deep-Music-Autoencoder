#include "ae.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>

// init
void autoencoder::ae_init(){
  assert(WgtBias.size() == 0);
  double r = sqrt(6);
  unordered_map<string, MatrixXd> InitWgtBias;
  for (int i = 0; i < n_lyr; i++) {
    MatrixXd W1 = (MatrixXd::Random(hidden_size[i], visible_size[i]).array() * 2 * r - r).matrix();
    MatrixXd W2 = (MatrixXd::Random(visible_size[i], hidden_size[i]).array() * 2 * r - r).matrix();
    VectorXd b1 = VectorXd::Random(hidden_size[i]);
    VectorXd b2 = VectorXd::Random(visible_size[i]);
    InitWgtBias["W1"] = W1;
    InitWgtBias["W2"] = W2;
    InitWgtBias["b1"] = b1;
    InitWgtBias["b2"] = b2;

    WgtBias.push_back(InitWgtBias);
  }
}

MatrixXd autoencoder::acti_fun(string fun_name, const MatrixXd & non_acti_data){
  if (fun_name == "sigmoid") {
    return ( (1.0 / (1 + exp(-x.array()))).matrix() ); 
  }
  else if (fun_name == "ReLU") {
    MatrixXd acti_data;
    for (int i = 0; i < non_acti_data.rows(); i++) {
      for (int j = 0; j < non_acti_data.cols(); j++) {
        acti_data(i, j) = max(non_acti_data(i, j), 0);
      }
    }
    return acti_data;
  }
  else if (fun_name == "tanh") { 
    ArrayXXd tmp = non_acti_data.array();
    return ( ((exp(tmp) - exp(-tmp)) / (exp(tmp) + exp(-tmp))).matrix() );
  }
  else{
    std::cerr << "The activation function is not implemented by far." << std::endl;
    exit(-1);
  }
  
}

// compute the cost of a single layer of NN
double autoencoder::ae_cost(int lyr) const {
  int i;
  double cost = 0;
  VectorXd sparse_kl;  // sparse penalty
  MatrixXd W1 = WgtBias[lyr].at("W1");
  MatrixXd W2 = WgtBias[lyr].at("W2");
  VectorXd b1 = WgtBias[lyr].at("b1");
  VectorXd b2 = WgtBias[lyr].at("b2");
  g_rho = VectorXd::Zero(b1.rows(), b1.cols());
  unordered_map<int, VectorXd> a;
  unordered_map<int, VectorXd> z;
  // traverse network
  for (i = 0; i < data.cols(); i++) {
    a[1] = data.col(i);
    z[2] = W1 * a[1] + b1;
    a[2] = acti_fun("sigmoid", z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun("sigmoid", z[3]);
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


// compute batch gradient
unordered_map<string, MatrixXd> autoencoder::ae_batch_grad(int lyr) const{
  MatrixXd W1 = WgtBias[lyr].at("W1");
  MatrixXd W2 = WgtBias[lyr].at("W2");
  VectorXd b1 = WgtBias[lyr].at("b1");
  VectorXd b2 = WgtBias[lyr].at("b2");

  MatrixXd W1_delta = MatrixXd::Zero(W1.rows(), W1.cols());
  MatrixXd W2_delta = MatrixXd::Zero(W2.rows(), W2.cols());
  VectorXd b1_delta = VectorXd::Zero(b1.size());
  VectorXd b2_delta = VectorXd::Zero(b2.size());

  // Use ArrayXd may be better! Take into consideration.
  unordered_map<int, VectorXd> a;
  unordered_map<int, VectorXd> z;
  unordered_map<int, VectorXd> sigma;
  for (int i = 0; i < data.cols(); i++) {
    a[1] = data.col(i);
    z[2] = W1 * a[1] + b1;
    a[2] = acti_fun("sigmoid", z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun("sigmoid", z[3]);
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
  unordered_map<string, MatrixXd> WgtBiasGrad;
  WgtBiasGrad["W1"] = W1_grad;
  WgtBiasGrad["W2"] = W2_grad;
  WgtBiasGrad["b1"] = b1_grad;
  WgtBiasGrad["b2"] = b2_grad;

  return WgtBiasGrad;
}


// compute the stochastic gradient
unordered_map<string, MatrixXd> autoencoder::ae_stoc_grad(int lyr, int index) const {
  unordered_map<string, MatrixXd> WgtBiasGrad;
  MatrixXd W1 = WgtBias[lyr].at("W1");
  MatrixXd W2 = WgtBias[lyr].at("W2");
  VectorXd b1 = WgtBias[lyr].at("b1");
  VectorXd b2 = WgtBias[lyr].at("b2");
  
  // means no mini-batch
  unordered_map<int, VectorXd> a;
  unordered_map<int, VectorXd> z;
  unordered_map<int, VectorXd> sigma;
  a[1] = data.col(index);
  a[2] = acti_fun("sigmoid", z[2]);
  VectorXd rho = a[2];  // Get rho first
  z[3] = W2 * a[2] + b2;
  a[3] = acti_fun("sigmoid", z[3]);
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

  return WgtBiasGrad;
}


// compute the mini-batch stochastic gradient
unordered_map<string, MatrixXd> autoencoder::ae_mibt_stoc_grad(int lyr, vector<int> index_data) const {

  size_t mini_batch_size = index_data.size();
  unordered_map<string, MatrixXd> WgtBiasGrad;
  MatrixXd W1 = WgtBias[lyr].at("W1");
  MatrixXd W2 = WgtBias[lyr].at("W2");
  VectorXd b1 = WgtBias[lyr].at("b1");
  VectorXd b2 = WgtBias[lyr].at("b2");
  
  if (!(mini_batch_size-1)) {
    // means no mini-batch
    unordered_map<int, VectorXd> a;
    unordered_map<int, VectorXd> z;
    unordered_map<int, VectorXd> sigma;
    a[1] = data.col(index_data[0]);
    a[2] = acti_fun("sigmoid", z[2]);
    VectorXd rho = a[2];  // Get rho first
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun("sigmoid", z[3]);
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
    MatrixXd W1_delta = MatrixXd::Zero(W1.rows(), W1.cols());
    MatrixXd W2_delta = MatrixXd::Zero(W2.rows(), W2.cols());
    VectorXd b1_delta = VectorXd::Zero(b1.size());
    VectorXd b2_delta = VectorXd::Zero(b2.size());
    unordered_map<int, VectorXd> a;
    unordered_map<int, VectorXd> z;
    unordered_map<int, VectorXd> sigma;

    // Get rho first
    VectorXd rho(b1.size());
    for (int i = 0; i < mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_fun("sigmoid", z[2]);
      rho += a[2];
    }
    rho = (rho.array() / mini_batch_size).matrix();

    // BP
    for (int i = 0; i < mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_fun("sigmoid", z[2]);
      z[3] = W2 * a[2] + b2;
      a[3] = acti_fun("sigmoid", z[3]);
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

  }  // else ends
  return WgtBiasGrad;
}


