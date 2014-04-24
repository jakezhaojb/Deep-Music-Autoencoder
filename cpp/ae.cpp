#include <algorithm>
#include <iostream>
#include <fstream>
#include "ae.hpp"
#include "ps.hpp"
#include "utils.hpp"

namespace paracel{

// construction function
autoencoder::autoencoder(paracel::Comm comm, string hosts_dct_str,
          vector<string> _input, string output, vector<int> _hidden_size,
          int _visible_size, string method, int _rounds, 
          double _alpha, bool _debug, int limit_s, bool ssp_switch, 
          double _lamb, double _sparsity_param, double _beta, int _mibt_size) :
  paracel::paralg(hosts_dct_str, comm, output, _rounds, limit_s, ssp_switch),
  worker_id(comm.get_rank()),
  input(_input),  // AVAILBLE???
  hidden_size(_hidden_size);  // AVAILBLE???
  visible_size(_visible_size);
  learning_method(method),
  rounds(_rounds),
  alpha(_alpha),
  debug(_debug),
  lamb(_lamb),
  sparsity_param(_sparsity_param),
  beta(_beta),
  mibt_size(_mibt_size) {
    //hidden_size.assign(_hidden_size.begin(), _hidden_size.end());
    n_lyr = hidden_size.size();  // number of hidden layers
    layer_size.assign(hidden_size.begin(), hidden_size.end());
    layer_size.insert(layer_size.begin(), visible_size);
    ae_init();
  }


autoencoder::~autoencoder() {}


// init
void autoencoder::ae_init(){
  assert(WgtBias.size() == 0);
  double r = sqrt(6);
  unordered_map<string, MatrixXd> InitWgtBias;
  for (int i = 0; i < n_lyr; i++) {
    MatrixXd W1 = (MatrixXd::Random(layer_size[i+1], layer_size[i]).array() * 2 * r - r).matrix();
    MatrixXd W2 = (MatrixXd::Random(layer_size[i], layer_size[i+1]).array() * 2 * r - r).matrix();
    VectorXd b1 = VectorXd::Random(layer_size[i+1]);
    VectorXd b2 = VectorXd::Random(layer_size[i]);
    InitWgtBias["W1"] = W1;
    InitWgtBias["W2"] = W2;
    InitWgtBias["b1"] = b1;
    InitWgtBias["b2"] = b2;

    WgtBias.push_back(InitWgtBias);
  }
}

MatrixXd autoencoder::acti_fun(const MatrixXd & non_acti_data, string fun_name ) const {
  if (fun_name == "sigmoid") {
    return ( (1.0 / (1 + exp(-non_acti_data.array()))).matrix() ); 
  }
  else if (fun_name == "ReLU") {
    MatrixXd acti_data;
    for (int i = 0; i < non_acti_data.rows(); i++) {
      for (int j = 0; j < non_acti_data.cols(); j++) {
        acti_data(i, j) = max(non_acti_data(i, j), 0.);
      }
    }
    return acti_data;
  }
  else if (fun_name == "tanh") { 
    Eigen::ArrayXXd tmp = non_acti_data.array();
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
    a[2] = acti_fun(z[2], "sigmoid");
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun(z[3], "sigmoid");
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
    a[2] = acti_fun(z[2], "sigmoid");
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun(z[3], "sigmoid");
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
  a[2] = acti_fun(z[2], "sigmoid");
  VectorXd rho = a[2];  // Get rho first
  z[3] = W2 * a[2] + b2;
  a[3] = acti_fun(z[3], "sigmoid");
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
    a[2] = acti_fun(z[2], "sigmoid");
    VectorXd rho = a[2];  // Get rho first
    z[3] = W2 * a[2] + b2;
    a[3] = acti_fun(z[3], "sigmoid");
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
    for (int i = 0; i < (int)mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_fun(z[2], "sigmoid");
      rho += a[2];
    }
    rho = (rho.array() / mini_batch_size).matrix();

    // BP
    for (int i = 0; i < (int)mini_batch_size; i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_fun(z[2], "sigmoid");
      z[3] = W2 * a[2] + b2;
      a[3] = acti_fun(z[3], "sigmoid");
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

// distributed bgd
void autoencoder::distribute_bgd(int lyr){
  unordered_map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  _paracel_write("W1", WgtBias_lyr.at("W1"));
  _paracel_write("W2", WgtBias_lyr.at("W2"));
  _paracel_write("b1", WgtBias_lyr.at("b1"));
  _paracel_write("b2", WgtBias_lyr.at("b2"));
  paracel_register_bupdate("./update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());
  for (int rd = 0; rd < rounds; rd++) {
    WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
    WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
    WgtBias_lyr.at("b1") = _paracel_read("b1");
    WgtBias_lyr.at("b2") = _paracel_read("b2");
    delta = ae_batch_grad(lyr);
    delta.at("W1") *= alpha;
    delta.at("W2") *= alpha;
    delta.at("b1") *= alpha;
    delta.at("b2") *= alpha;
    if (debug) {
      loss_error.push_back(ae_cost(lyr));
    }
    // push
    _paracel_bupdate("W1", delta.at("W1"));
    _paracel_bupdate("W2", delta.at("W2"));
    _paracel_bupdate("b1", delta.at("b1"));
    _paracel_bupdate("b2", delta.at("b2"));
    iter_commit();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  } // rounds
  // last pull
  WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
  WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
  WgtBias_lyr.at("b1") = _paracel_read("b1");
  WgtBias_lyr.at("b2") = _paracel_read("b2");
}


// downpour sgd
void autoencoder::downpour_sgd(int lyr){
  int data_sz = data.cols();
  int cnt = 0, read_batch = data_sz/ 1000, update_batch = data_sz / 100;
  assert( (lyr > 0 && lyr < n_lyr) && "Input layer not qualified!");
  if (read_batch == 0) { read_batch = 10; }
  if (update_batch == 0) { update_batch = 10; }
  // Reference operator
  unordered_map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  _paracel_write("W1", WgtBias_lyr.at("W1"));
  _paracel_write("W2", WgtBias_lyr.at("W2"));
  _paracel_write("b1", WgtBias_lyr.at("b1"));
  _paracel_write("b2", WgtBias_lyr.at("b2"));
  vector<int> idx;
  for (int i = 0; i < data.cols(); i++) {
    idx.push_back(i);
  }
  paracel_register_bupdate("./update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());

  for (int rd = 0; rd < rounds; rd++) {
    std::random_shuffle(idx.begin(), idx.end());

    // init read
    WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
    WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
    WgtBias_lyr.at("b1") = _paracel_read("b1");
    WgtBias_lyr.at("b2") = _paracel_read("b2");
    unordered_map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);

    // traverse data
    cnt = 0;
    for (auto sample_id : idx) {
      if ( (cnt % read_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
        WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
        WgtBias_lyr.at("b1") = _paracel_read("b1");
        WgtBias_lyr.at("b2") = _paracel_read("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      unordered_map<string, MatrixXd> WgtBias_grad = ae_stoc_grad(lyr, sample_id);
      WgtBias_lyr.at("W1") += alpha * WgtBias_grad["W1"];
      WgtBias_lyr.at("W2") += alpha * WgtBias_grad["W2"];
      WgtBias_lyr.at("b1") += alpha * WgtBias_grad["b1"];
      WgtBias_lyr.at("b2") += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
      }
      if ( (cnt % update_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        delta.at("W1") = WgtBias_lyr.at("W1") - WgtBias_lyr_old.at("W1");
        delta.at("W2") = WgtBias_lyr.at("W2") - WgtBias_lyr_old.at("W2");
        delta.at("b1") = WgtBias_lyr.at("b1") - WgtBias_lyr_old.at("b1");
        delta.at("b2") = WgtBias_lyr.at("b2") - WgtBias_lyr_old.at("b2");
        // push
        _paracel_bupdate("W1", delta.at("W1"));
        _paracel_bupdate("W2", delta.at("W2"));
        _paracel_bupdate("b1", delta.at("b1"));
        _paracel_bupdate("b2", delta.at("b2"));
      }
      cnt += 1;
    } // traverse
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
  WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
  WgtBias_lyr.at("b1") = _paracel_read("b1");
  WgtBias_lyr.at("b2") = _paracel_read("b2");
}


// mini-batch downpour sgd
void autoencoder::downpour_sgd_mibt(int lyr){
  int data_sz = data.cols();
  int mibt_cnt = 0, read_batch = data_sz / (mibt_size*100), update_batch = data_sz / (mibt_size*100);
  assert( (lyr > 0 && lyr < n_lyr) && "Input layer not qualified!");
  if (read_batch == 0) { read_batch = 10; }
  if (update_batch == 0) { update_batch = 10; }
  // Reference operator
  unordered_map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  _paracel_write("W1", WgtBias_lyr.at("W1"));
  _paracel_write("W2", WgtBias_lyr.at("W2"));
  _paracel_write("b1", WgtBias_lyr.at("b1"));
  _paracel_write("b2", WgtBias_lyr.at("b2"));
  vector<int> idx;
  for (int i = 0; i < data.cols(); i++) {
    idx.push_back(i);
  }
  // ABSOULTE PATH
  paracel_register_bupdate("./update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());

  for (int rd = 0; rd < rounds; rd++) {
    std::random_shuffle(idx.begin(), idx.end());
    vector<vector<int>> mibt_idx; // mini-batch id
    for (auto i = idx.begin(); ; i += mibt_size) {
      if (idx.end() - i < mibt_size) {
        if (idx.end() == i) {
          break;
        }else{
          vector<int> tmp;
          tmp.assign(i, idx.end());
          mibt_idx.push_back(tmp);
        }
      }
      vector<int> tmp;
      tmp.assign(i, i + mibt_size);
      // SUPPOSE IT TO BE NOT ACCUMULATED OVER WORKERS?
      mibt_idx.push_back(tmp);
    }
    // init push
    WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
    WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
    WgtBias_lyr.at("b1") = _paracel_read("b1");
    WgtBias_lyr.at("b2") = _paracel_read("b2");
    unordered_map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);
    
    // traverse data
    mibt_cnt = 0;
    for (auto mibt_sample_id : mibt_idx) {
      if ( (mibt_cnt % read_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
        WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
        WgtBias_lyr.at("b1") = _paracel_read("b1");
        WgtBias_lyr.at("b2") = _paracel_read("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      unordered_map<string, MatrixXd> WgtBias_grad = ae_mibt_stoc_grad(lyr, mibt_sample_id);
      WgtBias_lyr.at("W1") += alpha * WgtBias_grad["W1"];
      WgtBias_lyr.at("W2") += alpha * WgtBias_grad["W2"];
      WgtBias_lyr.at("b1") += alpha * WgtBias_grad["b1"];
      WgtBias_lyr.at("b2") += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
      }
      if ( (mibt_cnt % update_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        delta.at("W1") = WgtBias_lyr.at("W1") - WgtBias_lyr_old.at("W1");
        delta.at("W2") = WgtBias_lyr.at("W2") - WgtBias_lyr_old.at("W2");
        delta.at("b1") = WgtBias_lyr.at("b1") - WgtBias_lyr_old.at("b1");
        delta.at("b2") = WgtBias_lyr.at("b2") - WgtBias_lyr_old.at("b2");
        // push
        _paracel_bupdate("W1", delta.at("W1"));
        _paracel_bupdate("W2", delta.at("W2"));
        _paracel_bupdate("b1", delta.at("b1"));
        _paracel_bupdate("b2", delta.at("b2"));
      }
      mibt_cnt += 1;
    }  // traverse
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias_lyr.at("W1") = _paracel_read("W1", WgtBias_lyr.at("W1").rows(), WgtBias_lyr.at("W1").cols());
  WgtBias_lyr.at("W2") = _paracel_read("W2", WgtBias_lyr.at("W2").rows(), WgtBias_lyr.at("W2").cols());
  WgtBias_lyr.at("b1") = _paracel_read("b1");
  WgtBias_lyr.at("b2") = _paracel_read("b2");
}


void autoencoder::train(int lyr){
  int i;
  auto lines = paracel_load(input[lyr]);
  local_parser(lines); 
  sync();
  assert(data.rows() == layer_size[lyr]);  // QA
  if (learning_method == "dbgd") {
    std::cout << "chose distributed batch gradient descent" << std::endl;
    set_total_iters(rounds); // default value
    for (i = 0; i < n_lyr; i++) {
      distribute_bgd(lyr);
    }
  } else if (learning_method == "dsgd") {
    std::cout << "chose downpour stochasitc gradient descent" << std::endl;
    set_total_iters(rounds); // default value
    for (i = 0; i < n_lyr; i++) {
      downpour_sgd(lyr);
    }
  } else if (learning_method == "mbdsgd") {
    std::cout << "chose mini-batch downpour stochastic gradient descent" << std::endl;
    set_total_iters(rounds); // default value
    for (i = 0; i < n_lyr; i++) {
      downpour_sgd_mibt(lyr);
    }
  } else {
    std::cout << "learning method not supported." << std::endl;
    return;
  }
  sync();
  data = (WgtBias[lyr].at("W1") * data).colwise() + WgtBias[lyr].at("b1");
  local_dump_Mat(data, input[lyr+1]);
}


void autoencoder::train(){
  // top function
  for (int i = 0; i < n_lyr; i++) {
    train(i);
    dump_result(i);
  }
  std::cout << "Mission complete" << std::endl;
}


void autoencoder::local_parser(const vector<string> & linelst, const char sep, bool spv){
  samples.resize(0);
  labels.resize(0);
  if (spv) {  // supervised
    for (auto & line: linelst) {
      vector<double> tmp;
      auto linev = paracel::str_split(line, sep);
      // WHY???
      //tmp.push_back(1.);  
      for (size_t i = 0; i < linev.size() - 1; i++) {
        tmp.push_back(std::stod(linev[i]));
      }
      samples.push_back(tmp);
      labels.push_back(std::stod(linev.back()));
    } // traverse file
  } else {  // unsupervised
    for (auto & line : linelst) {
      vector<double> tmp;
      auto linev = paracel::str_split(line, sep);
      // WHY??
      //tmp.push_back(1.);
      for (size_t i = 0; i < linev.size(); i++) {
        tmp.push_back(std::stod(linev[i]));
      }
      samples.push_back(tmp);
    }
  }
  // AVAILBLE WITH SPMD???
  data = vec_to_mat(samples).transpose();  // transpose is needed, since the data is sliced by-row 
                                 // and samples are stored by-column in variable "data".
}

void autoencoder::local_dump_Mat(const MatrixXd & m, const string filename, const char sep){
  std::ofstream os;
  os.open(filename);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      os << std::to_string(data(i, j)) << sep;
    }
    os << "\n";
  }
  os.close();
}


MatrixXd autoencoder::vec_to_mat(vector< vector<double> > & v) {
  MatrixXd m(v.size(), v[0].size());
  for (int i = 0; i < (int)v.size(); i++) {
    m.row(i) = VectorXd::Map(&v[i][0], v[i].size());  // row ordered
  }
  return m;
}


VectorXd autoencoder::vec_to_mat(vector<double> & v) {
  VectorXd m(v.size());
  m = VectorXd::Map(&v[0], v.size()); // column ordered
  return m;
}

MatrixXd autoencoder::vec_to_mat(vector<double> & v, int r, int c){
  assert( (int)v.size() == r * c );
  MatrixXd m(r, c);
  m = MatrixXd::Map(&v[0], r, c); // column ordered
  return m;
}

vector<double> autoencoder::Mat_to_vec(MatrixXd & m){
  vector<double> v(m.size());
  // column ordered
  Eigen::Map<MatrixXd>(v.data(), m.rows(), m.cols()) = m;
  return v;
}


void autoencoder::_paracel_write(string key, MatrixXd & m){
  paracel_write(key, Mat_to_vec(m));
}

MatrixXd autoencoder::_paracel_read(string key, int r, int c){
  vector<double> v = paracel_read<vector<double> >(key);
  return vec_to_mat(v, r, c);
}

VectorXd autoencoder::_paracel_read(string key){
  vector<double> v = paracel_read<vector<double> >(key);
  return vec_to_mat(v);
}

void autoencoder::_paracel_bupdate(string key, MatrixXd & m){
  paracel_bupdate(key, Mat_to_vec(m));
}

void autoencoder::dump_result(int lyr) const{
  int i;
  MatrixXd tmp;
  if (get_worker_id() == 0) {
    for (i = 0; i < WgtBias[lyr].at("W1").rows(); i++) {
      tmp = WgtBias[lyr].at("W1").row(i);
      paracel_dump_vector(Mat_to_vec(tmp), 
            ("ae_layer_" + std::to_string(lyr) + "_W1_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr].at("W2").rows(); i++) {
      tmp = WgtBias[lyr].at("W2").row(i);
      paracel_dump_vector(Mat_to_vec(tmp), 
            ("ae_layer_" + std::to_string(lyr) + "_W2_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr].at("b1").rows(); i++) {
      tmp = WgtBias[lyr].at("b1").row(i);
      paracel_dump_vector(Mat_to_vec(tmp), 
            ("ae_layer_" + std::to_string(lyr) + "_b1_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr].at("b2").rows(); i++) {
      tmp = WgtBias[lyr].at("b2").row(i);
      paracel_dump_vector(Mat_to_vec(tmp), 
            ("ae_layer_" + std::to_string(lyr) + "_b2_"), ",", false);
    }
  }
}


} // namespace paracel

