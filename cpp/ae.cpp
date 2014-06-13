#include <algorithm>
#include <iostream>
#include "ae.hpp"
#include <cmath>
#include <random>

// global var
VectorXd g_rho;  // for sparse penalty

namespace paracel{

// construction function
autoencoder::autoencoder(paracel::Comm comm, string hosts_dct_str,
          string _input, string _output, vector<int> _hidden_size,
          int _visible_size, string method, string _acti_func_type, 
          int _rounds, double _alpha, bool _debug, int limit_s, 
          bool ssp_switch, double _lamb, double _sparsity_param, 
          double _beta, int _mibt_size, int _read_batch, int _update_batch, 
          bool _corrupt, double _dvt, double _foc) :
  paracel::paralg(hosts_dct_str, comm, _output, _rounds, limit_s, ssp_switch),
  input(_input),
  output(_output),
  worker_id(comm.get_rank()),
  rounds(_rounds),
  mibt_size(_mibt_size),
  read_batch(_read_batch),
  update_batch(_update_batch),
  learning_method(method),
  acti_func_type(_acti_func_type),
  debug(_debug),
  lamb(_lamb),
  sparsity_param(_sparsity_param),
  beta(_beta),
  alpha(_alpha),
  hidden_size(_hidden_size),
  visible_size(_visible_size),
  corrupt(_corrupt),
  dvt(_dvt),
  foc(_foc)  {
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
  //double r = sqrt(1);
  unordered_map<string, MatrixXd> InitWgtBias;
  for (int i = 0; i < n_lyr; i++) {
    //MatrixXd W1 = (MatrixXd::Random(layer_size[i+1], layer_size[i]).array() * 2 * r - r).matrix();
    //MatrixXd W2 = (MatrixXd::Random(layer_size[i], layer_size[i+1]).array() * 2 * r - r).matrix();
    MatrixXd W1 = MatrixXd::Random(layer_size[i+1], layer_size[i]);
    MatrixXd W2 = MatrixXd::Random(layer_size[i], layer_size[i+1]);
    VectorXd b1 = VectorXd::Zero(layer_size[i+1]);
    VectorXd b2 = VectorXd::Zero(layer_size[i]);
    InitWgtBias["W1"] = W1;
    InitWgtBias["W2"] = W2;
    InitWgtBias["b1"] = b1;
    InitWgtBias["b2"] = b2;

    WgtBias.push_back(InitWgtBias);
  }
}

inline MatrixXd autoencoder::acti_func(const MatrixXd & non_acti_data) const {
  if (acti_func_type == "sigmoid") {
    return ( (1.0 / (1 + exp(-non_acti_data.array()))).matrix() ); 
  }
  else if (acti_func_type == "ReLU") {
    MatrixXd acti_data(non_acti_data.rows(), non_acti_data.cols());
    for (int i = 0; i < acti_data.rows(); i++) {
      for (int j = 0; j < acti_data.cols(); j++) {
        acti_data(i, j) = max(non_acti_data(i, j), 0.);
      }
    }
    return acti_data;
  }
  else if (acti_func_type == "tanh") { 
    Eigen::ArrayXXd tmp = non_acti_data.array();
    return ( ((exp(tmp) - exp(-tmp)) / (exp(tmp) + exp(-tmp))).matrix() );
  }
  else{
    std::cerr << "The activation function is not implemented by far." << std::endl;
    exit(-1);
  }
}


inline ArrayXXd autoencoder::acti_func_der(const MatrixXd & acti_data) const {
  if (acti_func_type == "sigmoid") {
    return acti_data.array()*(1-acti_data.array());
  }
  else if (acti_func_type == "ReLU") {
    ArrayXXd der(acti_data.rows(), acti_data.cols());
    for (int i = 0; i < der.rows(); i++) {
      for (int j = 0; j < der.cols(); j++) {
        der(i, j) = (acti_data(i, j) > 0) ? 1: 0;
      }
    }
    return der;
  }
  else if (acti_func_type == "tanh") { 
    std::cout << "Not applicable by far" << std::endl;
    exit(-1);
  }
  else{
    std::cerr << "The activation function is not implemented by far." << std::endl;
    exit(-1);
  }
}


vector<unordered_map<string, MatrixXd> > autoencoder::GetWgtBias() const{
  return WgtBias;
 }


// compute the cost of a single layer of NN
double autoencoder::ae_cost(int lyr) const {
  int i;
  double cost = 0;
  VectorXd sparse_kl;  // sparse penalty
  const MatrixXd & W1 = WgtBias[lyr].at("W1");
  const MatrixXd & W2 = WgtBias[lyr].at("W2");
  const VectorXd & b1 = WgtBias[lyr].at("b1");
  const VectorXd & b2 = WgtBias[lyr].at("b2");
  unordered_map<int, VectorXd> a;
  unordered_map<int, VectorXd> z;
  if (beta != 0 && learning_method == "dbgd") {
    g_rho = VectorXd::Zero(b1.rows(), b1.cols());
    // traverse network
    for (i = 0; i < data.cols(); i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_func(z[2]);
      z[3] = W2 * a[2] + b2;
      a[3] = acti_func(z[3]);
      cost += ((a[1]-a[3]).array().pow(2)/2).sum();
      g_rho += a[2];
    }
    // rho post-process
    g_rho = (g_rho.array() / data.cols()).matrix();

    // cost post-process
    sparse_kl = sparsity_param * log(sparsity_param/g_rho.array()) +\
                (1-sparsity_param) * log((1-sparsity_param)/(1-g_rho.array()));
    cost /= data.cols();
    cost += lamb/2. * (W1.array().pow(2).sum() + W2.array().pow(2).sum()) +\
            beta*sparse_kl.sum();
  }else{ // without sparse term
    // traverse network
    for (i = 0; i < data.cols(); i++) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_func(z[2]);
      z[3] = W2 * a[2] + b2;
      a[3] = acti_func(z[3]);
      cost += ((a[1]-a[3]).array().pow(2)/2).sum();
    }
    // cost post-process
    cost /= data.cols();
    cost += lamb/2. * (W1.array().pow(2).sum() + W2.array().pow(2).sum());
  }
  return cost;
}


// compute batch gradient
unordered_map<string, MatrixXd> autoencoder::ae_batch_grad(int lyr) const{
  const MatrixXd & W1 = WgtBias[lyr].at("W1");
  const MatrixXd & W2 = WgtBias[lyr].at("W2");
  const VectorXd & b1 = WgtBias[lyr].at("b1");
  const VectorXd & b2 = WgtBias[lyr].at("b2");

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
    a[2] = acti_func(z[2]);
    z[3] = W2 * a[2] + b2;
    a[3] = acti_func(z[3]);
    sigma[3] = (-(a[1]-a[3]).array() * (acti_func_der(a[3]))).matrix();
    VectorXd sparsity_sigma = -sparsity_param/g_rho.array() +\
                          (1-sparsity_param)*(1-g_rho.array());
    sigma[2] = (((W2.transpose()*sigma[3]).array() + beta*sparsity_sigma.array())*\
                acti_func_der(a[2])).matrix();

    W1_delta += sigma[2] * a[1].transpose();
    W2_delta += sigma[3] * a[2].transpose();
    b1_delta += sigma[2];
    b2_delta += sigma[3];
  }

  // return the gradients
  unordered_map<string, MatrixXd> WgtBiasGrad;
  WgtBiasGrad["W1"] = (W1_delta.array() / data.cols() + lamb * W1.array()).matrix();
  WgtBiasGrad["W2"] = (W2_delta.array() / data.cols() + lamb * W2.array()).matrix();
  WgtBiasGrad["b1"] = (b1_delta.array() / data.cols()).matrix();
  WgtBiasGrad["b2"] = (b2_delta.array() / data.cols()).matrix();

  return WgtBiasGrad;
}


// compute the stochastic gradient
unordered_map<string, MatrixXd> autoencoder::ae_stoc_grad(int lyr, int index) const {
  unordered_map<string, MatrixXd> WgtBiasGrad;
  const MatrixXd & W1 = WgtBias[lyr].at("W1");
  const MatrixXd & W2 = WgtBias[lyr].at("W2");
  const VectorXd & b1 = WgtBias[lyr].at("b1");
  const VectorXd & b2 = WgtBias[lyr].at("b2");
  
  // means no mini-batch
  unordered_map<int, VectorXd> a;
  unordered_map<int, VectorXd> z;
  unordered_map<int, VectorXd> sigma;
  a[1] = data.col(index);
  z[2] = W1 * a[1] + b1;
  a[2] = acti_func(z[2]);
  z[3] = W2 * a[2] + b2;
  a[3] = acti_func(z[3]);
  sigma[3] = (-(a[1]-a[3]).array() * (acti_func_der(a[3]))).matrix();
  sigma[2] = (((W2.transpose()*sigma[3]).array())*acti_func_der(a[2])).matrix();
  //sigma[3] = (-(a[1]-a[3]).array() * (a[3].array()*(1-a[3].array()))).matrix();
  //sigma[2] = (((W2.transpose()*sigma[3]).array())*a[2].array()*(1-a[2].array())).matrix();
  // gradient of that sample
  WgtBiasGrad["W1"] = sigma[2] * a[1].transpose();  
  WgtBiasGrad["W2"] = sigma[3] * a[2].transpose();  
  WgtBiasGrad["W1"] = (WgtBiasGrad.at("W1").array() + lamb * W1.array()).matrix();  
  WgtBiasGrad["W2"] = (WgtBiasGrad.at("W2").array() + lamb * W2.array()).matrix();  
  WgtBiasGrad["b1"] = sigma[2];  
  WgtBiasGrad["b2"] = sigma[3];  

  return WgtBiasGrad;
}


// compute the mini-batch stochastic gradient
unordered_map<string, MatrixXd> autoencoder::ae_mibt_stoc_grad(int lyr, vector<int> index_data) const {

  size_t mini_batch_size = index_data.size();
  unordered_map<string, MatrixXd> WgtBiasGrad;
  const MatrixXd & W1 = WgtBias[lyr].at("W1");
  const MatrixXd & W2 = WgtBias[lyr].at("W2");
  const VectorXd & b1 = WgtBias[lyr].at("b1");
  const VectorXd & b2 = WgtBias[lyr].at("b2");
  
  if (!(mini_batch_size-1)) {
    // means no mini-batch
    WgtBiasGrad = ae_stoc_grad(lyr, index_data[0]);

  }else{
    // Got a mini-batch SGD
    MatrixXd W1_delta = MatrixXd::Zero(W1.rows(), W1.cols());
    MatrixXd W2_delta = MatrixXd::Zero(W2.rows(), W2.cols());
    VectorXd b1_delta = VectorXd::Zero(b1.size());
    VectorXd b2_delta = VectorXd::Zero(b2.size());
    unordered_map<int, VectorXd> a;
    unordered_map<int, VectorXd> z;
    unordered_map<int, VectorXd> sigma;
    /*
    // Get rho first
    VectorXd rho = VectorXd::Zero(b1.size());
    for (auto i : index_data) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_func(z[2]);
      rho += a[2];
    }
    rho = (rho.array() / mini_batch_size).matrix();
    */
    // BP
    for (auto i: index_data) {
      a[1] = data.col(i);
      z[2] = W1 * a[1] + b1;
      a[2] = acti_func(z[2]);
      z[3] = W2 * a[2] + b2;
      a[3] = acti_func(z[3]);
      sigma[3] = (-(a[1]-a[3]).array() * (acti_func_der(a[3]))).matrix();
      sigma[2] = (((W2.transpose()*sigma[3]).array())*acti_func_der(a[2])).matrix();

      W1_delta += sigma[2] * a[1].transpose();
      W2_delta += sigma[3] * a[2].transpose();
      b1_delta += sigma[2];
      b2_delta += sigma[3];
    }
    WgtBiasGrad["W1"] = (W1_delta.array() / mini_batch_size + lamb * W1.array()).matrix();
    WgtBiasGrad["W2"] = (W2_delta.array() / mini_batch_size + lamb * W2.array()).matrix();
    WgtBiasGrad["b1"] = (b1_delta.array() / mini_batch_size).matrix();
    WgtBiasGrad["b2"] = (b2_delta.array() / mini_batch_size).matrix();

  }  // else ends
  return WgtBiasGrad;
}

// for DAE
inline void autoencoder::corrupt_data(){
  assert(corrupt);
  assert(dvt < 0.5 + 1e-4);
  vector<int> id;
  int corrupt_elem_num = data.rows();
  VectorXd gauss_array = VectorXd::Zero(corrupt_elem_num);
  for (int i = 0; i < data.cols(); i++) {
    id.push_back(i);
  }
  std::random_shuffle(id.begin(), id.end());
  int corrupt_data_num = floor(data.cols() * foc);
  id.erase(id.begin() + corrupt_data_num, id.end());
  for (auto it = id.begin(); it != id.end(); ++it) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, dvt);
    for (int i = 0; i < corrupt_elem_num; i++) {
      gauss_array(i) = distribution(generator);
    }
    data.col(*it) = data.col(*it) + gauss_array;
  }
}

// distributed bgd
void autoencoder::distribute_bgd(int lyr){
  // flag
  std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
  _paracel_write("W1", WgtBias[lyr].at("W1"));
  _paracel_write("W2", WgtBias[lyr].at("W2"));
  _paracel_write("b1", WgtBias[lyr].at("b1"));
  _paracel_write("b2", WgtBias[lyr].at("b2"));
  paracel_register_bupdate("/mfs/user/zhaojunbo/paracel/build/lib/libae_update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());
  for (int rd = 0; rd < rounds; rd++) {
    WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
    WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
    WgtBias[lyr].at("b1") = _paracel_read("b1");
    WgtBias[lyr].at("b2") = _paracel_read("b2");
    delta = ae_batch_grad(lyr);
    delta.at("W1") = (-alpha * delta.at("W1").array()).matrix();
    delta.at("W2") = (-alpha * delta.at("W2").array()).matrix();
    delta.at("b1") = (-alpha * delta.at("b1").array()).matrix();
    delta.at("b2") = (-alpha * delta.at("b2").array()).matrix();
    if (debug) {
      loss_error.push_back(ae_cost(lyr));
    }
    // push
    _paracel_bupdate("W1", delta.at("W1"));
    _paracel_bupdate("W2", delta.at("W2"));
    _paracel_bupdate("b1", delta.at("b1"));
    _paracel_bupdate("b2", delta.at("b2"));
    iter_commit();
    
    // flag
    WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
    WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
    WgtBias[lyr].at("b1") = _paracel_read("b1");
    WgtBias[lyr].at("b2") = _paracel_read("b2");
    std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
  } // rounds
  // last pull
  WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  WgtBias[lyr].at("b1") = _paracel_read("b1");
  WgtBias[lyr].at("b2") = _paracel_read("b2");
}


// downpour sgd
void autoencoder::downpour_sgd(int lyr){
  // flag
  std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
  int cnt = 0;
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
  paracel_register_bupdate("/mfs/user/zhaojunbo/paracel/build/lib/libae_update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());

  for (int rd = 0; rd < rounds; rd++) {
    std::random_shuffle(idx.begin(), idx.end());

    // init read
    WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
    WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
    WgtBias[lyr].at("b1") = _paracel_read("b1");
    WgtBias[lyr].at("b2") = _paracel_read("b2");
    unordered_map<string, MatrixXd> WgtBias_lyr_old(WgtBias[lyr]);

    // traverse data
    cnt = 0;
    for (auto sample_id : idx) {
      if ( (cnt % read_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
        WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
        WgtBias[lyr].at("b1") = _paracel_read("b1");
        WgtBias[lyr].at("b2") = _paracel_read("b2");
        WgtBias_lyr_old = WgtBias[lyr];
      }
      unordered_map<string, MatrixXd> WgtBias_grad = ae_stoc_grad(lyr, sample_id);
      WgtBias[lyr].at("W1") -= (alpha * WgtBias_grad["W1"].array()).matrix();
      WgtBias[lyr].at("W2") -= (alpha * WgtBias_grad["W2"].array()).matrix();
      WgtBias[lyr].at("b1") -= (alpha * WgtBias_grad["b1"].array()).matrix();
      WgtBias[lyr].at("b2") -= (alpha * WgtBias_grad["b2"].array()).matrix();
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
      }
      if ( (cnt % update_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        delta.at("W1") = WgtBias[lyr].at("W1") - WgtBias_lyr_old.at("W1");
        delta.at("W2") = WgtBias[lyr].at("W2") - WgtBias_lyr_old.at("W2");
        delta.at("b1") = WgtBias[lyr].at("b1") - WgtBias_lyr_old.at("b1");
        delta.at("b2") = WgtBias[lyr].at("b2") - WgtBias_lyr_old.at("b2");
        // push
        _paracel_bupdate("W1", delta.at("W1"));
        _paracel_bupdate("W2", delta.at("W2"));
        _paracel_bupdate("b1", delta.at("b1"));
        _paracel_bupdate("b2", delta.at("b2"));
        iter_commit();
        // flag
        std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
      }
      cnt += 1;
    } // traverse
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  WgtBias[lyr].at("b1") = _paracel_read("b1");
  WgtBias[lyr].at("b2") = _paracel_read("b2");
}


// mini-batch downpour sgd
void autoencoder::downpour_sgd_mibt(int lyr){  // TODO Adagrad
  // flag
  std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
  int mibt_cnt = 0;
  if (read_batch == 0) { read_batch = 4; }
  if (update_batch == 0) { update_batch = 4; }
  // Reference operator
  _paracel_write("W1", WgtBias[lyr].at("W1"));
  _paracel_write("W2", WgtBias[lyr].at("W2"));
  _paracel_write("b1", WgtBias[lyr].at("b1"));
  _paracel_write("b2", WgtBias[lyr].at("b2"));
  vector<int> idx;
//std::cout << get_worker_id() << " ok1" << std::endl;
  for (int i = 0; i < data.cols(); i++) {
    idx.push_back(i);
  }
  // ABSOULTE PATH
  paracel_register_bupdate("/mfs/user/zhaojunbo/paracel/build/lib/libae_update.so", 
      "ae_update");
  unordered_map<string, MatrixXd> delta;
  delta["W1"] = MatrixXd::Zero(WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  delta["W2"] = MatrixXd::Zero(WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  delta["b1"] = VectorXd::Zero(WgtBias[lyr].at("b1").size());
  delta["b2"] = VectorXd::Zero(WgtBias[lyr].at("b2").size());
//std::cout << get_worker_id() << " ok2" << std::endl;

  for (int rd = 0; rd < rounds; rd++) {
    std::random_shuffle(idx.begin(), idx.end());
    vector<vector<int>> mibt_idx; // mini-batch id
    for (auto i = idx.begin(); ; i += mibt_size) {
      if (idx.end() - i < mibt_size) {
        if (idx.end() - i < 2) { // point to the back() or out of range  
          break;
        }else{
          vector<int> tmp;
          tmp.assign(i, idx.end());
          mibt_idx.push_back(tmp);
          break;
        }
      }
      vector<int> tmp;
      tmp.assign(i, i + mibt_size);
      // SUPPOSE IT TO BE NOT ACCUMULATED OVER WORKERS?
      mibt_idx.push_back(tmp);
    }
    // init push
    WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
    WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
    WgtBias[lyr].at("b1") = _paracel_read("b1");
    WgtBias[lyr].at("b2") = _paracel_read("b2");
    unordered_map<string, MatrixXd> WgtBias_lyr_old(WgtBias[lyr]);
//std::cout << get_worker_id() << " ok3" << std::endl;
    
    // traverse data
    mibt_cnt = 0;
    for (auto mibt_sample_id : mibt_idx) {
      if ( (mibt_cnt % read_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
        WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
        WgtBias[lyr].at("b1") = _paracel_read("b1");
        WgtBias[lyr].at("b2") = _paracel_read("b2");
        WgtBias_lyr_old = WgtBias[lyr];
//std::cout << get_worker_id() << " ok4" << std::endl;
      }
      unordered_map<string, MatrixXd> WgtBias_grad = ae_mibt_stoc_grad(lyr, mibt_sample_id);
      WgtBias[lyr].at("W1") -= (alpha * WgtBias_grad["W1"].array()).matrix();
      WgtBias[lyr].at("W2") -= (alpha * WgtBias_grad["W2"].array()).matrix();
      WgtBias[lyr].at("b1") -= (alpha * WgtBias_grad["b1"].array()).matrix();
      WgtBias[lyr].at("b2") -= (alpha * WgtBias_grad["b2"].array()).matrix();
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
      }
      if ( (mibt_cnt % update_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        delta.at("W1") = WgtBias[lyr].at("W1") - WgtBias_lyr_old.at("W1");
        delta.at("W2") = WgtBias[lyr].at("W2") - WgtBias_lyr_old.at("W2");
        delta.at("b1") = WgtBias[lyr].at("b1") - WgtBias_lyr_old.at("b1");
        delta.at("b2") = WgtBias[lyr].at("b2") - WgtBias_lyr_old.at("b2");
        // push
        _paracel_bupdate("W1", delta.at("W1"));
        _paracel_bupdate("W2", delta.at("W2"));
        _paracel_bupdate("b1", delta.at("b1"));
        _paracel_bupdate("b2", delta.at("b2"));
        iter_commit();
//std::cout << get_worker_id() << " ok5" << std::endl;
        // flag
        std::cout << "worker" << get_worker_id() << ", cost: " << ae_cost(lyr) << std::endl;
      }
      mibt_cnt += 1;
    }  // traverse
//std::cout << get_worker_id() << " ok6" << std::endl;
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias[lyr].at("W1") = _paracel_read("W1", WgtBias[lyr].at("W1").rows(), WgtBias[lyr].at("W1").cols());
  WgtBias[lyr].at("W2") = _paracel_read("W2", WgtBias[lyr].at("W2").rows(), WgtBias[lyr].at("W2").cols());
  WgtBias[lyr].at("b1") = _paracel_read("b1");
  WgtBias[lyr].at("b2") = _paracel_read("b2");
}


void autoencoder::train(int lyr){
  if (lyr == 0) {
    string data_dir = todir(input); // distributed stored data
    auto lines = paracel_load(data_dir);
    local_parser(lines, ' ', true); // includes label
    data = vec_to_mat(samples).transpose();   
    samples.resize(0);
    lines.resize(0);

    // DAE configuration
    if (corrupt) {
      std::cout << "worker" << get_worker_id() << " Setting for Denoising" << std::endl;
      corrupt_data();
    }
  }
  assert(data.rows() == layer_size[lyr] &&\
      "Modify layers' size in .json file to adjust data's dimension");  // QA
  if (learning_method == "dbgd") {
    std::cout << "worker" << get_worker_id() << " chose distributed batch gradient descent" << std::endl;
    set_total_iters(rounds); // default value
    distribute_bgd(lyr);
  } else if (learning_method == "dsgd") {
    std::cout << "worker" << get_worker_id() << " chose downpour stochasitc gradient descent" << std::endl;
    set_total_iters(rounds * ceil(data.cols() / float(update_batch))); // consider update_batch
    downpour_sgd(lyr);
  } else if (learning_method == "mbdsgd") {
    std::cout << "worker" << get_worker_id() << " chose mini-batch downpour stochastic gradient descent" << std::endl;
    int n_mibt = ceil(data.cols() / float(mibt_size));
    set_total_iters(rounds * ceil(n_mibt / float(update_batch))); // consider update_batch
    downpour_sgd_mibt(lyr);
  } else {
    std::cout << "worker" << get_worker_id() << " learning method not supported." << std::endl;
    return;
  }
  // data for next layer
  data = acti_func((WgtBias[lyr].at("W1") * data).colwise() + MatrixXd::ColXpr(WgtBias[lyr].at("b1").col(0)));
  // Discard IO operations
  /*
  if (get_worker_id() == 0) {  // delete the previous data file, since it is stored by ios::app
    string rmname;
    rmname = todir(input) + "data_" + std::to_string(lyr+1) + ".txt";
    std::remove(rmname.c_str());
  }
  sync();
  local_dump_Mat(data.transpose(), (todir(input) + "data_" + std::to_string(lyr+1) + ".txt"), ' ');
  data.resize(0, 0); // data clear
  */
  std::cout << "Finish training layer: " << lyr << std::endl;
}


void autoencoder::train(){
  // top function
  for (int i = 0; i < n_lyr; i++) {
    std::cout << "worker" << get_worker_id() << " starts training layer " << i+1 << std::endl;
    train(i);
    if (get_worker_id() == 0) {
      dump_result(i);
    }
  }
  sync();
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
}

void autoencoder::local_dump_Mat(const MatrixXd & m, const string filename, const char sep){
  std::ofstream os;
  os.open(filename, std::ofstream::app);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      os << std::to_string(m(i, j)) << sep;
    }
    os << "\n";
  }
  os.close();
}


inline MatrixXd autoencoder::vec_to_mat(const vector< vector<double> > & v) {
  MatrixXd m(v.size(), v[0].size());
  for (size_t i = 0; i < v.size(); i++) {
    m.row(i) = VectorXd::Map(&v[i][0], v[i].size());  // row ordered
  }
  return m;
}


inline VectorXd autoencoder::vec_to_mat(const vector<double> & v) {
  VectorXd m(v.size());
  m = VectorXd::Map(&v[0], v.size()); // column ordered
  return m;
}

inline MatrixXd autoencoder::vec_to_mat(const vector<double> & v, int r){
  assert( v.size() % r == 0);
  int c = v.size() / r;
  return vec_to_mat(v, r, c);
}

inline MatrixXd autoencoder::vec_to_mat(const vector<double> & v, int r, int c){
  assert( (int)v.size() == r * c );
  MatrixXd m(r, c);
  m = MatrixXd::Map(&v[0], r, c); // column ordered
  return m;
}

inline vector<double> autoencoder::Mat_to_vec(const MatrixXd & m){
  vector<double> v(m.size());
  // column ordered
  Eigen::Map<MatrixXd>(v.data(), m.rows(), m.cols()) = m;
  return v;
}


inline void autoencoder::_paracel_write(string key, MatrixXd & m){
  paracel_write(key, Mat_to_vec(m));
}

inline MatrixXd autoencoder::_paracel_read(string key, int r, int c){
  vector<double> v = paracel_read<vector<double> >(key);
  return vec_to_mat(v, r, c);
}

inline VectorXd autoencoder::_paracel_read(string key){
  vector<double> v = paracel_read<vector<double> >(key);
  return vec_to_mat(v);
}

inline void autoencoder::_paracel_bupdate(string key, MatrixXd & m){
  paracel_bupdate(key, Mat_to_vec(m));
}


void autoencoder::dump_mat(const MatrixXd & m, const string filename) const {
  std::fstream fout;
  fout.open(filename, std::ios::out);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      fout << std::to_string(m(i, j)) << ' ';
    }
    fout << "\n";
  }
  fout.close();
}


void autoencoder::dump_result(int lyr) const {
  MatrixXd tmp;
  dump_mat(WgtBias[lyr].at("W1"), (todir(output) + "ae_layer_" + std::to_string(lyr) + "_W1"));
  dump_mat(WgtBias[lyr].at("W2"), (todir(output) + "ae_layer_" + std::to_string(lyr) + "_W2"));
  dump_mat(WgtBias[lyr].at("b1"), (todir(output) + "ae_layer_" + std::to_string(lyr) + "_b1"));
  dump_mat(WgtBias[lyr].at("b2"), (todir(output) + "ae_layer_" + std::to_string(lyr) + "_b2"));
  }


} // namespace paracel

