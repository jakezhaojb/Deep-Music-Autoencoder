#include "ae.hpp"
#include <algorithm>
#include <iostream>
#include "ps.hpp"
#include "utils.hpp"

namespace paracel{

// construction function
autoencoder::autoencoder(paracel::Comm comm, string hosts_dct_str,
          string _input, string output, MatrixXd _data, string method, 
          int _rounds, double _alpha, bool _debug, int limit_s, 
          bool ssp_switch, vector<int> _hidden_size, vector<int> _visible_size, 
          double _lamb, double _sparsity_param, double _beta, int _mibt_size) :
  paracel::paralg(hosts_dct_str, comm, output, _rounds, limit_s, ssp_switch),
  input(_input),
  data(_data),
  learning_method(method),
  worker_id(comm.get_rank()),
  rounds(_rounds),
  alpha(_alpha),
  debug(_debug),
  lamb(_lamb),
  sparsity_param(_sparsity_param),
  beta(_beta),
  mibt_size(_mibt_size) {
    int i;
    assert(_hidden_size.size()() == _visible_size.size());
    n_lyr = _hidden_size.size();
    hidden_size.assign(_hidden_size.begin(), _hidden_size.end());
    visible_size.assign(_visible_size.begin(), _visible_size.end());
    for (i = 0; i < n_lyr; i++) {
      WgtBias.push_back(initialize_parameter(hidden_size[i], visible_size[i]));
    }

  }


autoencoder::~autoencoder() {}


// compute the cost of Neural Networks
double autoencoder::ae_cost(int lyr) const {
  return compute_cost(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta);
}


// compute the batch gradient
std::map<string, MatrixXd> autoencoder::ae_batch_grad(int lyr) const{
  return compute_batch_grad(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta);
}


// compute the stochastic gradient
std::map<string, MatrixXd> autoencoder::ae_stoc_grad(int lyr, int index){
  return compute_stoc_grad(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta, index);
}


// compute the mini-batch stochastic gradient
std::map<string, MatrixXd> autoencoder::ae_mibt_stoc_grad(int lyr, vector<int> index_data){
  return compute_mibt_stoc_grad(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta, index_data);
}


// distributed bgd
void autoencoder::distribute_bgd(int lyr){
  int data_sz = data.cols();
  std::map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  paracel_write("W1", WgtBias_lyr["W1"]);
  paracel_write("W2", WgtBias_lyr["W2"]);
  paracel_write("b1", WgtBias_lyr["b1"]);
  paracel_write("b2", WgtBias_lyr["b2"]);
  paracel_register_bupdate("./update.so", 
      "ae_update");
  std::map<string, MatrixXd> delta;
  for (int rd = 0; rd < rounds; rd++) {
    WgtBias_lyr["W1"] = paracel_read("W1");
    WgtBias_lyr["W2"] = paracel_read("W2");
    WgtBias_lyr["b1"] = paracel_read("b1");
    WgtBias_lyr["b2"] = paracel_read("b2");
    delta = ae_batch_grad(lyr);
    delta["W1"] *= alpha;
    delta["W2"] *= alpha;
    delta["b1"] *= alpha;
    delta["b2"] *= alpha;
    if (debug) {
      loss_error.push_back(compute_cost(lyr));
    }
    // push
    paracel_bupdate("W1", delta["W1"]);
    paracel_bupdate("W2", delta["W2"]);
    paracel_bupdate("b1", delta["b1"]);
    paracel_bupdate("b2", delta["b2"]);
    iter_commit();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  } // rounds
  // last pull
  WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
  WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
  WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
  WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
}


// downpour sgd
void autoencoder::downpour_sgd(int lyr){
  int data_sz = data.cols();
  int cnt = 0, read_batch = data_sz/ 1000, update_batch = data_sz / 100;
  assert(lyr > 0 && lyr < layers && "Input layer not qualified!");
  if (read_batch == 0) { read_batch = 10; }
  if (update_batch == 0) { update_batch = 10; }
  // Reference operator
  std::map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  paracel_write("W1", WgtBias_lyr["W1"]);
  paracel_write("W2", WgtBias_lyr["W2"]);
  paracel_write("b1", WgtBias_lyr["b1"]);
  paracel_write("b2", WgtBias_lyr["b2"]);
  vector<int> idx;
  for (int i = 0; i < data.cols(); i++) {
    idx.push_back(i);
  }
  paracel_register_bupdate("./update.so", 
      "ae_update");
  std::map<string, MatrixXd> delta;

  for (int rd = 0; rd < rounds; rd++) {
    std::random_shuffle(idx.begin(), idx.end());

    // init read
    WgtBias_lyr["W1"] = paracel_read("W1");
    WgtBias_lyr["W2"] = paracel_read("W2");
    WgtBias_lyr["b1"] = paracel_read("b1");
    WgtBias_lyr["b2"] = paracel_read("b2");
    std::map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);

    // traverse data
    cnt = 0;
    for (auto sample_id : idx) {
      if ( (cnt % read_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        WgtBias_lyr["W1"] = paracel_read("W1");
        WgtBias_lyr["W2"] = paracel_read("W2");
        WgtBias_lyr["b1"] = paracel_read("b1");
        WgtBias_lyr["b2"] = paracel_read("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      std::map<string, MatrixXd> WgtBias_grad = ae_stoc_grad(lyr, sample_id);
      WgtBias_lyr["W1"] += alpha * WgtBias_grad["W1"];
      WgtBias_lyr["W2"] += alpha * WgtBias_grad["W2"];
      WgtBias_lyr["b1"] += alpha * WgtBias_grad["b1"];
      WgtBias_lyr["b2"] += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(compute_cost(lyr));
      }
      if ( (cnt % update_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        delta["W1"] = WgtBias_lyr["W1"] - WgtBias_lyr_old["W1"];
        delta["W2"] = WgtBias_lyr["W2"] - WgtBias_lyr_old["W2"];
        delta["b1"] = WgtBias_lyr["b1"] - WgtBias_lyr_old["b1"];
        delta["b2"] = WgtBias_lyr["b2"] - WgtBias_lyr_old["b2"];
        // push
        paracel_bupdate("W1", delta["W1"]);
        paracel_bupdate("W2", delta["W2"]);
        paracel_bupdate("b1", delta["b1"]);
        paracel_bupdate("b2", delta["b2"]);
      }
      cnt += 1;
    } // traverse
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
  WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
  WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
  WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
}


// mini-batch downpour sgd
void autoencoder::downpour_sgd_mibt(int lyr){
  int data_sz = data.cols();
  int mibt_cnt = 0, read_batch = data_sz / (mibt_size*100), update_batch = data_sz / (mibt_size*100);
  assert(lyr > 0 && lyr < layers && "Input layer not qualified!");
  if (read_batch == 0) { read_batch = 10; }
  if (update_batch == 0) { update_batch = 10; }
  // Reference operator
  std::map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  paracel_write("W1", WgtBias_lyr["W1"]);
  paracel_write("W2", WgtBias_lyr["W2"]);
  paracel_write("b1", WgtBias_lyr["b1"]);
  paracel_write("b2", WgtBias_lyr["b2"]);
  vector<int> idx;
  for (int i = 0; i < data.cols(); i++) {
    idx.push_back(i);
  }
  paracel_register_bupdate("./update.so", 
      "ae_update");
  std::map<string, MatrixXd> delta;

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
    WgtBias_lyr["W1"] = paracel_read("W1");
    WgtBias_lyr["W2"] = paracel_read("W2");
    WgtBias_lyr["b1"] = paracel_read("b1");
    WgtBias_lyr["b2"] = paracel_read("b2");
    std::map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);
    
    // traverse data
    mibt_cnt = 0;
    for (auto mibt_sample_id : mibt_idx) {
      if ( (mibt_cnt % read_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        WgtBias_lyr["W1"] = paracel_read("W1");
        WgtBias_lyr["W2"] = paracel_read("W2");
        WgtBias_lyr["b1"] = paracel_read("b1");
        WgtBias_lyr["b2"] = paracel_read("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      std::map<string, MatrixXd> WgtBias_grad = ae_mibt_stoc_grad(lyr, mibt_sample_id);
      WgtBias_lyr["W1"] += alpha * WgtBias_grad["W1"];
      WgtBias_lyr["W2"] += alpha * WgtBias_grad["W2"];
      WgtBias_lyr["b1"] += alpha * WgtBias_grad["b1"];
      WgtBias_lyr["b2"] += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(compute_cost(lyr));
      }
      if ( (mibt_cnt % update_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        delta["W1"] = WgtBias_lyr["W1"] - WgtBias_lyr_old["W1"];
        delta["W2"] = WgtBias_lyr["W2"] - WgtBias_lyr_old["W2"];
        delta["b1"] = WgtBias_lyr["b1"] - WgtBias_lyr_old["b1"];
        delta["b2"] = WgtBias_lyr["b2"] - WgtBias_lyr_old["b2"];
        // push
        paracel_bupdate("W1", delta["W1"]);
        paracel_bupdate("W2", delta["W2"]);
        paracel_bupdate("b1", delta["b1"]);
        paracel_bupdate("b2", delta["b2"]);
      }
      mibt_cnt += 1;
    }  // traverse
    sync();
    std::cout << "worker" << get_worker_id() << "at the end of rd" << rd << std::endl;
  }  // rounds
  // last pull
  WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
  WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
  WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
  WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
}

void autoencoder::train(){
  // init data, TO BE UPDATED
  //auto lines = paracel_load(input);
  //local_parser(lines); 
  //sync();
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
}

} // namespace paracel
