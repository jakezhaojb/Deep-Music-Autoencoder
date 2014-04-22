#include "ae.hpp"
#include <algorithm>
#include <iostream>
#include "ps.hpp"
#include "utils.hpp"

namespace paracel{

// construction function
autoencoder::autoencoder(paracel::Comm comm, string hosts_dct_str,
          string _input, string output, vector<int> _hidden_size,
          vector<int> _visible_size, string method, int _rounds, 
          double _alpha, bool _debug, int limit_s, bool ssp_switch, 
          double _lamb, double _sparsity_param, double _beta, int _mibt_size) :
  paracel::paralg(hosts_dct_str, comm, output, _rounds, limit_s, ssp_switch),
  input(_input),
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
    assert(_hidden_size.size() == _visible_size.size());
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
std::map<string, MatrixXd> autoencoder::ae_stoc_grad(int lyr, int index) const {
  return compute_stoc_grad(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta, index);
}


// compute the mini-batch stochastic gradient
std::map<string, MatrixXd> autoencoder::ae_mibt_stoc_grad(int lyr, vector<int> index_data) const {
  return compute_mibt_stoc_grad(WgtBias[lyr], data, hidden_size[lyr], visible_size[lyr], lamb, sparsity_param, beta, index_data);
}


// distributed bgd
void autoencoder::distribute_bgd(int lyr){
  std::map<string, MatrixXd>& WgtBias_lyr = WgtBias[lyr] ;
  paracel_write("W1", WgtBias_lyr["W1"]);
  paracel_write("W2", WgtBias_lyr["W2"]);
  paracel_write("b1", WgtBias_lyr["b1"]);
  paracel_write("b2", WgtBias_lyr["b2"]);
  paracel_register_bupdate("./update.so", 
      "ae_update");
  std::map<string, MatrixXd> delta;
  for (int rd = 0; rd < rounds; rd++) {
    WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
    WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
    WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
    WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
    delta = ae_batch_grad(lyr);
    delta["W1"] *= alpha;
    delta["W2"] *= alpha;
    delta["b1"] *= alpha;
    delta["b2"] *= alpha;
    if (debug) {
      loss_error.push_back(ae_cost(lyr));
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
  assert( (lyr > 0 && lyr < n_lyr) && "Input layer not qualified!");
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
    WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
    WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
    WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
    WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
    std::map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);

    // traverse data
    cnt = 0;
    for (auto sample_id : idx) {
      if ( (cnt % read_batch == 0) || (cnt == (int)idx.size() - 1) ) {
        WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
        WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
        WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
        WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      std::map<string, MatrixXd> WgtBias_grad = ae_stoc_grad(lyr, sample_id);
      WgtBias_lyr["W1"] += alpha * WgtBias_grad["W1"];
      WgtBias_lyr["W2"] += alpha * WgtBias_grad["W2"];
      WgtBias_lyr["b1"] += alpha * WgtBias_grad["b1"];
      WgtBias_lyr["b2"] += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
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
  assert( (lyr > 0 && lyr < n_lyr) && "Input layer not qualified!");
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
  // ABSOULTE PATH
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
    WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
    WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
    WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
    WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
    std::map<string, MatrixXd> WgtBias_lyr_old(WgtBias_lyr);
    
    // traverse data
    mibt_cnt = 0;
    for (auto mibt_sample_id : mibt_idx) {
      if ( (mibt_cnt % read_batch == 0) || (mibt_cnt == (int)mibt_idx.size()-1) ) {
        WgtBias_lyr["W1"] = paracel_read<MatrixXd>("W1");
        WgtBias_lyr["W2"] = paracel_read<MatrixXd>("W2");
        WgtBias_lyr["b1"] = paracel_read<MatrixXd>("b1");
        WgtBias_lyr["b2"] = paracel_read<MatrixXd>("b2");
        WgtBias_lyr_old = WgtBias_lyr;
      }
      std::map<string, MatrixXd> WgtBias_grad = ae_mibt_stoc_grad(lyr, mibt_sample_id);
      WgtBias_lyr["W1"] += alpha * WgtBias_grad["W1"];
      WgtBias_lyr["W2"] += alpha * WgtBias_grad["W2"];
      WgtBias_lyr["b1"] += alpha * WgtBias_grad["b1"];
      WgtBias_lyr["b2"] += alpha * WgtBias_grad["b2"];
      if (debug) {
        loss_error.push_back(ae_cost(lyr));
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


void autoencoder::train(int lyr){
  int i;
  auto lines = paracel_load(input);
  local_parser(lines); 
  sync();
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
      tmp.push_back(1.);  
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
      tmp.push_back(1.);
      for (size_t i = 0; i < linev.size(); i++) {
        tmp.push_back(std::stod(linev[i]));
      }
      samples.push_back(tmp);
    }
  }
  data = vec_to_mat(samples).transpose();  // transpose is needed, since the data is sliced by-row 
                                 // and samples are stored by-column in variable "data".
}


MatrixXd vec_to_mat(vector< vector<double> > v){
  MatrixXd m(v.size(), v[0].size());
  for (int i = 0; i < v.size(); i++) {
    for (int j = 0; j < v[0].size(); j++) {
      m(i, j) = v[i][j];
    }
  }
  return m;
}


VectorXd vec_to_mat(vector<double> v){
  VectorXd m(v.size());
  for (int i = 0; i < v.size(); i++) {
    m(i) = v[i];
  }
  return m;
}


vector<double> Vec_to_vec(MatrixXd & m){
  assert( (m.cols()==1 || m.rows() == 1) && "Input of Vec_to_vec should be a Vector or RowVector");
  vector<double> v;
  for (int i = 0; i < m.size(); i++) {
    v.push_back(m(i));
  }
  return v;
}


void autoencoder::dump_result(int lyr){
  int i;
  if (get_worker_id() == 0) {
    for (i = 0; i < WgtBias[lyr]["W1"].rows(); i++) {
      paracel_dump_vector(Vec_to_vec(WgtBias[lyr]["W1"].row(i)), 
            ("ae_layer_" + std::to_string(lyr) + "_W1_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr]["W2"].rows(); i++) {
      paracel_dump_vector(Vec_to_vec(WgtBias[lyr]["W2"].row(i)), 
            ("ae_layer_" + std::to_string(lyr) + "_W2_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr]["b1"].rows(); i++) {
      paracel_dump_vector(Vec_to_vec(WgtBias[lyr]["b1"].row(i)), 
            ("ae_layer_" + std::to_string(lyr) + "_b1_"), ",", false);
    }
    for (i = 0; i < WgtBias[lyr]["b2"].rows(); i++) {
      paracel_dump_vector(Vec_to_vec(WgtBias[lyr]["b2"].row(i)), 
            ("ae_layer_" + std::to_string(lyr) + "_b2_"), ",", false);
    }
  }
}

} // namespace paracel
