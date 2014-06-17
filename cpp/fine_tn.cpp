#include "fine_tn.hpp" 
#include <cmath>
#include <cassert>

vector<int> GID{331, 335, 325, 337, 328, 334, 336, 327, 326, 332, 333, 324, 329, 330};

namespace paracel{

fine_tune::fine_tune(paracel::Comm comm, string hosts_dct_str,
          string _input, string _output, vector<int> _hidden_size,
          int _visible_size, vector<unordered_map<string, MatrixXd> > _WgtBias,
          string method, string _acti_func_type, 
          int _rounds, double _alpha, bool _debug, int limit_s, 
          bool ssp_switch, double _lamb, double _sparsity_param, 
          double _beta, int _mibt_size, int _read_batch, int _update_batch,
          int _n_class):
      autoencoder(comm, hosts_dct_str, _input, _output, 
                  _hidden_size, _visible_size, method, 
                  _acti_func_type, _rounds, _alpha, _debug, 
                  limit_s, ssp_switch, _lamb, _sparsity_param, 
                  _beta, _mibt_size, _read_batch, _update_batch),
      input(_input),
      output(_output),
      n_class(_n_class) {
        WgtBias = _WgtBias;
        //smx_init();
      }

fine_tune::~fine_tune() {}


void fine_tune::smx_init(){
  // randomly generate smx_W $ smx_b
  assert(data_top.rows() != 0 && "We don't have data_top yet\n");
  smx_W = MatrixXd::Random(n_class, data_top.rows());
  // process labels.
  for(auto it = labels.begin(); it != labels.end(); it++) {
    auto pos = std::find(GID.begin(), GID.end(), *it);
    assert(pos != GID.end());
    *it = pos - GID.begin();
  }
  data_lbl = VectorXd::Map(&labels[0], labels.size());
}


vector<vector<double> > fine_tune::smx_prob(MatrixXd & theta) const{
  MatrixXd prob;
  assert(theta.rows() == n_class)
  assert(theta.cols() == data_top.rows())
  ArrayXXd prob_tmp = (theta * data_top).array();
  prob = (exp(prob_tmp)).matrix();
  MatrixXd norm_fact = prob.colwise().sum();
  for (i = 0; i < prob.rows(); i++) {
    prob.row(i) = (prob.row(i).array() / norm_fact.array()).matrix();
  }
  return prob;
}


double fine_tune::smx_cost(MatrixXd & prob) const{
  assert(prob.cols() == data_top.cols());
  double cost = 0;
  MatrixXd lbl = MatrixXd::Zero(data_top.cols(), n_class);
  for (int i = 0; i < lbl.rows(); i++) {
    lbl(i, data_lbl(i)) = 1;
  }
  MatrixXd cost_array = lbl * (log(prob.array())).matrix();
  cost = cost_array.trace();
  cost *= -1. / data_top.cols();
  return cost;
}


vector<unordered_map<string, MatrixXd> > fine_tune::smx_grad(MatrixXd & prob) const{
  assert(prob.cols() == data_top.cols());
  MatrixXd lbl = MatrixXd::Zero(data_top.cols(), n_class);
  for (int i = 0; i < lbl.rows(); i++) {
    lbl(i, data_lbl(i)) = 1;
  }
  lbl = lbl.transpose() - prob;
  MatrixXd grad = lbl * data_top.transpose();
  grad = (grad.array() * -1. / data_top.cols()).matrix();
  return grad;
}


void fine_tune::smx_nume_grad() {
  // Reload data? // TODO, not printing these shit.
  std::cout << get_worker_id() <<"printing" << std::endl;
  std::cout << data.rows() << std::endl;
  std::cout << data.cols() << std::endl;
  std::cout << get_worker_id() << "printing done" << std::endl;

  std::cout << "Check gradients computing" << std::endl;
  MatrixXd nume_grad = MatrixXd::Zero(smx_W.rows(), smx_W.cols());
  for (int i = 0; i < smx_W.rows(); i++) {
    for (int j = 0; j < smx_cols(); j++) {
      MatrixXd smx_w_nume1 = smx_W;
      MatrixXd smx_w_nume2 = smx_W;
      smx_W_nume1(i, j) += 1e-4;
      smx_W_nume2(i, j) -= 1e-4;
      MatrixXd prob1 = smx_prob(smx_W_nume1);
      MatrixXd prob2 = smx_prob(smx_W_nume2);
      nume_grad(i, j) = smx_cost(prob1) - smx_cost(prob2);
      nume_grad(i, j) /= 2 * 1e-4;
    }
  }
  prob = smx_prob(smx_W);
  grad = smx_grad(prob);
  std::cout << (grad - nume_grad).norm() / \
               (grad + nume_grad).norm() << std::endl;
  std::cout << "Values printed above should be less than 1e-9" << std::endl;
}


double fine_tune::fn_cost() const{
  double cost = 0;
  int n_lyr = WgtBias.size(); //
  MatrixXd data_lyr = data;
  for (int i = 0; i < n_lyr; i++) {
    data_lyr = acti_func(WgtBias[i].at("W1") * data_lyr + \
                         WgtBias[i].at("b1"));
  }
  if(data_top.rows() == 0){ // not initialize softmax
    smx_init();
  }
  data_top = data_lyr;
  cost = smx_cost(smx_prob(smx_W)); // TODO, add some asserts
  return cost;
}


vector<unordered_map<string, MatrixXd> > fine_tune::fn_stoc_grad(int idx) const{
  vector<unordered_map<string, MatrixXd> > WgtBiasGrad;
  int n_lyr = WgtBias.size();
  for (int i = 0; i < n_lyr; i++) {
    // initialize
    unordered_map<string, MatrixXd> grad_elem;
    grad_elem["W1"] = MatrixXd::Random(WgtBias[i].at("W1").rows(), WgtBias[i].at("W1").cols());
    grad_elem["b1"] = MatrixXd::Random(WgtBias[i].at("b1").rows(), WgtBias[i].at("b1").cols());
    WgtBiasGrad.push_back(grad_elem);
  } 
  WgtBiasGrad;
  unordered_map<int, MatrixXd> a;
  unordered_map<int, MatrixXd> z;
  unordered_map<int, MatrixXd> sigma; // grad elements
  a[1] = data.col(idx);
  for (int i = 1; i < n_lyr; i++) {
      // not i TODO
    z[i+1] = WgtBias[i].at("W1") * a[i] + WgtBias[i].at("b1");
    a[i+1] = acti_func(z[i+1]);
  }
  // TODO TODO
  auto top_grad = smx_grad(smx_prob(smx_W));
  vector<vector<double> > prob = smx_prob(smx_W);
  vector<double> I(prob.size(), 0);
  for (auto p : n_class) {
    I[p] = i;
  }

  sigma[n_lyr] = smx_W * (I - P);
  for (i = 0; i < n_lyr; i++) {
    
  }
  sigma[n_lyr] = -(a[1])

}



vector<unordered_map<string, MatrixXd> > fine_tune::fn_mibt_stoc_grad() const{
  vector<unordered_map<string, MatrixXd> > WgtBiasGrad;
  int n_lyr = WgtBias.size();
  for (int i = 0; i < n_lyr; i++) {
    // initialize
    unordered_map<string, MatrixXd> grad_elem;
    grad_elem["W1"] = MatrixXd::Random(WgtBias[i].at("W1").rows(), WgtBias[i].at("W1").cols());
    grad_elem["b1"] = MatrixXd::Random(WgtBias[i].at("b1").rows(), WgtBias[i].at("b1").cols());
    WgtBiasGrad.push_back(grad_elem);
  }
  for (auto l : n_lyr) {
    for (auto i: index_data) {
      
    }
  }
  return WgtBiasGrad;
}


} // namaspace paracel
