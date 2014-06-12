#include "fine_tn.hpp" 
#include <cmath>
#include <cassert>


namespace paracel{

fine_tune::fine_tune(paracel::Comm comm, string hosts_dct_str,
          string _input, string _output, vector<int> _hidden_size,
          int _visible_size, string method, string _acti_func_type, 
          int _rounds, double _alpha, bool _debug, int limit_s, 
          bool ssp_switch, double _lamb, double _sparsity_param, 
          double _beta, int _mibt_size, int _read_batch, int _update_batch,
          int _n_class, vector<unordered_map<string, MatrixXd> > _WgtBias):
      autoencoder(comm, hosts_dct_str, _input, _output, 
                  _hidden_size, _visible_size, method, 
                  _acti_func_type, _rounds, _alpha, _debug, 
                  limit_s, ssp_switch, _lamb, _sparsity_param, 
                  _beta, _mibt_size, _read_batch, _update_batch),
      WgtBias(_WgtBias),
      n_class(_n_class) {}

fine_tune::~fine_tune {}


double fine_tune::smx_cost(vector<vector<double> > prob) const{
  assert(prob.size() == data.cols());
  assert(prob.size() == labels.size())
  double cost = 0;
  for (int i = 0; i < data.cols(); i++) {
    for (int j = 0; j < n_class; j++) {
      prob[i][j] = int(j == labels[i]) * log(prob[i][j]);
      cost += prob[i][j];
    }
  }
  return cost;
}


vector<unordered_map<string, MatrixXd>> smx_grad(vector<vector<double> > prob) const{
  vector<unordered_map<string, MatrixXd>> grad;
  ArrayXXd tmp_grad_W, tmp_grad_b;
  for (int i = 0; i < n_class; i++) {
    unordered_map<string, MatrixXd> grad_lbl;
    grad_lbl["W"] = MatrixXd::Zero(data.rows(), 1);
    grad_lbl["b"] = MatrixXd::Zero(1, 1);
    for (int j = 0; j < data.cols(); j++) {
      tmp_grad_W += data.col(j).array() * (int(labels[j] == i) - prob[j][i]);
      tmp_grad_b += int(labels[j] == i) - prob[j][i];
    }
    grad_lbl["W"] = (tmp_grad_W * -1 / data.cols()).matrix();
    grad_lbl["b"] = (tmp_grad_b * -1 / data.cols()).matrix();
    grad.push_back(grad_lbl);
  }
  return grad;
}


vector<vector<double> > fine_tune::smx_prob(vector<MatrixXd> W, vector<MatrixXd> b) const{
  vector<vector<double> > prob;
  for (int i = 0; i < data.cols(); i++) {
    vector<double> prob_elem(n_class, 0);
    double prob_sum = 0;
    for (int j = 0; j < n_class; j++) {
      prob_elem[i] = exp(W[j].transpose() * data.col(i) + b[j]);
      sum += prob_elem[i];
    }
    for (int j = 0; j < n_class; j++) {
      prob_elem[i] /= sum;
    }
    prob.push_back(double(prob_elem(0, 0)));
  } // traverse data
  return prob;
}


unordered_map<string, MatrixXd> smx_nume_grad() const{
  MatrixXd epsilon = MatrixXd::Identity(data.rows(), data.rows());
  epsilon = (epsilon.array() * 1e-4).matrix();
  grad = smx_grad(smx_prob());
  for (int i = 0; i < n_class; i++) {
    vector<MatrixXd> W_u = smx_W;
    vector<MatrixXd> W_d = smx_W;
    vector<MatrixXd> b_u = smx_b;
    vector<MatrixXd> b_d = smx_b;
    vector<unordered_map<string, MatrixXd>> nume_grad;
    nume_grad["W"] = MatrixXd::Zero(smx_W.rows(), smx_W.cols());
    nume_grad["b"] = MatrixXd::Zero(smx_b.rows(), smx_b.cols());
    //  nume_grad for W
    for (int j = 0; j < nume_grad["W"].rows(); j++) {
      W_u[i] = smx_W[i].colwise() + epsilon.col(j);
      W_d[i] = smx_W[i].colwise() - epsilon.col(j);
      nume_grad["W"](j) = compute_cost(smx_prob(W_u)) - \
                          compute_cost(smx_prob(W_d));
      nume_grad["W"](j) /= 2 * 1e-4; 
    }
    // nume_grad for b
    for (int j = 0; j < nume_grad["b"].rows(); j++) {
      b_u[i] = smx_b[i].colwise() + epsilon.col(j);
      b_d[i] = smx_b[i].colwise() - epsilon.col(j);
      nume_grad["b"](j) = compute_cost(smx_prob(b_u)) - \
                          compute_cost(smx_prob(b_d));
      nume_grad["b"](j) /= 2 * 1e-4; 
    }
    std::cout << (grad[i].at("W") - numegrad["W"]).norm() / \
          (grad[i].at("W") + numegrad["W"]).norm() << std::endl;
  }
  std::cout << "Values printed above should be less than 1e-9" << std::endl;
}

} // namaspace paracel
