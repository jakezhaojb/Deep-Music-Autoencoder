#ifndef _A_E_HPP_
#define _A_E_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include "ps.hpp"
#include "utils.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;

namespace paracel{

class autoencoder: public paracel::paralg{

 public:
  autoencoder(paracel::Comm, string, string, string, vector<int>, int, string = "sgd", string = "sigmoid", int = 1, double = 0.01, bool = false, int = 0, bool = false, double = 0.001, double = 0.0001, double = 3., int = 1, int = 0, int = 0, bool = false, double = 0.30, double = 0.1); // TO BE COMPLETED
  virtual ~autoencoder();

  void downpour_sgd(int); // downpour stochastic gradient descent
  void distribute_bgd(int);          // conventional batch-gradient descent
  void downpour_sgd_mibt(int); // downpour stochastic gradient descent and mini-batch involved
  
  void local_parser(const vector<string> &, const char = ',', bool = false);
  void local_dump_Mat(const MatrixXd &, const string filename, const char = ',');
  void train(int);
  void train(); // top function
  void dump_mat(const MatrixXd &, const string) const;
  void dump_result(int) const;
  MatrixXd acti_func(const MatrixXd &) const;
  ArrayXXd acti_func_der(const MatrixXd &) const;
  vector<unordered_map<string, MatrixXd> > GetWgtBias() const;

  // init
  void ae_init(void);
  // compute cost function
  double ae_cost(int) const;
  // back-propogation batch gradient compute
  unordered_map<string, MatrixXd> ae_batch_grad(int) const;
  // back-propogation stochastic gradient compute
  unordered_map<string, MatrixXd> ae_stoc_grad(int, int) const;
  // BP with Mini-batch
  unordered_map<string, MatrixXd> ae_mibt_stoc_grad(int, vector<int>) const;

  // for DAE
  void corrupt_data();

  // compatinility of paracel and MatrixXd
  void _paracel_write(string key, MatrixXd & m);
  MatrixXd _paracel_read(string key, int r, int c);
  VectorXd _paracel_read(string key);
  void _paracel_bupdate(string key, MatrixXd & m);

  // IT SHOULD BE CLASS-INVARIANT!!!
  // conversion between Eigen::MatrixXd and std::vector
  MatrixXd vec_to_mat(const vector<vector<double> > &); // row ordered
  VectorXd vec_to_mat(const vector<double> &);  // column ordered
  MatrixXd vec_to_mat(const vector<double> &, int);  // column ordered
  MatrixXd vec_to_mat(const vector<double> &, int, int);  // column ordered
  vector<double> Mat_to_vec(const MatrixXd &);  // column ordered

 private:
  string input;  // where you store data over layers
  string output; // where you dump out results

 protected:
  int worker_id;
  int rounds;
  int n_lyr;  // number of hidden layers
  int mibt_size;
  int read_batch;
  int update_batch;
  string learning_method;
  string acti_func_type;
  bool debug = false;
  vector<double> loss_error;
  vector<unordered_map<string, MatrixXd> > WgtBias;
  MatrixXd data;
  vector< vector<double> > samples;
  vector<int> labels; // if necessary
  double lamb;            // weight decay
  double sparsity_param;    // sparse KL comparison
  double beta;              // sparse penalty
  double alpha;             // learning step size
  vector<int> hidden_size;
  int visible_size;
  vector<int> layer_size;  // combine hidden_size and layer_size together

  // for DAE
 private:
  bool corrupt;
  double dvt;  // deviation of Gaussion noise
  double foc;  // fraction of corrupted neurons 

}; // class

} // namespace paracel


#endif
