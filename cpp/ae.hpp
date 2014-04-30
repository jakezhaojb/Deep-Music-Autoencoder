#ifndef _A_E_HPP
#define _A_E_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include "ps.hpp"
#include "utils.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// global var
VectorXd g_rho;  // for sparse penalty

namespace paracel{

class autoencoder: public paracel::paralg{

 public:
  autoencoder(paracel::Comm, string, string, string, vector<int>, int, string = "sgd", int = 1, double = 0.01, bool = false, int = 0, bool = false, double = 0.001, double = 0.0001, double = 3., int = 1, int = 0, int = 0); // TO BE COMPLETED
  virtual ~autoencoder();

  void downpour_sgd(int); // downpour stochastic gradient descent
  void distribute_bgd(int);          // conventional batch-gradient descent
  void downpour_sgd_mibt(int); // downpour stochastic gradient descent and mini-batch involved
  
  void local_parser(const vector<string> &, const char = ',', bool = false);
  void local_dump_Mat(const MatrixXd &, const string filename, const char = ',');
  void train(int);
  void train(); // top function
  void dump_mat(const MatrixXd &, const string);
  void dump_result(int);
  MatrixXd acti_fun(const MatrixXd &, string = "sigmoid") const;

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

  // compatinility of paracel and MatrixXd
  void _paracel_write(string key, MatrixXd & m);
  MatrixXd _paracel_read(string key, int r, int c);
  VectorXd _paracel_read(string key);
  void _paracel_bupdate(string key, MatrixXd & m);

  // IT SHOULD BE CLASS-INVARIANT!!!
  // conversion between Eigen::MatrixXd and std::vector
  MatrixXd vec_to_mat(vector<vector<double> > &); // row ordered
  VectorXd vec_to_mat(vector<double> &);  // column ordered
  MatrixXd vec_to_mat(vector<double> &, int);  // column ordered
  MatrixXd vec_to_mat(vector<double> &, int, int);  // column ordered
  vector<double> Mat_to_vec(MatrixXd &);  // column ordered

 private:
  string input;  // where you store data over layers
  int worker_id;
  int rounds;
  int n_lyr;  // number of hidden layers
  int mibt_size;
  int read_batch;
  int update_batch;
  string learning_method;
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

}; // class

} // namespace paracel


#endif
