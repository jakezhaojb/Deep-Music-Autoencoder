#ifndef _A_E_HPP_
#define _A_E_HPP_

#include <string>
#include <vector>
#include <stdlib.h>
#include </mfs/user/zhaojunbo/Eigen/Dense>
#include "ps.hpp"
#include "utils.hpp"
#include "sparse_autoencoder.h"

using namespace std;
using namespace Eigen;

namespace paracel{

class autoencoder: public paracel::paralg{

 public:
  autoencoder(paracel::Comm, string, string, string, string = "sgd", int = 1, double = 0.01, bool = false, int = 0, bool = false, vector<int>, vector<int>, double = 0.001, double = 0.0001, double = 3., int = 1); // TO BE COMPLETED
  virtual ~autoencoder();

  void downpour_sgd(int); // downpour stochastic gradient descent
  void distribute_bgd(int);          // conventional batch-gradient descent
  void downpour_sgd_mibt(int) // downpour stochastic gradient descent and mini-batch involved
  
  void local_parser(const vector<string> &, const char, bool = false);
  void train(int);
  void train(); // top function
  void dump_result(int);

  // Included in sparse_autoencoder.h
  // compute cost function
  double ae_cost(int) const;
  // back-propogation batch gradient compute
  map<string, MatrixXd> ae_batch_grad(int) const;
  // back-propogation stochastic gradient compute
  map<string, MatrixXd> ae_stoc_grad(int, int) const;
  // BP with Mini-batch
  std::map<string, MatrixXd> ae_mibt_stoc_grad(int, vector<int>) const;

 private:
  string input;
  int worker_id;
  int rounds;
  int n_lyr;
  int mibt_size;
  string learning_method;
  bool debug = false;
  std::vector<double> loss_error;
  std::vector<std::map<string, MatrixXd>> WgtBias;
  MatrixXd data;
  vector< vector<double> > samples;
  vector<int> labels; // if necessary
  double lamb;            // weight decay
  double sparsity_param;    // sparse KL comparison
  double beta;              // sparse penalty
  double alpha;             // learning step size
  vector<int> hidden_size;
  vector<int> visible_size;

}; // class

} // namespace paracel

// convert Eigen::MatrixXd to std::vector<double>
MatrixXd vec_to_mat(vector<vector<double> > &);
VectorXd vec_to_mat(vector<double> &);
vector<double> Vec_to_vec(MatrixXd &);

#endif
