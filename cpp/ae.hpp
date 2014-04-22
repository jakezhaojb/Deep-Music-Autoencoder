#ifndef _A_E_HPP
#define _A_E_HPP_

#include <string>
#include <vector>
#include <cstdlib>
#include </mfs/user/zhaojunbo/Eigen/Dense>
#include "ps.hpp"
#include "utils.hpp"
#include "sparse_autoencoder.h"

namespace paracel{

class autoencoder: public paracel::paralg{

 public:
  autoencoder(paracel::Comm, std::string, std::string, std::string, std::vector<int>, std::vector<int>, std::string = "sgd", int = 1, double = 0.01, bool = false, int = 0, bool = false, double = 0.001, double = 0.0001, double = 3., int = 1); // TO BE COMPLETED
  virtual ~autoencoder();

  void downpour_sgd(int); // downpour stochastic gradient descent
  void distribute_bgd(int);          // conventional batch-gradient descent
  void downpour_sgd_mibt(int); // downpour stochastic gradient descent and mini-batch involved
  
  void local_parser(const std::vector<std::string> &, const char = ',', bool = false);
  void train(int);
  void train(); // top function
  void dump_result(int);

  // Included in sparse_autoencoder.h
  // compute cost function
  double ae_cost(int) const;
  // back-propogation batch gradient compute
  std::map<std::string, Eigen::MatrixXd> ae_batch_grad(int) const;
  // back-propogation stochastic gradient compute
  std::map<std::string, Eigen::MatrixXd> ae_stoc_grad(int, int) const;
  // BP with Mini-batch
  std::map<std::string, Eigen::MatrixXd> ae_mibt_stoc_grad(int, std::vector<int>) const;

 private:
  std::string input;
  int worker_id;
  int rounds;
  int n_lyr;
  int mibt_size;
  std::string learning_method;
  bool debug = false;
  std::vector<double> loss_error;
  std::vector<std::map<std::string, Eigen::MatrixXd> > WgtBias;
  Eigen::MatrixXd data;
  std::vector< std::vector<double> > samples;
  std::vector<int> labels; // if necessary
  double lamb;            // weight decay
  double sparsity_param;    // sparse KL comparison
  double beta;              // sparse penalty
  double alpha;             // learning step size
  std::vector<int> hidden_size;
  std::vector<int> visible_size;

}; // class

} // namespace paracel

// convert Eigen::MatrixXd to std::vector<double>
Eigen::MatrixXd vec_to_mat(std::vector<std::vector<double> > &);
Eigen::VectorXd vec_to_mat(std::vector<double> &);
std::vector<double> Vec_to_vec(Eigen::MatrixXd &);

#endif
