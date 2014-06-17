#ifndef _FINE_TN_HPP_
#define _FINE_TN_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include "ps.hpp"
#include "utils.hpp"
#include "ae.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;

namespace paracel{

class fine_tune: public autoencoder {

 public:
   fine_tune(paracel::Comm, string, string, string, vector<int>, int, vector<unordered_map<string, MatrixXd> >, string = "sgd", string = "sigmoid", int = 1, double = 0.01, bool = false, int = 0, bool = false, double = 0.001, double = 0.0001, double = 3., int = 1, int = 0, int = 0, int = 2); // TO BE COMPLETED
   virtual ~fine_tune();

   // softmax
   void smx_init();
   MatrixXd smx_prob(MatrixXd &) const; // probabilities of softmax classifier
   double smx_cost(MatrixXd &) const;
   MatrixXd smx_grad(MatrixXd &) const;
   void smx_nume_grad();

   // fine tuning the whole networks
   double fn_cost() const;
   vector<unordered_map<string, MatrixXd> > fn_stoc_grad(int) const;
   vector<unordered_map<string, MatrixXd> > fn_mibt_stoc_grad() const;
   void fn_downpour_sgd(); // downpour stochastic gradient descent
   void fn_distribute_bgd(); // conventional batch-gradient descent
   void fn_downpour_sgd_mibt(); // downpour stochastic gradient descent and mini-batch involved

 private:
   //vector<unordered_map<string, MatrixXd> > WgtBias;
   string input;
   string output;
   MatrixXd smx_W;
   MatrixXd data_top;
   VectorXd data_lbl;
   int n_class;  // TODO

};

} // namespace paracel

#endif
