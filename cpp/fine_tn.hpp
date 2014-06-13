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

   double smx_cost(vector<vector<double> >) const;
   void smx_init();
   vector<vector<double> > smx_prob(vector<MatrixXd>, vector<MatrixXd>) const; // probabilities of softmax classifier
   vector<unordered_map<string, MatrixXd>> smx_grad(vector<vector<double> >) const;
   void smx_nume_grad() const;

   void smx_ae_downpour_sgd(int); // downpour stochastic gradient descent
   void smx_ae_distribute_bgd(int);          // conventional batch-gradient descent
   void smx_ae_downpour_sgd_mibt(int); // downpour stochastic gradient descent and mini-batch involved
   void map_label();

 private:
   //vector<unordered_map<string, MatrixXd> > WgtBias;
   string input;
   string output;
   vector<MatrixXd> smx_W;
   vector<MatrixXd> smx_b; 
   int n_class;  // TODO

};

} // namespace paracel

#endif
