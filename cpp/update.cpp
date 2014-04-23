#include <eigen3/Eigen/Dense>
#include "proxy.hpp"
#include "paracel_types.hpp"

extern "C"{
  extern paracel::update_result ae_update;
}

// Eigen::MatrixXd seems not compatible with paracel
/*
Eigen::MatrixXd local_update(Eigen::MatrixXd a, Eigen::MatrixXd b){
  Eigen::MatrixXd r = a + b;
  return r;
}
*/

vector<double> local_update(vector<double> a, vector<double> b) {
  vector<double> r;
  for(int i = 0; i < (int)a.size(); ++i) {
    r.push_back(a[i] + b[i]);
  }
  return r;
 }     

paracel::update_result ae_update = paracel::update_proxy(local_update);


