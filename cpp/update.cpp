#include <string>
#include <map>
#include <Eigen/Dense>
#include "proxy.hpp"
#include "paracel_types.hpp"

using Eigen::MatrixXd;
using std::string;

extern "C"{
  extern paracel::update_result ae_update;
}

MatrixXd local_update(MatrixXd a, MatrixXd b){
  r = a + b;
  return r;
}

paracel::update_result ae_update = paracel::update_proxy(local_update);


