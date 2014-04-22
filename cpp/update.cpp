#include </mfs/user/zhaojunbo/Eigen/Dense>
#include "proxy.hpp"
#include "paracel_types.hpp"

extern "C"{
  extern paracel::update_result ae_update;
}

Eigen::MatrixXd local_update(Eigen::MatrixXd a, Eigen::MatrixXd b){
  Eigen::MatrixXd r = a + b;
  return r;
}

paracel::update_result ae_update = paracel::update_proxy(local_update);


