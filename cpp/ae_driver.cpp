#include <string>
#include <iostream>

#include <mpi.h>
#include <google/gflags.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "ae.hpp"
#include <utils.hpp>

using namespace boost::property_tree;

DEFINE_string(server_info, "host1:7777PARACELhost2:8888", "hosts name string of paracel-servers.\n");

DEFINE_string(cfg_file, "", "config json file with absolute path.\n");

int main(int argc, const char *argv[])
{
  paracel::main_env comm_main_env(argc, argv);
  paracel::Comm comm(MPI_COMM_WORLD);

  google::SetUsageMessage("[options]\n\t--server_info\n\t--cfg_file\n");
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  ptree pt;
  json_parser::read_json(FLAGS_cfg_file, pt);
  std::string input = pt.get<std::string>("input");
  std::string output = pt.get<std::string>("output");
  double alpha = pt.get<double>("alpha");
  double beta = pt.get<double>("beta");
  double lamb = pt.get<double>("lamb");
  double sparsity_param = pt.get<double>("sparsity_param");
  int rounds = pt.get<int>("rounds");
  int limit_s = pt.get<int>("limit_s");
  int mibt_size = pt.get<int>("mibt_size");
  int n_lyr = pt.get<int>("n_lyr");
    
  // AVAILABLE??
  vector<int> hidden_size = pt.get<vector<int> >("hidden_size");
  vector<int> visible_size = pt.get<vector<int> >("visible_size");
  
  paracel::autoencoder ae_solver(comm, FLAGS_server_info, input, output, "dsgd", rounds, alpha, false, limit_s,
            true, hidden_size, visible_size, lamb, sparsity_param, beta, mibt_size);
  ae_solver.train();

  return 0;
}
