#include <string>
#include <vector>
#include <iostream>

#include <mpi.h>
#include <google/gflags.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
//#include <boost/filesystem>

#include "ae.hpp"
#include "utils.hpp"

using namespace boost::property_tree;

DEFINE_string(server_info, "host1:7777PARACELhost2:8888", "hosts name string of paracel-servers.\n");

DEFINE_string(cfg_file, "", "config json file with absolute path.\n");

std::vector<int> split(std::string & str, char sep = ','){
  std::vector<int> res;
  size_t en = 0, st = 0;
  while(1){
    en = str.find(sep, st);
    std::string s = str.substr(st, en - st);
    res.push_back(stod(s));
    if (en == str.npos) {
      break;
    }
    st = en + 1;
  }
  return res;
}


int main(int argc, char *argv[])
{
  paracel::main_env comm_main_env(argc, argv);
  paracel::Comm comm(MPI_COMM_WORLD);

  google::SetUsageMessage("[options]\n\t--server_info\n\t--cfg_file\n");
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  ptree pt;
  json_parser::read_json(FLAGS_cfg_file, pt);
  std::string output = pt.get<std::string>("output");
  std::string input = pt.get<std::string>("input");
  std::string learning_method = pt.get<std::string>("learning_method");
  std::string acti_func_type = pt.get<std::string>("acti_func_type");
  std::string _hidden_size = pt.get<std::string>("hidden_size");
  double alpha = pt.get<double>("alpha");
  double beta = pt.get<double>("beta");
  double lamb = pt.get<double>("lamb");
  double sparsity_param = pt.get<double>("sparsity_param");
  double dvt = pt.get<double>("deviation");
  double foc = pt.get<double>("frac_of_corrupt");
  int rounds = pt.get<int>("rounds");
  int limit_s = pt.get<int>("limit_s");
  int mibt_size = pt.get<int>("mibt_size");
  int visible_size = pt.get<int>("visible_size");
  int read_batch = pt.get<int>("read_batch");
  int update_batch = pt.get<int>("update_batch");
  bool corrupt = pt.get<bool>("corrupt");
  bool fine_tuning = pt.get<bool>("fine_tuning");

  // Processing the parsing
  vector<int> hidden_size = split(_hidden_size);
//  if(!boost::filesystem::exists(input))
//    boost::filesystem::create_directories(input);
//  if(!boost::filesystem::exists(output))
//    boost::filesystem::create_directories(output);

  {
    paracel::autoencoder ae_solver(comm, FLAGS_server_info, input, output, hidden_size, visible_size, learning_method, acti_func_type, rounds, alpha, false, limit_s,
              true, lamb, sparsity_param, beta, mibt_size, read_batch, update_batch, corrupt, dvt, foc, fine_tuning);
    ae_solver.train();
  }

  return 0;
}
