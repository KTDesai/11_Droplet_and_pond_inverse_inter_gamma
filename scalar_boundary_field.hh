#ifndef SCALAR_BOUNDARY_FIELD_HH
#define SCALAR_BOUNDARY_FIELD_HH

#include<Eigen/Dense>
#include<string>
#include"vector_field.hh"
#include"mesh.hh"

class Scalar_boundary_field 
{
  public:

  std::vector<double> boundary_indices;
  std::vector<std::string> boundary_names;
  std::vector<std::string> boundary_type;
  std::vector<double> scalar_boudndary_field_values;

  Scalar_boundary_field(std::vector<Boundary>, std::vector<std::string>, std::vector<double>);

  std::vector<double> get_scalar_boundary_field_values();
  std::string get_scalar_boundary_name(int);
  std::string get_scalar_boundary_type(int);
};

#endif