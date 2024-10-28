#ifndef VECTOR_BOUNDARY_FIELD_HH
#define VECTOR_BOUNDARY_FIELD_HH

#include<Eigen/Dense>
#include<string>
#include"vector_field.hh"
#include"mesh.hh"

class Vector_boundary_field 
{
  public:

  std::vector<double> boundary_indices;
  std::vector<std::string> boundary_names;
  std::vector<std::string> boundary_type;
  std::vector<std::vector<double>> vector_boundary_field_values;

  Vector_boundary_field(std::vector<Boundary>, std::vector<std::string>, std::vector<std::vector<double>>);

  std::vector<std::vector<double>> get_vector_boundary_field_values();
  std::string get_vector_boundary_name(int);
  std::string get_vector_boundary_type(int);
};

#endif