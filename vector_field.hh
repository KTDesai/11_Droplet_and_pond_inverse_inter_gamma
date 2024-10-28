#ifndef VECTOR_FIELD_HH
#define VECTOR_FIELD_HH
#include<Eigen/Dense>
#include "vector.hh"

class Vector_field
{
  public:

  Eigen::VectorXd vector_field_values_x; 
  Eigen::VectorXd vector_field_values_y;
  Eigen::VectorXd vector_field_values_z;

  Eigen::VectorXd vector_old_field_values_x; 
  Eigen::VectorXd vector_old_field_values_y;
  Eigen::VectorXd vector_old_field_values_z;

  Vector_field(int);

  void set_vector_field_values(Vector obj);
  void set_old_vector_field_values();
  void add_to_vector_field_values(Vector obj, int);

};
#endif