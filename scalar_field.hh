#ifndef SCALAR_FIELD_HH
#define SCALAR_FIELD_HH

#include<Eigen/Dense>
#include"vector_field.hh"
#include"mesh.hh"

class Scalar_field 
{
   public:
   
   Eigen::VectorXd scalar_field_values;
   Eigen::VectorXd scalar_old_field_values;

   Scalar_field(int);

   void set_scalar_field_values(double);
   void set_old_scalar_field_values();

   Vector_field compute_gauss_gradient(std::vector<Cell> &listOfCells, std::vector<Face> &list_of_faces);
};

#endif