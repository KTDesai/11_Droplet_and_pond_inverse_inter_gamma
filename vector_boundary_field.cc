#include "vector_boundary_field.hh"
#include<Eigen/Dense>

Vector_boundary_field::Vector_boundary_field(std::vector<Boundary> list_of_boundaries, std::vector<std::string> boundary_types, std::vector<std::vector<double>> vector_boundary_values)
{
    boundary_indices.resize(list_of_boundaries.size());
    boundary_names.resize(list_of_boundaries.size());
    boundary_type.resize(list_of_boundaries.size());
    vector_boundary_field_values.resize(list_of_boundaries.size(),std::vector<double> (3)); 
    
    for(int i = 0; i<list_of_boundaries.size(); i++)
    {
        boundary_indices[i] = list_of_boundaries[i].get_boundary_index();
        boundary_names[i] = list_of_boundaries[i].return_boundary_name();
        boundary_type[i] = boundary_types[i];
        
      for(int j=0; j<3; j++)
      {  
        vector_boundary_field_values[i][j] =  vector_boundary_values[i][j];
      }  
    }
}


std::vector<std::vector<double>> Vector_boundary_field::get_vector_boundary_field_values()
{
    return vector_boundary_field_values;
}

std::string Vector_boundary_field::get_vector_boundary_name (int boundary_index)
{
    return boundary_names[boundary_index];
}

std::string Vector_boundary_field::get_vector_boundary_type (int boundary_index)
{
    return boundary_type[boundary_index];
}