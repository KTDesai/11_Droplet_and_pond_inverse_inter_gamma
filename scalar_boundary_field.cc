#include "scalar_boundary_field.hh"
#include<Eigen/Dense>

Scalar_boundary_field::Scalar_boundary_field(std::vector<Boundary> list_of_boundaries, std::vector<std::string> boundary_types, std::vector<double> scalar_boundary_values)
{
    boundary_indices.resize(list_of_boundaries.size());
    boundary_names.resize(list_of_boundaries.size());
    boundary_type.resize(list_of_boundaries.size());
    scalar_boudndary_field_values.resize(list_of_boundaries.size()); 
    
    for(int i = 0; i<list_of_boundaries.size(); i++)
    {
        boundary_indices[i] = list_of_boundaries[i].get_boundary_index();
        boundary_names[i] = list_of_boundaries[i].return_boundary_name();
        boundary_type[i] = boundary_types[i];
        scalar_boudndary_field_values[i] =  scalar_boundary_values[i];
    }
}


std::vector<double> Scalar_boundary_field::get_scalar_boundary_field_values()
{
    return scalar_boudndary_field_values;
}

std::string Scalar_boundary_field::get_scalar_boundary_name (int boundary_index)
{
    return boundary_names[boundary_index];
}

std::string Scalar_boundary_field::get_scalar_boundary_type (int boundary_index)
{
    return boundary_type[boundary_index];
}