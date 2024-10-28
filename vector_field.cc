#include "vector_field.hh"

Vector_field::Vector_field(int n_cells)
{
    vector_field_values_x.resize(n_cells);
    vector_field_values_y.resize(n_cells);
    vector_field_values_z.resize(n_cells);

    vector_old_field_values_x.resize(n_cells);
    vector_old_field_values_y.resize(n_cells);
    vector_old_field_values_y.resize(n_cells);

    vector_field_values_x.setZero();
    vector_field_values_y.setZero();
    vector_field_values_z.setZero();

    vector_old_field_values_x.setZero();
    vector_old_field_values_y.setZero();
    vector_old_field_values_z.setZero();
}

void Vector_field::set_vector_field_values(Vector obj)
{
   Vector const_values = obj;
 
    for(int i = 0; i<vector_field_values_x.rows(); i++)
    { 
        vector_field_values_x(i) = const_values[0] ;
        vector_field_values_y(i) = const_values[1] ;
        vector_field_values_z(i) = const_values[2] ;
    }
}

void Vector_field::set_old_vector_field_values()
{

    vector_old_field_values_x = vector_field_values_x;
    vector_old_field_values_y = vector_field_values_y;
    vector_old_field_values_z = vector_field_values_z;

}

void Vector_field::add_to_vector_field_values(Vector obj, int index)
{
    Vector const_values = obj;
     
    vector_field_values_x(index) = vector_field_values_x(index) + const_values[0] ;
    vector_field_values_y(index) = vector_field_values_y(index) + const_values[1] ;
    vector_field_values_z(index) = vector_field_values_z(index) + const_values[2] ;

}

