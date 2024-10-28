#include "scalar_field.hh"
#include<Eigen/Dense>

Scalar_field::Scalar_field(int number_of_cells)
{
    scalar_field_values.resize(number_of_cells);
    scalar_old_field_values.resize(number_of_cells);

    scalar_field_values.setZero();
    scalar_old_field_values.setZero();
}


void Scalar_field::set_scalar_field_values(double a)
{
   for(int i = 0; i<scalar_field_values.size(); i++)
   {
         scalar_field_values(i) = a ;
   }

}

void Scalar_field::set_old_scalar_field_values()
{
  scalar_old_field_values = scalar_field_values;
}


Vector_field Scalar_field::compute_gauss_gradient(std::vector<Cell> &list_of_cells, std::vector<Face> &list_of_faces)
{
    Vector_field grad_p(list_of_cells.size());

        for(int i = 0; i < list_of_faces.size(); i++)
        {   
            Vector grad_o(0.0, 0.0, 0.0);
            Vector grad_n(0.0, 0.0, 0.0);

            int face_index = list_of_faces[i].get_face_index();

            int owner_index = list_of_faces[i].get_owner_index();
            int neighbour_index = list_of_faces[i].get_neighbour_index();

            double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
          
            double owner_cell_vol = list_of_cells[owner_index].get_cell_volume();
            double neighbour_cell_vol = list_of_cells[neighbour_index].get_cell_volume();

            Vector face_normal = list_of_faces[i].get_face_normal();

            if(list_of_faces[i].get_is_boundary_face() == true)
            {
                grad_o = (face_normal * scalar_field_values(owner_index) * interpolation_factor);
                grad_o = grad_o/owner_cell_vol; 

                double pressure_face = scalar_field_values(owner_index);

                grad_p.vector_field_values_x (owner_index) = grad_p.vector_field_values_x (owner_index) + grad_o[0];
                grad_p.vector_field_values_y (owner_index) = grad_p.vector_field_values_y (owner_index) + grad_o[1]; 
                grad_p.vector_field_values_z (owner_index) = grad_p.vector_field_values_z (owner_index) + grad_o[2];   
            }

            else
            {
                double pressure_face = (interpolation_factor * scalar_field_values(owner_index)) + ((1 - interpolation_factor) * scalar_field_values(neighbour_index));
 
                grad_o = (face_normal * pressure_face);

           
                grad_n = (face_normal * pressure_face * -1);

                 grad_o = grad_o/owner_cell_vol;
                 grad_n = grad_n/neighbour_cell_vol;

                grad_p.vector_field_values_x (owner_index) = grad_p.vector_field_values_x (owner_index) + grad_o[0];
                grad_p.vector_field_values_y (owner_index) = grad_p.vector_field_values_y (owner_index) + grad_o[1]; 
                grad_p.vector_field_values_z (owner_index) = grad_p.vector_field_values_z (owner_index) + grad_o[2];

                grad_p.vector_field_values_x (neighbour_index) = grad_p.vector_field_values_x (neighbour_index) + grad_n[0];
                grad_p.vector_field_values_y (neighbour_index) = grad_p.vector_field_values_y (neighbour_index) + grad_n[1]; 
                grad_p.vector_field_values_z (neighbour_index) = grad_p.vector_field_values_z (neighbour_index) + grad_n[2];

            }
         }
    
    return grad_p;
}
