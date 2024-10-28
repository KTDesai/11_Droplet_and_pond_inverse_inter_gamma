#include "pv_coupling.hh"
#include<iostream>
#include<fstream>
#include <vector>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <Eigen/Sparse>

PV_coupling::PV_coupling(int no_of_cells, double nu_value, double initial_pressure, double initial_alpha, double initial_rho, double initial_mu, Vector initial_velocity): nu(no_of_cells), velocity(no_of_cells), pressure(no_of_cells), alpha(no_of_cells), rho(no_of_cells), mu(no_of_cells)
{
    nu.set_scalar_field_values(nu_value);
    velocity.set_vector_field_values(initial_velocity);
    pressure.set_scalar_field_values(initial_pressure);
    alpha.set_scalar_field_values(initial_alpha);
    rho.set_scalar_field_values(initial_rho);
    mu.set_scalar_field_values(initial_mu);
    
    velocity_a_matrix_rate_of_change.resize(no_of_cells, no_of_cells);
    velocity_a_matrix_diffusion.resize(no_of_cells, no_of_cells);
    velocity_a_matrix_convection.resize(no_of_cells, no_of_cells);
    velocity_a_matrix_combined.resize(no_of_cells, no_of_cells);
    

    velocity_b_vector_rate_of_change_ux.resize(no_of_cells);
    velocity_b_vector_rate_of_change_uy.resize(no_of_cells);
    velocity_b_vector_source_ux.resize(no_of_cells);
    velocity_b_vector_source_uy.resize(no_of_cells);
    velocity_b_vector_diffusion_ux.resize(no_of_cells);
    velocity_b_vector_diffusion_uy.resize(no_of_cells);
    velocity_b_vector_convection_ux.resize(no_of_cells);
    velocity_b_vector_convection_uy.resize(no_of_cells);
    velocity_b_vector_combined_ux.resize(no_of_cells);
    velocity_b_vector_combined_uy.resize(no_of_cells);

   velocity_a_matrix_rate_of_change.setZero();
   velocity_a_matrix_diffusion.setZero();
   velocity_a_matrix_convection.setZero();
   velocity_a_matrix_combined.setZero();

   velocity_b_vector_rate_of_change_ux.setZero();
   velocity_b_vector_rate_of_change_uy.setZero();
   velocity_b_vector_source_ux.setZero();
   velocity_b_vector_source_uy.setZero();
   velocity_b_vector_diffusion_ux.setZero();
   velocity_b_vector_diffusion_uy.setZero();
   velocity_b_vector_convection_ux.setZero();
   velocity_b_vector_convection_uy.setZero();
   velocity_b_vector_combined_ux.setZero();
   velocity_b_vector_combined_uy.setZero();

    pressure_a_matrix_rate_of_change.resize(no_of_cells, no_of_cells);
    pressure_a_matrix_diffusion.resize(no_of_cells, no_of_cells);
    pressure_a_matrix_convection.resize(no_of_cells, no_of_cells);
    pressure_a_matrix_combined.resize(no_of_cells, no_of_cells);

    pressure_b_vector_rate_of_change.resize(no_of_cells);
    pressure_b_vector_source.resize(no_of_cells);
    pressure_b_vector_diffusion.resize(no_of_cells);
    pressure_b_vector_convection.resize(no_of_cells);
    pressure_b_vector_combined.resize(no_of_cells);

   pressure_a_matrix_rate_of_change.setZero();
   pressure_a_matrix_diffusion.setZero();
   pressure_a_matrix_convection.setZero();
   pressure_a_matrix_combined.setZero();

   pressure_b_vector_rate_of_change.setZero();
   pressure_b_vector_source.setZero();
   pressure_b_vector_diffusion.setZero();
   pressure_b_vector_convection.setZero();
   pressure_b_vector_combined.setZero();

    alpha_a_matrix_rate_of_change.resize(no_of_cells, no_of_cells);
    alpha_a_matrix_convection.resize(no_of_cells, no_of_cells);
    alpha_a_matrix_combined.resize(no_of_cells, no_of_cells);

    alpha_b_vector_rate_of_change.resize(no_of_cells);
    alpha_b_vector_source.resize(no_of_cells);
    alpha_b_vector_convection.resize(no_of_cells);
    alpha_b_vector_combined.resize(no_of_cells);

   alpha_a_matrix_rate_of_change.setZero();
   alpha_a_matrix_convection.setZero();
   alpha_a_matrix_combined.setZero();

   alpha_b_vector_rate_of_change.setZero();
   alpha_b_vector_source.setZero();
   alpha_b_vector_convection.setZero();
   alpha_b_vector_combined.setZero();

}

void PV_coupling::velocity_reset_rate_of_change()
{
    velocity_a_matrix_rate_of_change.setZero();
    velocity_b_vector_rate_of_change_ux.setZero();
    velocity_b_vector_rate_of_change_uy.setZero();
}

void PV_coupling::velocity_reset_source()
{
    velocity_b_vector_source_ux.setZero();
    velocity_b_vector_source_uy.setZero();
}

void PV_coupling::velocity_reset_diffusion()
{
    velocity_a_matrix_diffusion.setZero();
    velocity_b_vector_diffusion_ux.setZero();
    velocity_b_vector_diffusion_uy.setZero();
}

void PV_coupling::velocity_reset_convection()
{
    velocity_a_matrix_convection.setZero();
    velocity_b_vector_convection_ux.setZero();
    velocity_b_vector_convection_uy.setZero();

}

void PV_coupling::velocity_reset_combined()
{
    velocity_a_matrix_combined.setZero();
    velocity_b_vector_combined_ux.setZero();
    velocity_b_vector_combined_uy.setZero();
}

void PV_coupling::velocity_update_a_matrix_rate_of_change(double a, int index)
{
   velocity_a_matrix_rate_of_change(index, index) = velocity_a_matrix_rate_of_change(index, index) + a ;         
}

void PV_coupling::velocity_update_b_vector_rate_of_change_ux(double a, int index)
{
    velocity_b_vector_rate_of_change_ux(index) = velocity_b_vector_rate_of_change_ux(index) + a;
}

void PV_coupling::velocity_update_b_vector_rate_of_change_uy(double a, int index)
{
    velocity_b_vector_rate_of_change_uy(index) = velocity_b_vector_rate_of_change_uy(index) + a;
}

void PV_coupling::velocity_update_b_vector_source_ux(double a, int index)
{
    velocity_b_vector_source_ux(index) = velocity_b_vector_source_ux(index) + a;
}

void PV_coupling::velocity_update_b_vector_source_uy(double a, int index)
{
    velocity_b_vector_source_uy(index) = velocity_b_vector_source_uy(index) + a;
}

void PV_coupling::velocity_update_a_matrix_diffusion(double a, int row_index, int column_index)
{ 
    velocity_a_matrix_diffusion(row_index, column_index) = velocity_a_matrix_diffusion(row_index, column_index) + a;                
}                                                                         

void PV_coupling::velocity_update_b_vector_diffusion_ux(double a, int index)
{
    velocity_b_vector_diffusion_ux(index) = velocity_b_vector_diffusion_ux(index) + a;           
}

void PV_coupling::velocity_update_b_vector_diffusion_uy(double a, int index)
{
    velocity_b_vector_diffusion_uy(index) = velocity_b_vector_diffusion_uy(index) + a;           
}

void PV_coupling::velocity_update_a_matrix_convection(double a, int row_index, int column_index)
{ 
    velocity_a_matrix_convection(row_index, column_index) = velocity_a_matrix_convection(row_index, column_index) + a;                 
}                                                                         

void PV_coupling::velocity_update_b_vector_convection_ux(double a, int index)
{
    velocity_b_vector_convection_ux(index) = velocity_b_vector_convection_ux(index) + a;           
}

void PV_coupling::velocity_update_b_vector_convection_uy(double a, int index)
{
    velocity_b_vector_convection_uy(index) = velocity_b_vector_convection_uy(index) + a;           
}

void PV_coupling::velocity_combine_a_matrices()
{
    for(int i = 0; i<velocity_a_matrix_combined.rows(); i++)
    {
        for(int j=0; j<velocity_a_matrix_combined.cols(); j++)
        {
            velocity_a_matrix_combined(i,j) = velocity_a_matrix_rate_of_change(i,j) + velocity_a_matrix_diffusion(i,j) + velocity_a_matrix_convection(i,j);
        }
    }

}

void PV_coupling::velocity_combine_b_matrices()
{
    for(int k =0 ; k <velocity_b_vector_combined_ux.size(); k++)
    {
        velocity_b_vector_combined_ux(k) = velocity_b_vector_rate_of_change_ux(k) + velocity_b_vector_source_ux(k) + velocity_b_vector_diffusion_ux(k) + velocity_b_vector_convection_ux(k) ;
    }

    for(int k =0 ; k <velocity_b_vector_combined_ux.size(); k++)
    {
        velocity_b_vector_combined_uy(k) = velocity_b_vector_rate_of_change_uy(k) + velocity_b_vector_source_uy(k) + velocity_b_vector_diffusion_uy(k) + velocity_b_vector_convection_uy(k) ;
    }
}

// Momentum Predictor - Temporal Derivitive Term Discretization 
 void PV_coupling::velocity_compute_rate_of_change_matrix(std::vector<Cell> &list_of_cells, double &delta_time)                                                               
 {
      velocity_reset_rate_of_change();  

      for(int i = 0; i<list_of_cells.size(); i++)
      {  
            int cell_index = list_of_cells[i].get_cell_index();
            double cell_vol = list_of_cells[cell_index].get_cell_volume();
            double velocity_x_old = velocity.vector_field_values_x(cell_index);
            double velocity_y_old = velocity.vector_field_values_y(cell_index);
            double rho_new = rho.scalar_field_values(cell_index);
            double rho_old = rho.scalar_old_field_values(cell_index);

            double rate_of_change_diagonal_contribution = (cell_vol * rho_new)/delta_time;
            double rate_of_change_source_contribution_ux = (cell_vol * rho_old * velocity_x_old)/ delta_time;
            double rate_of_change_source_contribution_uy = (cell_vol * rho_old * velocity_y_old)/ delta_time;

             velocity_update_a_matrix_rate_of_change(rate_of_change_diagonal_contribution, cell_index);
             velocity_update_b_vector_rate_of_change_ux(rate_of_change_source_contribution_ux, cell_index);
             velocity_update_b_vector_rate_of_change_uy(rate_of_change_source_contribution_uy, cell_index);
      }
 } 

// Velocity Under-relaxation
void PV_coupling::velocity_under_relaxation(std::vector<Cell> &list_of_cells, double &alpha)                                                               
 {  
    double alpha_source_factor = (1 - alpha)/alpha ;

    for(int i=0; i<list_of_cells.size(); i++)
    {
         velocity_b_vector_combined_ux(i)  =  velocity_b_vector_combined_ux(i) + (alpha_source_factor* velocity_a_matrix_combined(i,i) * velocity.vector_field_values_x(i)) ;
         velocity_b_vector_combined_uy(i)  =  velocity_b_vector_combined_uy(i) + (alpha_source_factor* velocity_a_matrix_combined(i,i) * velocity.vector_field_values_y(i)) ; 
         velocity_a_matrix_combined(i,i) =  velocity_a_matrix_combined(i,i) / alpha;
    }
 }  

// Momentum Predictor - Diffusion Term Discretization 
void PV_coupling::velocity_compute_diffusion_matrix(std::vector<Face> &list_of_faces, std::vector<Boundary> &list_of_boundaries, Vector_boundary_field &velocity_boundary)                                                               
 { 
      velocity_reset_diffusion(); 

       for(int i = 0; i<list_of_faces.size(); i++)
        {  
          double a_owner_diagonal, a_neighbour_diagonal, a_owner_neighbour, a_neighbour_owner, b_owner_source_ux, b_owner_source_uy, b_neighbour_source_ux, b_neighbour_source_uy;

          int face_owner_index = list_of_faces[i].get_owner_index();
          int face_neighbour_index = list_of_faces[i].get_neighbour_index();

          double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
          double face_delta = list_of_faces[i].get_face_delta(); 

          Vector face_normal = list_of_faces[i].get_face_normal();

          double face_normal_magnitude = face_normal.magnitude_of_vector();

          if(list_of_faces[i].get_is_boundary_face() == true)
            {       
              int boundary_index = list_of_faces[i].get_boundary_index();
              Boundary b = list_of_boundaries[boundary_index];

                if(velocity_boundary.get_vector_boundary_name(boundary_index) == "empty")
                  {
                    continue;
                  }

                else
                  {
                    double mu_owner = mu.scalar_field_values(face_owner_index);
                    double mu_boundary_face = interpolation_factor * mu_owner;

                    if(velocity_boundary.get_vector_boundary_type(boundary_index) == "dirichlet")
                    {   
                        std::vector<std::vector<double>> dirichlet_value = velocity_boundary.get_vector_boundary_field_values();

                        a_owner_diagonal = mu_boundary_face*face_normal_magnitude*face_delta;
                        velocity_update_a_matrix_diffusion(a_owner_diagonal, face_owner_index, face_owner_index); 

                        b_owner_source_ux = mu_boundary_face*face_normal_magnitude*face_delta*dirichlet_value[boundary_index][0];
                        velocity_update_b_vector_diffusion_ux(b_owner_source_ux, face_owner_index) ;

                        b_owner_source_uy = mu_boundary_face*face_normal_magnitude*face_delta*dirichlet_value[boundary_index][1];
                        velocity_update_b_vector_diffusion_uy(b_owner_source_uy, face_owner_index) ;

                    }

                    else
                    {
                        std::vector<std::vector<double>> neumann_gradient = velocity_boundary.get_vector_boundary_field_values();

                        b_owner_source_ux = mu_owner * face_normal_magnitude * neumann_gradient[boundary_index][0];
                        velocity_update_b_vector_diffusion_ux(b_owner_source_ux, face_owner_index);

                        b_owner_source_uy = mu_owner * face_normal_magnitude * neumann_gradient[boundary_index][1];
                        velocity_update_b_vector_diffusion_uy(b_owner_source_ux, face_owner_index);
                    }
                  }
            }

                else
                {
                   double mu_owner_cell = mu.scalar_field_values(face_owner_index);
                   double mu_neighbour_cell = mu.scalar_field_values(face_neighbour_index);

                   double mu_face = (interpolation_factor*mu_owner_cell) + ((1 - interpolation_factor)*(mu_neighbour_cell)) ;

                   a_owner_diagonal = mu_face * face_normal_magnitude * face_delta;
                   a_neighbour_diagonal = mu_face * face_normal_magnitude * face_delta;

                   a_owner_neighbour =  -1 * mu_face * face_normal_magnitude * face_delta;
                   a_neighbour_owner =  -1 * mu_face * face_normal_magnitude * face_delta;
                   
                   velocity_update_a_matrix_diffusion(a_owner_diagonal, face_owner_index, face_owner_index);
                   velocity_update_a_matrix_diffusion(a_neighbour_diagonal, face_neighbour_index, face_neighbour_index);
                   velocity_update_a_matrix_diffusion(a_owner_neighbour, face_owner_index, face_neighbour_index);
                   velocity_update_a_matrix_diffusion(a_neighbour_owner, face_neighbour_index, face_owner_index);
            
                }
        }     
 }


// Momentum Predictor - Convection Term Discretization 
void PV_coupling::velocity_compute_convection_matrix(std::vector<Face> &list_of_faces, std::vector<Boundary> &list_of_boundaries, Vector_boundary_field &velocity_boundary)                                                               
 { 
      velocity_reset_convection();  

      for(int i = 0; i<list_of_faces.size(); i++)
        {   
             Vector face_velocity;
             int face_index = list_of_faces[i].get_face_index();

             double a_owner_diagonal, a_neighbour_diagonal, a_owner_neighbour, a_neighbour_owner, b_owner_source_ux, b_neighbour_source_ux, b_owner_source_uy, b_neighbour_source_uy;
             double neighbour_velocity_x, neighbour_velocity_y, neighbour_velocity_z;
             double rho_owner, rho_neighbour, rho_face;

             int owner_index = list_of_faces[i].get_owner_index();
             int neighbour_index = list_of_faces[i].get_neighbour_index();

             double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
             double face_delta = list_of_faces[i].get_face_interpolation_factor();

            if(list_of_faces[i].get_is_boundary_face() == true)
            {
                int boundary_index = list_of_faces[i].get_boundary_index();
                Boundary b = list_of_boundaries[list_of_faces[i].get_boundary_index()];
     
                std::vector<std::vector<double>> vel = velocity_boundary.get_vector_boundary_field_values();

                face_velocity = Vector(vel[boundary_index][0], vel[boundary_index][1], vel[boundary_index][2]);

                rho_owner = rho.scalar_old_field_values (owner_index);
                rho_face = rho_owner * interpolation_factor; 
            }

            else
            {      
                double owner_vx = velocity.vector_field_values_x(owner_index);
                double owner_vy = velocity.vector_field_values_y(owner_index);
                double owner_vz = velocity.vector_field_values_z(owner_index);
                double neighbour_vx = velocity.vector_field_values_x(neighbour_index);
                double neighbour_vy = velocity.vector_field_values_y(neighbour_index);
                double neighbour_vz = velocity.vector_field_values_z(neighbour_index);
                Vector owner_velocity(owner_vx, owner_vy, owner_vz);
                Vector neighbour_velocity(neighbour_vx, neighbour_vy, neighbour_vz);
                face_velocity = owner_velocity * interpolation_factor + neighbour_velocity * (1 - interpolation_factor);

                rho_owner = rho.scalar_old_field_values (owner_index);
                rho_neighbour = rho.scalar_old_field_values(neighbour_index);

                rho_face = (interpolation_factor * rho_owner) + ((1.0 - interpolation_factor) * rho_neighbour);
            }

                Vector face_normal = list_of_faces[i].get_face_normal();

                double velocity_flux = face_normal.dot_product(face_velocity);

                velocity_flux = velocity_flux * rho_face;

                double face_normal_magnitude = face_normal.magnitude_of_vector();

                if(velocity_flux >= 0)
                {
                    interpolation_factor = 1;
                }

                else
                {
                    interpolation_factor = 0;
                }

                 if(list_of_faces[i].get_is_boundary_face() == true)
                {
                     
                     int boundary_index = list_of_faces[i].get_boundary_index();
                     Boundary b = list_of_boundaries[boundary_index];

                    if(velocity_boundary.get_vector_boundary_name(boundary_index) == "empty")
                     {
                        continue;
                     }

                     else
                     {
                        if(velocity_boundary.get_vector_boundary_type(boundary_index) == "dirichlet")
                        {   
                            std::vector<std::vector<double>> dirichlet_value = velocity_boundary.get_vector_boundary_field_values();

                            b_owner_source_ux = -1*velocity_flux * dirichlet_value[boundary_index][0];
                            velocity_update_b_vector_convection_ux(b_owner_source_ux, owner_index);

                            b_owner_source_uy = -1*velocity_flux * dirichlet_value[boundary_index][1];
                            velocity_update_b_vector_convection_uy(b_owner_source_uy, owner_index);
                        }

                       if(velocity_boundary.get_vector_boundary_type(boundary_index) == "neumann")
                        {
                            std::vector<std::vector<double>> neumann_gradient = velocity_boundary.get_vector_boundary_field_values();

                            a_owner_diagonal = velocity_flux;
                            velocity_update_a_matrix_convection(a_owner_diagonal, owner_index, owner_index);

                            b_owner_source_ux = -1*(velocity_flux/face_delta) * neumann_gradient[boundary_index][0];
                            velocity_update_b_vector_convection_ux(b_owner_source_ux, owner_index);

                            b_owner_source_uy = -1*(velocity_flux/face_delta) * neumann_gradient[boundary_index][1];
                            velocity_update_b_vector_convection_uy(b_owner_source_uy, owner_index);
                        }
                     }
                }

                else
                {       

                  a_owner_diagonal = interpolation_factor*velocity_flux;
                  a_neighbour_diagonal = -1*(1 - interpolation_factor)*velocity_flux;

                  velocity_update_a_matrix_convection(a_owner_diagonal, owner_index, owner_index);
                  velocity_update_a_matrix_convection(a_neighbour_diagonal, neighbour_index, neighbour_index);

                  a_owner_neighbour = (1 - interpolation_factor)*velocity_flux;
                  a_neighbour_owner = -1*velocity_flux*interpolation_factor;
            
                  velocity_update_a_matrix_convection(a_owner_neighbour, owner_index, neighbour_index);
                  velocity_update_a_matrix_convection(a_neighbour_owner, neighbour_index, owner_index);
                   
                } 
        }
    }  

// Velocity - Initial Residuals
void PV_coupling::velocity_calculate_initial_residuals(std::vector<double> &x_distance, std::vector<double> &y_distance, int &iteration_number, int &output_interval, std::ofstream &sum_initial_residual_ux, 
                                               std::ofstream &sum_initial_residual_uy)
{ 
   if((iteration_number % output_interval) == 0)
  { 
   std::ofstream file_initial_residuals_ux("initial_residuals_ux.txt");
   std::ofstream file_initial_residuals_uy("initial_residuals_uy.txt");

   velocity_initial_residuals_ux = velocity_b_vector_combined_ux - velocity_a_matrix_combined * velocity.vector_field_values_x;
   velocity_initial_residuals_uy = velocity_b_vector_combined_uy - velocity_a_matrix_combined * velocity.vector_field_values_y;

   for(int i=0; i<y_distance.size(); i++)
   {
      for(int j=0; j<x_distance.size(); j++)
      {
            file_initial_residuals_ux<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity_initial_residuals_ux(j + (i*x_distance.size()))<<std::endl;
            file_initial_residuals_uy<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity_initial_residuals_uy(j + (i*x_distance.size()))<<std::endl;
      }
   } 

   file_initial_residuals_ux.close();
   file_initial_residuals_uy.close();
  } 
 
  velocity_initial_residuals_ux = velocity_b_vector_combined_ux - velocity_a_matrix_combined * velocity.vector_field_values_x;
  velocity_initial_residuals_uy = velocity_b_vector_combined_uy - velocity_a_matrix_combined * velocity.vector_field_values_y;

  double sum_r_ux = 0.0, sum_r_uy = 0.0;

  for(int i=0; i<velocity_initial_residuals_ux.size(); i++)
  {
     sum_r_ux = sum_r_ux + velocity_initial_residuals_ux(i);
     sum_r_uy = sum_r_uy + velocity_initial_residuals_uy(i);
  } 

  sum_initial_residual_ux<<iteration_number<<"\t"<<sum_r_ux<<std::endl;
  sum_initial_residual_uy<<iteration_number<<"\t"<<sum_r_ux<<std::endl;
 
}    

// Solving Momentum Predictor
void PV_coupling::velocity_solve_matrices(std::vector<Cell> list_of_cells)
{
   int size = list_of_cells.size();

    SpMat Au(size, size);

    Au.reserve(Eigen::VectorXi::Constant(size,5));
    for(int row=0; row<size; row++)
    {
      for(int col = 0; col < size; col++ )
      {
        if(fabs(velocity_a_matrix_combined(row,col)) > 1e-10)
        {
           Au.insert(row,col) = velocity_a_matrix_combined(row,col);
        }
      }
    }

    // Conversion to Compressed Row Format 
    Au.makeCompressed();

    // Using Sparse Matrix Solver by Bi-Conjugate Gradient Method
    Eigen::BiCGSTAB<SpMat> solver;
    solver.setTolerance(1e-10);
    solver.compute(Au);


    velocity.vector_field_values_x = solver.solve(velocity_b_vector_combined_ux);  
     
    velocity.vector_field_values_y = solver.solve(velocity_b_vector_combined_uy);      

    velocity.set_old_vector_field_values();
}  

// Velocity - Final Residuals
void PV_coupling::velocity_calculate_final_residuals(std::vector<double> &x_distance, std::vector<double> &y_distance, int &iteration_number, int &output_interval, std::ofstream &sum_final_residual_ux, std::ofstream &sum_final_residual_uy)
{
   if((iteration_number % output_interval) ==0)
 {  
   std::ofstream file_final_residuals_ux("final_residuals_ux.txt");
   std::ofstream file_final_residuals_uy("final_residuals_uy.txt");
   
   velocity_final_residuals_ux = velocity_b_vector_combined_ux - velocity_a_matrix_combined * velocity.vector_field_values_x;
   velocity_final_residuals_uy = velocity_b_vector_combined_uy - velocity_a_matrix_combined * velocity.vector_field_values_y;

      for(int i=0; i<y_distance.size(); i++)
   {
      for(int j=0; j<x_distance.size(); j++)
      {
        file_final_residuals_ux<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity_final_residuals_ux(j + (i*x_distance.size()))<<std::endl;
        file_final_residuals_uy<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity_final_residuals_uy(j + (i*x_distance.size()))<<std::endl;
      }

   } 

   file_final_residuals_ux.close();
   file_final_residuals_uy.close();
 }

  velocity_final_residuals_ux = velocity_b_vector_combined_ux - velocity_a_matrix_combined * velocity.vector_field_values_x;
  velocity_final_residuals_uy = velocity_b_vector_combined_uy - velocity_a_matrix_combined * velocity.vector_field_values_y;

  double sum_r_ux = 0.0, sum_r_uy = 0.0;

  for(int i=0; i<velocity_initial_residuals_ux.size(); i++)
  {
     sum_r_ux = sum_r_ux + velocity_final_residuals_ux(i);
     sum_r_uy = sum_r_uy + velocity_final_residuals_uy(i);
  } 

  sum_final_residual_ux<<iteration_number<<"\t"<<sum_r_ux<<std::endl;
  sum_final_residual_uy<<iteration_number<<"\t"<<sum_r_ux<<std::endl;
}   


void PV_coupling::velocity_set_face_and_cell_fluxes(std::vector<Cell> &list_of_cells, std::vector<Face> &list_of_faces, std::vector<Boundary> &list_of_boundaries, Vector_boundary_field &velocity_boundary)
{
    std::vector<double> cell_fluxes;

        for(int i=0; i<list_of_faces.size(); i++)
        { 
            double face_flux;
            Vector face_velocity;

            int owner_index = list_of_faces[i].get_owner_index();
            int neighbour_index = list_of_faces[i].get_neighbour_index();
            double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
            
            Vector face_normal = list_of_faces[i].get_face_normal();

            if(list_of_faces[i].get_is_boundary_face() == true)
            { 
                int boundary_index = list_of_faces[i].get_boundary_index();
                Boundary b = list_of_boundaries[boundary_index];
                
                if(velocity_boundary.get_vector_boundary_name(boundary_index) == "empty")
                {
                    break;
                }

                std::vector<std::vector<double>> dirichlet_value = velocity_boundary.get_vector_boundary_field_values();
                face_velocity = Vector(dirichlet_value[boundary_index][0], dirichlet_value[boundary_index][1], dirichlet_value[boundary_index][2]);  
            }  

            else
            { 
                double x_velocity_owner = velocity.vector_field_values_x(owner_index);
                double y_velocity_owner = velocity.vector_field_values_y(owner_index);
                double z_velocity_owner = velocity.vector_field_values_z(owner_index);

                double x_velocity_neighbour = velocity.vector_field_values_x(neighbour_index);
                double y_velocity_neighbour = velocity.vector_field_values_y(neighbour_index);
                double z_velocity_neighbour = velocity.vector_field_values_z(neighbour_index);

                Vector owner_vel(x_velocity_owner, y_velocity_owner, z_velocity_owner);
                Vector neighbour_vel(x_velocity_neighbour, y_velocity_neighbour, z_velocity_neighbour);

                face_velocity = (owner_vel * interpolation_factor) + (neighbour_vel * (1 - interpolation_factor));
            }  

            face_flux = face_normal.dot_product(face_velocity);  

            list_of_faces[i].set_face_flux(face_flux);

        }
}

// Momentum Predictor - Source Matrix (Gravity Term)
// g =  +9.81 for inverting the direction of gravity
void PV_coupling::velocity_compute_source_matrix(std::vector<Cell> &list_of_cells)                                                               
 {  
    velocity_reset_source(); 

    for(int i = 0; i<list_of_cells.size(); i++)
    {  
        int cell_index = list_of_cells[i].get_cell_index();

        double cell_volume = list_of_cells[i].get_cell_volume();
        double gravity_constant = 9.81; 
        double cell_density = rho.scalar_field_values(cell_index);

        double gravity_source_term_ux = 0.0;
        double gravity_source_term_uy = cell_density * gravity_constant * cell_volume;

        velocity_update_b_vector_source_ux(gravity_source_term_ux, cell_index);
        velocity_update_b_vector_source_uy(gravity_source_term_uy, cell_index);
    }    
 } 


std::vector<double> PV_coupling::velocity_store_ap_coefficients()
{
    std::vector<double> diag_coeffs(velocity_b_vector_combined_ux.size());

    for(int i=0; i<velocity_b_vector_combined_ux.size(); i++)
    {
        diag_coeffs[i] = 1.0/velocity_a_matrix_combined(i,i);
    }

    return diag_coeffs;
}


// Velocity Correction
  void PV_coupling::velocity_correct_cell_centre_velocities(std::vector<Cell> &list_of_cells, Vector_field &grad_p , std::vector<double> &temp_ap_coeffs)
  {
     for(int i=0; i<list_of_cells.size(); i++)
    {
         velocity.vector_field_values_x(i) -= temp_ap_coeffs[i] * grad_p.vector_field_values_x(i);
         velocity.vector_field_values_y(i) -= temp_ap_coeffs[i] * grad_p.vector_field_values_y(i);
    }
  }

// Div u computation to check divergence of simulation
void PV_coupling::velocity_compute_div_u(std::vector<Cell> &list_of_cells, int &n_cells, std::vector<Face> &list_of_faces)                                                               
 {  
    std::vector<double> div_u;
    div_u.resize(n_cells);
    
    for(int j=0; j<list_of_cells.size(); j++)
    {
      div_u[j] = 0.0;
    }


 for(int i = 0; i<list_of_faces.size(); i++)
       {   
          int owner_index = list_of_faces[i].get_owner_index();
          int neighbour_index = list_of_faces[i].get_neighbour_index();         
          
          double contribution_to_owner = list_of_faces[i].get_face_flux();
          double conttribution_to_neighbour = -1*contribution_to_owner;
          div_u[owner_index] += contribution_to_owner;

          if(neighbour_index > 0)
         {
          div_u[neighbour_index] += conttribution_to_neighbour;
         }

            
       }

       for(int j=0; j<list_of_cells.size(); j++)
       {
         if(div_u[j] > 1e-6)
         {
          std::cout<<"Alert!!!"<<std::endl;
         }
      }
 } 

//Output Matrix coefficients of discretized momentum predictor
void PV_coupling::velocity_output_vector_matrix_coefficients_to_file(double total_cells)
{
     std::ofstream convection_coeffs_a("vel_convection_coeffs_a.txt");
     std::ofstream convection_coeffs_b_x("vel_convection_coeffs_b_x.txt");
     std::ofstream convection_coeffs_b_y("vel_convection_coeffs_b_y.txt");
     std::ofstream diffusion_coeffs_a("vel_diffusion_coeffs_a.txt");
     std::ofstream diffusion_coeffs_b_x("vel_diffusion_coeffs_b_x.txt");
     std::ofstream diffusion_coeffs_b_y("vel_diffusion_coeffs_b_y.txt");
     std::ofstream roc_coeffs_a("vel_roc_coeffs_a.txt");
     std::ofstream roc_coeffs_b_x("vel_roc_coeffs_b_x.txt");
     std::ofstream roc_coeffs_b_y("vel_roc_coeffs_b_y.txt");
     std::ofstream source_coeffs_b_x("vel_source_coeffs_b_x.txt");
     std::ofstream source_coeffs_b_y("vel_source_coeffs_b_y.txt");
     std::ofstream combined_coeffs_a("vel_combined_coeffs_a.txt");
     std::ofstream combined_coeffs_b_x("vel_combined_coeffs_b_x.txt");
     std::ofstream combined_coeffs_b_y("vel_combined_coeffs_b_y.txt");
    
     for(int i = 0; i<total_cells; i++)
     {
        for(int j=0; j<total_cells; j++)
              {
                if(j < total_cells - 1)
                {
                 convection_coeffs_a<<velocity_a_matrix_convection(i,j)<<"\t";
                 diffusion_coeffs_a<<velocity_a_matrix_diffusion(i,j)<<"\t";
                 roc_coeffs_a<<velocity_a_matrix_rate_of_change(i,j)<<"\t";
                 combined_coeffs_a<<velocity_a_matrix_combined(i,j)<<"\t";

                }

                else
                {
                 convection_coeffs_a<<velocity_a_matrix_convection(i,j)<<"\n";
                 diffusion_coeffs_a<<velocity_a_matrix_diffusion(i,j)<<"\n";
                 roc_coeffs_a<<velocity_a_matrix_rate_of_change(i,j)<<"\n";
                 combined_coeffs_a<<velocity_a_matrix_combined(i,j)<<"\n";

                }
              }
     }

          for(int k=0; k < total_cells; k++)
          {
                 convection_coeffs_b_x<<velocity_b_vector_convection_ux(k)<<"\n";
                 convection_coeffs_b_y<<velocity_b_vector_convection_uy(k)<<"\n";

                 diffusion_coeffs_b_x<<velocity_b_vector_diffusion_ux(k)<<"\n";
                 diffusion_coeffs_b_y<<velocity_b_vector_diffusion_uy(k)<<"\n";

                 roc_coeffs_b_x<<velocity_b_vector_rate_of_change_ux(k)<<"\n";
                 roc_coeffs_b_y<<velocity_b_vector_rate_of_change_uy(k)<<"\n";

                 source_coeffs_b_x<<velocity_b_vector_source_ux(k)<<"\n";
                 source_coeffs_b_y<<velocity_b_vector_source_ux(k)<<"\n";

                 combined_coeffs_b_x<<velocity_b_vector_combined_ux(k)<<"\n";
                 combined_coeffs_b_y<<velocity_b_vector_combined_uy(k)<<"\n";
          }    

   convection_coeffs_a.close();
   convection_coeffs_b_x.close();
   convection_coeffs_b_y.close();
   diffusion_coeffs_a.close();
   diffusion_coeffs_b_x.close();
   diffusion_coeffs_b_y.close();
   roc_coeffs_a.close();
   roc_coeffs_b_x.close();
   roc_coeffs_b_y.close();
   combined_coeffs_a.close();
   combined_coeffs_b_x.close();
   combined_coeffs_b_y.close();
   source_coeffs_b_x.close();
   source_coeffs_b_y.close();
}


//Momemtum Predictor - Initial Convergence Ux

void PV_coupling::velocity_plot_convergence_initial_x(std::ofstream &x_velocity_convergence_initial, int &iteration_no)
{   
    int index;
    Eigen::VectorXd diff = (velocity.vector_field_values_x - velocity.vector_old_field_values_x);

    for(int i=0; i<velocity.vector_field_values_x.size(); i++)
    {
        diff(i) = abs(diff(i));
    }

     double max = diff(0);
    
    for(int i=1; i<velocity.vector_field_values_x.size(); i++)
    {   double next_term = diff(i);
        double last_term = diff(i-1);
        
        if(next_term > last_term)
        {
         max = diff(i);
         index = i;
        }
    }

     x_velocity_convergence_initial<<iteration_no<<"\t"<<max<<"\t"<<std::endl;
}


//Momemtum Predictor - Initial Convergence Uy

void PV_coupling::velocity_plot_convergence_initial_y(std::ofstream &y_velocity_convergence_initial, int &iteration_no)
{
    Eigen::VectorXd difference = (velocity.vector_field_values_y - velocity.vector_old_field_values_y);

    for(int i=0; i<velocity.vector_field_values_y.size(); i++)
    {
        difference(i) = abs(difference(i));
    }

     double max = difference(0);
    
    for(int i=1; i<velocity.vector_field_values_y.size(); i++)
    {   double next_term = difference(i);
        double last_term = difference(i-1);
        
        if(next_term > last_term)
        {
         max = difference(i);
        }
    }

     y_velocity_convergence_initial<<iteration_no<<"\t"<<max<<std::endl;

}

//Momemtum Predictor - Final Convergence Ux

void PV_coupling::velocity_plot_convergence_final_x(std::ofstream &x_velocity_convergence_final, int &iteration_no)
{
       Eigen::VectorXd difference = (velocity.vector_field_values_x - velocity.vector_old_field_values_x);

    for(int i=0; i<velocity.vector_field_values_x.size(); i++)
    {
        difference(i) = abs(difference(i));
    }

     double max = difference(0);
    
    for(int i=1; i<velocity.vector_field_values_x.size(); i++)
    {   double next_term = difference(i);
        double last_term = difference(i-1);
        
        if(next_term > last_term)
        {
         max = difference(i);
        }
    }

     x_velocity_convergence_final<<iteration_no<<"\t"<<max<<std::endl;
}


//Momemtum Predictor - Final Convergence Uy

void PV_coupling::velocity_plot_convergence_final_y(std::ofstream &y_velocity_convergence_final, int &iteration_no)
{
      Eigen::VectorXd difference = (velocity.vector_field_values_y - velocity.vector_old_field_values_y);

    for(int i=0; i<velocity.vector_field_values_y.size(); i++)
    {
        difference(i) = abs(difference(i));
    }

     double max = difference(0);
    
    for(int i=1; i<velocity.vector_field_values_y.size(); i++)
    {   double next_term = difference(i);
        double last_term = difference(i-1);
        
        if(next_term > last_term)
        {
         max = difference(i);
        }
    }

     y_velocity_convergence_final<<iteration_no<<"\t"<<max<<std::endl;
}


void PV_coupling::pressure_reset_rate_of_change()
{
    pressure_a_matrix_rate_of_change.setZero();
    pressure_b_vector_rate_of_change.setZero();
}

void PV_coupling::pressure_reset_source()
{
    pressure_b_vector_source.setZero();
}

void PV_coupling::pressure_reset_diffusion()
{
   pressure_a_matrix_diffusion.setZero();
   pressure_b_vector_diffusion.setZero();
}

void PV_coupling::pressure_reset_convection()
{
    pressure_a_matrix_convection.setZero();
    pressure_b_vector_convection.setZero();
}

void PV_coupling::pressure_reset_combined()
{
    pressure_a_matrix_combined.setZero();
    pressure_b_vector_combined.setZero();
}

void PV_coupling::pressure_update_a_matrix_rate_of_change(double a, int index)
{
   pressure_a_matrix_rate_of_change(index, index) = pressure_a_matrix_rate_of_change(index, index) + a ;         
}

void PV_coupling::pressure_update_b_vector_rate_of_change(double a, int index)
{
    pressure_b_vector_rate_of_change(index) = pressure_b_vector_rate_of_change(index) + a;
}

void PV_coupling::pressure_update_b_vector_source(double a, int index)
{
    pressure_b_vector_source(index) = pressure_b_vector_source(index) + a;
}

void PV_coupling::pressure_update_a_matrix_diffusion(double a, int row_index, int column_index)
{ 
    pressure_a_matrix_diffusion(row_index, column_index) = pressure_a_matrix_diffusion(row_index, column_index) + a;                
}                                                                         

void PV_coupling::pressure_update_b_vector_diffusion(double a, int index)
{
    pressure_b_vector_diffusion(index) = a;           
}

void PV_coupling::pressure_update_a_matrix_convection(double a, int row_index, int column_index)
{ 
    pressure_a_matrix_convection(row_index, column_index) = pressure_a_matrix_convection(row_index, column_index) + a;                 
}                                                                         

void PV_coupling::pressure_update_b_vector_convection(double a, int index)
{
    pressure_b_vector_convection(index) = pressure_b_vector_convection(index) + a;           
}

void PV_coupling::pressure_combine_a_and_b_matrices()
{
    for(int i = 0; i<pressure_a_matrix_combined.rows(); i++)
    {
        for(int j=0; j<pressure_a_matrix_combined.cols(); j++)
        {
            pressure_a_matrix_combined(i,j) = pressure_a_matrix_rate_of_change(i,j) + pressure_a_matrix_diffusion(i,j) + pressure_a_matrix_convection(i,j);
        }
    }

    for(int k =0 ; k <pressure_b_vector_combined.size(); k++)
    {
        pressure_b_vector_combined(k) = pressure_b_vector_rate_of_change(k) + pressure_b_vector_source(k) + pressure_b_vector_diffusion(k) + pressure_b_vector_convection(k) ;
    }
}


// Pressure Corrector - Source Matrix Computation

void PV_coupling::pressure_compute_source_matrix(std::vector<Cell> &list_of_cells, std::vector<Face> &list_of_faces)                                                               
 {  
    pressure_reset_source(); 

    for(int i = 0; i<list_of_faces.size(); i++)
    {  
        int face_index = list_of_faces[i].get_face_index();
        int owner_index = list_of_faces[i].get_owner_index();
        int neighbour_index = list_of_faces[i].get_neighbour_index();

        double face_flux_value = list_of_faces[i].get_face_flux();

        double face_flux_value_owner_cell = face_flux_value;
        double face_flux_value_neighbour_cell = -1 * face_flux_value;

        if(list_of_faces[i].get_is_boundary_face() == true)
        { 
          pressure_update_b_vector_source(face_flux_value_owner_cell, owner_index);
        }

        else
        {
           pressure_update_b_vector_source(face_flux_value_owner_cell, owner_index);
           pressure_update_b_vector_source(face_flux_value_neighbour_cell, neighbour_index);
        } 
    }
 } 

 // Pressure Corrector - Diffusion Matrix Computation

void PV_coupling::pressure_compute_diffusion_matrix(std::vector<Face> &list_of_faces,  std::vector <double> &ap_coeff, std::vector<Boundary> &list_of_boundaries, int &iteration_no, Scalar_boundary_field &pressure_boundary)                                                              
 { 
      pressure_reset_diffusion();  

      for(int i = 0; i<list_of_faces.size(); i++)
      {
          double a_owner_diagonal, a_neighbour_diagonal, a_owner_neighbour, a_neighbour_owner, b_owner_source, b_neighbour_source;
            
            int face_owner_index = list_of_faces[i].get_owner_index();
            int face_neighbour_index = list_of_faces[i].get_neighbour_index();

            double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
            double face_delta = list_of_faces[i].get_face_delta(); 

            Vector face_normal = list_of_faces[i].get_face_normal();

            double face_normal_magnitude = face_normal.magnitude_of_vector();

                if(list_of_faces[i].get_is_boundary_face() == true)
                {
                     int boundary_index = list_of_faces[i].get_boundary_index();
                     Boundary b = list_of_boundaries[boundary_index];

                     if(pressure_boundary.get_scalar_boundary_name(boundary_index) == "empty")
                     {
                        continue;
                     }

                     else
                     {
                            double ap_coeff_owner = ap_coeff[face_owner_index];
                            double ap_boundary_face = interpolation_factor * ap_coeff_owner;

                        if(pressure_boundary.get_scalar_boundary_type(boundary_index) == "dirichlet")
                        {   
                            std::vector<double> dirichlet_value = pressure_boundary.get_scalar_boundary_field_values();

                            a_owner_diagonal = -1*ap_boundary_face * face_normal_magnitude * face_delta;
                            pressure_update_a_matrix_diffusion(a_owner_diagonal, face_owner_index, face_owner_index); 


                            b_owner_source = -1* ap_boundary_face * face_normal_magnitude * face_delta * dirichlet_value[boundary_index];
                            pressure_update_b_vector_diffusion(b_owner_source, face_owner_index);
                        }

                        else
                        {
                            std::vector<double> neumann_gradient =  pressure_boundary.get_scalar_boundary_field_values();

                            b_owner_source = -1 * ap_coeff_owner * face_normal_magnitude * neumann_gradient[boundary_index] ;

                            pressure_update_b_vector_diffusion(b_owner_source, face_owner_index);
                        }
                     }
                }

                else
                {
                   double ap_owner_cell = ap_coeff[face_owner_index];
                   double ap_neighbour_cell = ap_coeff[face_neighbour_index];

                   double ap_face = (interpolation_factor*ap_owner_cell) + ((1 - interpolation_factor)*(ap_neighbour_cell)) ;

                   a_owner_diagonal = -1 * ap_face * face_normal_magnitude * face_delta;
                   a_neighbour_diagonal = -1 * ap_face * face_normal_magnitude * face_delta;

                   a_owner_neighbour =  1 * ap_face * face_normal_magnitude * face_delta;
                   a_neighbour_owner =  1 * ap_face * face_normal_magnitude * face_delta;
                   
                   pressure_update_a_matrix_diffusion(a_owner_diagonal, face_owner_index, face_owner_index);
                   pressure_update_a_matrix_diffusion(a_neighbour_diagonal, face_neighbour_index, face_neighbour_index);
                   pressure_update_a_matrix_diffusion(a_owner_neighbour, face_owner_index, face_neighbour_index);
                   pressure_update_a_matrix_diffusion(a_neighbour_owner, face_neighbour_index, face_owner_index);
            
                }
      }
 } 
    

// Initial Pressure Residuals
void PV_coupling::pressure_calculate_initial_residuals_p(std::vector<double> &x_distance, std::vector<double> &y_distance, int &iteration_number, int &output_interval, std::ofstream &sum_initial_residual_p)
{
     if((iteration_number % output_interval) == 0)
  { 
   std::ofstream file_initial_residuals_p("pressure_initial_residuals.txt");

   pressure_initial_residuals_p = pressure_b_vector_combined - pressure_a_matrix_combined * pressure.scalar_field_values;
   
   for(int i=0; i<y_distance.size(); i++)
   {
      for(int j=0; j<x_distance.size(); j++)
      {
         file_initial_residuals_p<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<pressure.scalar_field_values(j + (i*x_distance.size()))<<std::endl;
            
      }
   } 

   file_initial_residuals_p.close();
  } 
 
  pressure_initial_residuals_p = pressure_b_vector_combined - pressure_a_matrix_combined * pressure.scalar_field_values;

  double sum_r_p = 0.0;

  for(int i=0; i<pressure_initial_residuals_p.size(); i++)
  {
     sum_r_p = sum_r_p + pressure_initial_residuals_p(i);
  } 

  sum_initial_residual_p<<iteration_number<<"\t"<<sum_r_p<<std::endl;

}
        
void PV_coupling::pressure_combine_and_solve_matrices(std::vector<Cell> &list_of_cells)
{
    pressure_combine_a_and_b_matrices();

    int size = list_of_cells.size();
    SpMat A(size, size);
    A.reserve(Eigen::VectorXi::Constant(size,5));
    for(int row=0; row<size; row++)
    {
      for(int col = 0; col < size; col++ )
      {
        if(fabs(pressure_a_matrix_combined(row,col)) > 1e-12)
        {
           A.insert(row,col) = pressure_a_matrix_combined(row,col);
        }
      }
    }

    // Converting to compressed row format
    A.makeCompressed();

    // Solving using Bi-conjugate gradient method
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setTolerance(1e-10);
    cg.compute(A);
    pressure.scalar_field_values = cg.solve(pressure_b_vector_combined);

    pressure.set_old_scalar_field_values();

}


// Pressure under-relaxation
void PV_coupling::pressure_under_relax()
{
    pressure.scalar_field_values = pressure.scalar_old_field_values + (0.7) * (pressure.scalar_field_values - pressure.scalar_old_field_values); 
} 


//Pressure Final Residuals

void PV_coupling::pressure_calculate_final_residuals_p(std::vector<double> &x_distance, std::vector<double> &y_distance, int &iteration_number, int &output_interval, std::ofstream &sum_final_residual_p)
{
  if((iteration_number % output_interval) == 0)
  { 
   std::ofstream file_final_residuals_p("pressure_final_residuals.txt");

   pressure_final_residuals_p = pressure_b_vector_combined - pressure_a_matrix_combined * pressure.scalar_field_values;
   
   for(int i=0; i<y_distance.size(); i++)
   {
      for(int j=0; j<x_distance.size(); j++)
      {
            file_final_residuals_p<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<pressure.scalar_field_values(j + (i*x_distance.size()))<<std::endl;     
      }
   } 

   file_final_residuals_p.close();
  }


  pressure_final_residuals_p = pressure_b_vector_combined - pressure_a_matrix_combined * pressure.scalar_field_values;

  double sum_r_p = 0.0;

  for(int i=0; i<pressure_final_residuals_p.size(); i++)
  {
     sum_r_p = sum_r_p + pressure_final_residuals_p(i);
  } 

  sum_final_residual_p<<iteration_number<<"\t"<<sum_r_p<<std::endl;
}

//Pressure corrector - Flux Correction
void PV_coupling::pressure_compute_flux_correction(std::vector<Cell> &list_of_cells, std::vector<Face> &list_of_faces, std::vector<double> &temp_ap_coeffs, std::vector<Boundary> &list_of_boundaries)
{        
   double ap_face, grad_p_face, new_face_flux_value, f_corr, test_val = 0.0;

   for(int i=0; i<list_of_faces.size(); i++)
   {
      int owner_index = list_of_faces[i].get_owner_index();
      int neighbour_index = list_of_faces[i].get_neighbour_index();
      double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
      double face_delta = list_of_faces[i].get_face_delta();

      Vector face_normal = list_of_faces[i].get_face_normal();

      double face_normal_magnitude = face_normal.magnitude_of_vector();

      if(list_of_faces[i].get_is_boundary_face() == true)
        {   
            int boundary_index = list_of_faces[i].get_boundary_index();
            Boundary b = list_of_boundaries[boundary_index];

            if(b.return_boundary_name() == "empty")
            {
            continue;
            }

            else
            {
                double IF = list_of_faces[i].get_face_interpolation_factor();
                double ap_owner = temp_ap_coeffs[owner_index];

                grad_p_face = 0;
                new_face_flux_value = ap_face * grad_p_face;

                f_corr = list_of_faces[i].get_face_flux() - new_face_flux_value;
            }
        }

        else
        {
            double ap_owner_cell = temp_ap_coeffs[owner_index];
            double ap_neighbour_cell = temp_ap_coeffs[neighbour_index];

            ap_face = (interpolation_factor*ap_owner_cell) + ((1 - interpolation_factor)*(ap_neighbour_cell)) ;

            grad_p_face = ((pressure.scalar_field_values(neighbour_index) - pressure.scalar_field_values(owner_index))*face_delta); 

            new_face_flux_value =  ap_face* face_normal_magnitude *grad_p_face;
             f_corr = list_of_faces[i].get_face_flux() - new_face_flux_value;
        }
            list_of_faces[i].set_face_flux(f_corr); 

   }
}

          
// Pressure Corrector- Matrix coefficients

void PV_coupling::pressure_output_scalar_matrix_coefficients_to_file(double total_cells)
{
     std::ofstream pressure_convection_coeffs_a("pressure_convection_coeffs_a.txt");
     std::ofstream pressure_convection_coeffs_b("pressure_convection_coeffs_b.txt");
     std::ofstream pressure_diffusion_coeffs_a("pressure_diffusion_coeffs_a.txt");
     std::ofstream pressure_diffusion_coeffs_b("pressure_diffusion_coeffs_b.txt");
     std::ofstream pressure_roc_coeffs_a("pressure_roc_coeffs_a.txt");
     std::ofstream pressure_roc_coeffs_b("pressure_roc_coeffs_b.txt");
     std::ofstream pressure_source_coeffs_b("pressure_source_coeffs_b.txt");
     std::ofstream pressure_combined_coeffs_a("pressure_combined_coeffs_a.txt");
     std::ofstream pressure_combined_coeffs_b("pressure_combined_coeffs_b.txt");
    
     for(int i = 0; i<total_cells; i++)
     {
        for(int j=0; j<total_cells; j++)
              {
                if(j < total_cells - 1)
                {
                 pressure_convection_coeffs_a<<pressure_a_matrix_convection(i,j)<<"\t";
                 pressure_diffusion_coeffs_a<<pressure_a_matrix_diffusion(i,j)<<"\t";
                 pressure_roc_coeffs_a<<pressure_a_matrix_rate_of_change(i,j)<<"\t";
                 pressure_combined_coeffs_a<<pressure_a_matrix_combined(i,j)<<"\t";

                }

                else
                {
                 pressure_convection_coeffs_a<<pressure_a_matrix_convection(i,j)<<"\n";
                 pressure_diffusion_coeffs_a<<pressure_a_matrix_diffusion(i,j)<<"\n";
                 pressure_roc_coeffs_a<<pressure_a_matrix_rate_of_change(i,j)<<"\n";
                 pressure_combined_coeffs_a<<pressure_a_matrix_combined(i,j)<<"\n";

                }
              }
     }

     for(int k=0; k<total_cells; k++)
     {
                pressure_convection_coeffs_b<<pressure_b_vector_convection(k)<<"\n";
                 pressure_diffusion_coeffs_b<<pressure_b_vector_diffusion(k)<<"\n";
                 pressure_roc_coeffs_b<<pressure_b_vector_rate_of_change(k)<<"\n";
                 pressure_source_coeffs_b<<pressure_b_vector_source(k)<<"\n";
                 pressure_combined_coeffs_b<<pressure_b_vector_combined(k)<<"\n";

     }

   pressure_convection_coeffs_a.close();
   pressure_convection_coeffs_b.close();
   pressure_diffusion_coeffs_a.close();
   pressure_diffusion_coeffs_b.close();
   pressure_roc_coeffs_a.close();
   pressure_roc_coeffs_b.close();
   pressure_combined_coeffs_a.close();
   pressure_combined_coeffs_b.close();
   pressure_source_coeffs_b.close();
}


 // Pressure Corrector - Initial convergence               
 void PV_coupling::pressure_plot_convergence_initial(std::ofstream &plot_convergence_initial_h, int &iteration_no)
 {
    Eigen::VectorXd difference = (pressure.scalar_field_values - pressure.scalar_old_field_values);

    for(int i=0; i<pressure.scalar_field_values.size(); i++)
    {
        difference(i) = abs(difference(i));
    }

     double max = difference(0);
    
    for(int i=1; i<pressure.scalar_field_values.size(); i++)
    {   double next_term = difference(i);
        double last_term = difference(i-1);
        
        if(next_term > last_term)
        {
         max = difference(i);
        }
    }
     plot_convergence_initial_h<<iteration_no<<"\t"<<max<<std::endl;
 }               


Scalar_field PV_coupling::retrieve_pressure_field()                                                               
 {  
     return pressure;
 } 


// Setting Initial Phase Faraction Fields (Like setfields from 'interFoam')
void PV_coupling::set_alpha_scalar_initial_fields(std::vector<Cell> &list_of_cells, double &rho_1, double &rho_2, double &nu_1, double &nu_2, double &mu_1, double &mu_2, double &x_max, double &y_max)
{
    std::ofstream plot_initial_alpha("initial_alpha.txt");
    std::ofstream plot_initial_nu("initial_nu.txt");
    std::ofstream plot_initial_rho("initial_rho.txt");
    std::ofstream plot_initial_mu("initial_mu.txt");

    for(int i=0; i<list_of_cells.size(); i++)
    {
     
        Vector distance = list_of_cells[i].get_cell_centre();

        double x = distance[0];
        double y = distance[1];
        
        if(( distance[1] <= 0.017) || ((pow ((x - 0.05),2)) + (pow((y - 0.075),2)) < (pow(0.0125,2))))
        {
            alpha.scalar_field_values(i) = 1.0;
        }

        else
        {
            alpha.scalar_field_values(i) = 0.0;
        }
        
        rho.scalar_field_values(i) = alpha.scalar_field_values(i) * rho_1 + ((1 - (alpha.scalar_field_values(i))) * rho_2);
        nu.scalar_field_values(i) = alpha.scalar_field_values(i) * nu_1 + ((1 - alpha.scalar_field_values(i)) * nu_2);
        mu.scalar_field_values(i) = alpha.scalar_field_values(i) * mu_1 + ((1 - alpha.scalar_field_values(i)) * mu_2);

        plot_initial_alpha<<distance[0]<<"\t"<<distance[1]<<"\t"<<alpha.scalar_field_values(i)<<std::endl;
        plot_initial_nu<<distance[0]<<"\t"<<distance[1]<<"\t"<<nu.scalar_field_values(i)<<std::endl;
        plot_initial_rho<<distance[0]<<"\t"<<distance[1]<<"\t"<<rho.scalar_field_values(i)<<std::endl;
        plot_initial_mu<<distance[0]<<"\t"<<distance[1]<<"\t"<<mu.scalar_field_values(i)<<std::endl;
    }

    plot_initial_alpha.close();
    plot_initial_nu.close();
    plot_initial_rho.close();
    plot_initial_mu.close();
}

void PV_coupling::alpha_reset_rate_of_change()
{
    alpha_a_matrix_rate_of_change.setZero();
    alpha_b_vector_rate_of_change.setZero();
}

void PV_coupling::alpha_reset_source()
{
    alpha_b_vector_source.setZero();
}

void PV_coupling::alpha_reset_convection()
{
    alpha_a_matrix_convection.setZero();
    alpha_b_vector_convection.setZero();
}

void PV_coupling::alpha_reset_combined()
{
    alpha_a_matrix_combined.setZero();
    alpha_b_vector_combined.setZero();
}

void PV_coupling::alpha_update_a_matrix_rate_of_change(double a, int index)
{
   alpha_a_matrix_rate_of_change(index, index) = alpha_a_matrix_rate_of_change(index, index) + a ;         
}

void PV_coupling::alpha_update_b_vector_rate_of_change(double a, int index)
{
    alpha_b_vector_rate_of_change(index) = alpha_b_vector_rate_of_change(index) + a;
}

void PV_coupling::alpha_update_b_vector_source(double a, int index)
{
    alpha_b_vector_source(index) = alpha_b_vector_source(index) + a;
}

void PV_coupling::alpha_update_a_matrix_convection(double a, int row_index, int column_index)
{ 
    alpha_a_matrix_convection(row_index, column_index) = alpha_a_matrix_convection(row_index, column_index) + a;                 
}                                                                         

void PV_coupling::alpha_update_b_vector_convection(double a, int index)
{
    alpha_b_vector_convection(index) = alpha_b_vector_convection(index) + a;           
}

void PV_coupling::alpha_combine_a_and_b_matrices()
{
    for(int i = 0; i<alpha_a_matrix_combined.rows(); i++)
    {
        for(int j=0; j<alpha_a_matrix_combined.cols(); j++)
        {
            alpha_a_matrix_combined(i,j) = alpha_a_matrix_rate_of_change(i,j) + alpha_a_matrix_convection(i,j);
        }
    }

    for(int k =0 ; k <alpha_b_vector_combined.size(); k++)
    {
        alpha_b_vector_combined(k) = alpha_b_vector_rate_of_change(k) + alpha_b_vector_source(k) + alpha_b_vector_convection(k) ;
    }
}


// Phase Fraction Continuity - Temporal Derivative Term
 void PV_coupling::alpha_rate_of_change_discretization(std::vector<Cell> &list_of_cells, double &delta_time)                                                               
 {
      alpha_reset_rate_of_change();  

      for(int i = 0; i<list_of_cells.size(); i++)
      {  
            int cell_index = list_of_cells[i].get_cell_index();
            double cell_vol = list_of_cells[cell_index].get_cell_volume();
            double alpha_old = alpha.scalar_field_values(cell_index);

            double rate_of_change_diagonal_contribution = cell_vol/delta_time;
            double rate_of_change_source_contribution = (cell_vol * alpha_old)/ delta_time;

            alpha_update_a_matrix_rate_of_change(rate_of_change_diagonal_contribution, cell_index);
            alpha_update_b_vector_rate_of_change(rate_of_change_source_contribution, cell_index);
      }
 } 


// Phase Fraction Continuity - Convection Term
 void PV_coupling::alpha_convection_discretization(std::vector<Face> &list_of_faces, std::vector<Boundary> &list_of_boundaries, Scalar_boundary_field &alpha_boundary, Vector_boundary_field &velocity_boundary, std::vector<Cell> &list_of_cells)                                                               
 { 
      alpha_reset_convection();  

      Vector_field grad_phi = alpha.compute_gauss_gradient(list_of_cells, list_of_faces);

      for(int i = 0; i<list_of_faces.size(); i++)
        {   
             double face_alpha, dummy_interpolation_factor;
             Vector face_velocity; 

             double alpha_c, alpha_d, grad_cx, grad_cy, grad_cz, grad_c_dot_d, numerator_phic_c, denominator_phi_c, phi_c_ratio, phi_c_tilda;
             Vector distance, distance_c, distance_d;

             int face_index = list_of_faces[i].get_face_index();

             double a_owner_diagonal, a_neighbour_diagonal, a_owner_neighbour, a_neighbour_owner, b_owner_source, b_neighbour_source;
             double neighbour_velocity_x, neighbour_velocity_y, neighbour_velocity_z;

             int owner_index = list_of_faces[i].get_owner_index();
             int neighbour_index = list_of_faces[i].get_neighbour_index();

             double interpolation_factor = list_of_faces[i].get_face_interpolation_factor();
             dummy_interpolation_factor = interpolation_factor;
             double face_delta = list_of_faces[i].get_face_interpolation_factor();

            if(list_of_faces[i].get_is_boundary_face() == true)
            {
                int boundary_index = list_of_faces[i].get_boundary_index();
                Boundary b = list_of_boundaries[list_of_faces[i].get_boundary_index()];

                std::vector<std::vector<double>> vel = velocity_boundary.get_vector_boundary_field_values();

                face_velocity = Vector(vel[boundary_index][0], vel[boundary_index][1], vel[boundary_index][2]);
            }

            else
            {      
                double owner_vx = velocity.vector_field_values_x(owner_index);
                double owner_vy = velocity.vector_field_values_y(owner_index);
                double owner_vz = velocity.vector_field_values_z(owner_index);
                double neighbour_vx = velocity.vector_field_values_x(neighbour_index);
                double neighbour_vy = velocity.vector_field_values_y(neighbour_index);
                double neighbour_vz = velocity.vector_field_values_z(neighbour_index);
                Vector owner_velocity(owner_vx, owner_vy, owner_vz);
                Vector neighbour_velocity(neighbour_vx, neighbour_vy, neighbour_vz);
                face_velocity = owner_velocity * interpolation_factor + neighbour_velocity * (1 - interpolation_factor);
            }

                Vector face_normal = list_of_faces[i].get_face_normal();

                double velocity_flux = face_normal.dot_product(face_velocity);

                double face_normal_magnitude = face_normal.magnitude_of_vector();

              if(list_of_faces[i].get_is_boundary_face() != true)   
              {
                if(velocity_flux >= 0)
               {
                   alpha_c = alpha.scalar_field_values(owner_index);
                   alpha_d = alpha.scalar_field_values(neighbour_index);
                   grad_cx = grad_phi.vector_field_values_x(owner_index); 
                   grad_cy = grad_phi.vector_field_values_y(owner_index);
                   grad_cz = grad_phi.vector_field_values_z(owner_index);
                   Vector grad_c (grad_cx, grad_cy, grad_cz); 
                   distance_d = list_of_cells[neighbour_index].get_cell_centre();
                   distance_c = list_of_cells[owner_index].get_cell_centre();
                   distance = distance_d - distance_c;

                   grad_c_dot_d = grad_c.dot_product(distance);
                   numerator_phic_c = alpha_d - alpha_c;
                   denominator_phi_c = 2 * grad_c_dot_d;
                   phi_c_ratio = (numerator_phic_c/denominator_phi_c);
                   phi_c_tilda = 1.0 - phi_c_ratio;

                  if(denominator_phi_c == 0)
                   {
                    phi_c_tilda = 1.0;
                   }

                }  

                else
                {
                   alpha_c = alpha.scalar_field_values(neighbour_index);
                   alpha_d = alpha.scalar_field_values(owner_index);

                   grad_cx = grad_phi.vector_field_values_x(neighbour_index); 
                   grad_cy = grad_phi.vector_field_values_y(neighbour_index);
                   grad_cz = grad_phi.vector_field_values_z(neighbour_index);
                   Vector grad_c (grad_cx, grad_cy, grad_cz);
                   distance_d = list_of_cells[owner_index].get_cell_centre();
                   distance_c = list_of_cells[neighbour_index].get_cell_centre();
                   distance = distance_d - distance_c;

                   grad_c_dot_d = grad_c.dot_product(distance);
                   //std::cout<<grad_c_dot_d<<std::endl;
                   numerator_phic_c = alpha_d - alpha_c;
                   denominator_phi_c = 2 * grad_c_dot_d;
                   phi_c_ratio = (numerator_phic_c/denominator_phi_c);
                   phi_c_tilda = 1.0 - phi_c_ratio;

                   if(denominator_phi_c == 0)
                   {
                    phi_c_tilda = 1.0;
                   }
                }  
             

              double gamma_phi_c = phi_c_tilda/(0.5*0.5);

             if((phi_c_tilda >0)&&(phi_c_tilda <0.5))
               {
                if(velocity_flux >= 0)
                {
                 interpolation_factor = 1.0 - (gamma_phi_c/2.0);   
                }

                else
                {
                interpolation_factor = (gamma_phi_c/2.0);
                }

               }

             if((phi_c_tilda >= 0.5) && (phi_c_tilda <1))
              {
                    
                if(velocity_flux >= 0)
                {
                    interpolation_factor = 0;
                }

                else
                {
                    interpolation_factor = 1;
                }
              }

              if((phi_c_tilda<=0) || (phi_c_tilda>=1)) 
               {
                if(velocity_flux >= 0)
                {
                    interpolation_factor = 1;
                }

                else
                {
                    interpolation_factor = 0;
                }

               }

              }

           
                 if(list_of_faces[i].get_is_boundary_face() == true)
                {
                     
                     int boundary_index = list_of_faces[i].get_boundary_index();
                     Boundary b = list_of_boundaries[boundary_index];

                    if(alpha_boundary.get_scalar_boundary_name(boundary_index) == "empty")
                     {
                        continue;
                     }

                     else
                     {

                        if(alpha_boundary.get_scalar_boundary_type(boundary_index) == "dirichlet")
                        {   
                            std::vector<double> dirichlet_value = alpha_boundary.get_scalar_boundary_field_values();

                            b_owner_source = -1*velocity_flux * dirichlet_value[boundary_index];
                            alpha_update_b_vector_convection(b_owner_source, owner_index);
                        }

                        if(alpha_boundary.get_scalar_boundary_type(boundary_index) == "neumann")
                        {
                            std::vector<double> neumann_gradient = alpha_boundary.get_scalar_boundary_field_values();

                            a_owner_diagonal = velocity_flux;
                            alpha_update_a_matrix_convection(a_owner_diagonal, owner_index, owner_index);

                            b_owner_source = -1*(velocity_flux/face_delta) * neumann_gradient[boundary_index];
                            alpha_update_b_vector_convection(b_owner_source, owner_index);
                        }
                     }
                }

                else
                {       

                  a_owner_diagonal = interpolation_factor*velocity_flux;
                  a_neighbour_diagonal = -1*(1 - interpolation_factor)*velocity_flux;

                  alpha_update_a_matrix_convection(a_owner_diagonal, owner_index, owner_index);
                  alpha_update_a_matrix_convection(a_neighbour_diagonal, neighbour_index, neighbour_index);

                  a_owner_neighbour = (1 - interpolation_factor)*velocity_flux;
                  a_neighbour_owner = -1*velocity_flux*interpolation_factor;
                   
                  alpha_update_a_matrix_convection(a_owner_neighbour, owner_index, neighbour_index);
                  alpha_update_a_matrix_convection(a_neighbour_owner, neighbour_index, owner_index);
                   
                } 
        }
    }  



// Phase Fraction Continuity - Combining and Solving Matrices
void PV_coupling::alpha_combine_and_solve_matrices(std::vector<Cell> list_of_cells)
{
    alpha_combine_a_and_b_matrices();

    int size = list_of_cells.size();

    SpMat Au(size, size);


    Au.reserve(Eigen::VectorXi::Constant(size,5));
    for(int row=0; row<size; row++)
    {
      for(int col = 0; col < size; col++ )
      {
        if(fabs(alpha_a_matrix_combined(row,col)) > 1e-10)
        {
           Au.insert(row,col) = alpha_a_matrix_combined(row,col);
        }
      }
    }
    Au.makeCompressed();
    
    Eigen::BiCGSTAB<SpMat> solver;
    solver.setTolerance(1e-10);
    solver.compute(Au);

    alpha.scalar_field_values = solver.solve(alpha_b_vector_combined);  
}


// Updating density and viscosity
void PV_coupling::alpha_update_rho_nu(std::vector<Cell> &list_of_cells, double &rho_1, double &rho_2, double &nu_1, double &nu_2, double &mu_1, double &mu_2)
{
    rho.set_old_scalar_field_values();
    nu.set_old_scalar_field_values();
    mu.set_old_scalar_field_values();

    for(int i=0; i<list_of_cells.size(); i++)
    {
        int cell_index = list_of_cells[i].get_cell_index();

        double alpha_cell = alpha.scalar_field_values(cell_index);

        rho.scalar_field_values(cell_index) = (alpha_cell * rho_1) + ((1.0 - alpha_cell ) * rho_2);
        nu.scalar_field_values(cell_index) = (alpha_cell * nu_1) + ((1.0 - alpha_cell ) * nu_2);
        mu.scalar_field_values(cell_index) = (alpha_cell * mu_1) + ((1.0 - alpha_cell ) * mu_2);
    }
}


// // Phase Fraction Continuity - Matrix coefficients
void PV_coupling::alpha_output_scalar_matrix_coefficients_to_file(double total_cells)
{
     std::ofstream alpha_convection_coeffs_a("alpha_convection_coeffs_a.txt");
     std::ofstream alpha_convection_coeffs_b("alpha_convection_coeffs_b.txt");
     std::ofstream alpha_roc_coeffs_a("alpha_roc_coeffs_a.txt");
     std::ofstream alpha_roc_coeffs_b("alpha_roc_coeffs_b.txt");
     std::ofstream alpha_source_coeffs_b("alpha_source_coeffs_b.txt");
     std::ofstream alpha_combined_coeffs_a("alpha_combined_coeffs_a.txt");
     std::ofstream alpha_combined_coeffs_b("alpha_combined_coeffs_b.txt");
    
     for(int i = 0; i<total_cells; i++)
     {
        for(int j=0; j<total_cells; j++)
              {
                if(j < total_cells - 1)
                {
                 alpha_convection_coeffs_a<<alpha_a_matrix_convection(i,j)<<"\t";
                 alpha_roc_coeffs_a<<alpha_a_matrix_rate_of_change(i,j)<<"\t";
                 alpha_combined_coeffs_a<<alpha_a_matrix_combined(i,j)<<"\t";

                }

                else
                {
                 alpha_convection_coeffs_a<<alpha_a_matrix_convection(i,j)<<"\n";
                 alpha_roc_coeffs_a<<alpha_a_matrix_rate_of_change(i,j)<<"\n";
                 alpha_combined_coeffs_a<<alpha_a_matrix_combined(i,j)<<"\n";

                }
              }
     }

     for(int k=0; k<total_cells; k++)
     {
                alpha_convection_coeffs_b<<alpha_b_vector_convection(k)<<"\n";
                alpha_roc_coeffs_b<<alpha_b_vector_rate_of_change(k)<<"\n";
                alpha_source_coeffs_b<<alpha_b_vector_source(k)<<"\n";
                alpha_combined_coeffs_b<<alpha_b_vector_combined(k)<<"\n";

     }

  alpha_convection_coeffs_a.close();
   alpha_convection_coeffs_b.close();
   alpha_roc_coeffs_a.close();
   alpha_roc_coeffs_b.close();
   alpha_combined_coeffs_a.close();
   alpha_combined_coeffs_b.close();
   alpha_source_coeffs_b.close();
}


//Courant number display
void PV_coupling::display_courant_number_details(std::vector<Cell>&list_of_cells, std::ofstream &file_courant_minmax, std::ofstream &file_courant_x_minmax,std::ofstream &file_courant_y_minmax)
{
    std::ofstream file_courant("courant.txt");
    std::ofstream file_courant_x("courant_x.txt");
    std::ofstream file_courant_y("courant_y.txt");

    // std::ofstream file_courant_minmax("courant_minmax.txt");
    // std::ofstream file_courant_x_minmax("courant_x_minmax.txt");
    // std::ofstream file_courant_y_minmax("courant_y_minmax.txt");


  double delta_dist = 0.0025, delta_time = 0.0001;
  double delta_t_by_delta_x = delta_time/delta_dist ;
  double courant, courant_x, courant_y;
  double courant_min = 1.0, courant_x_min = 1.0, courant_y_min = 1.0;
  double courant_max = 0.0, courant_x_max = 0.0, courant_y_max = 0.0;

  double velocity_x, velocity_y, velocity_mag;

   for(int i=0; i<list_of_cells.size(); i++)
   {
      velocity_x = velocity.vector_field_values_x(i);
      velocity_y = velocity.vector_field_values_y(i);

      velocity_mag = sqrt((velocity_x * velocity_x) + (velocity_y * velocity_y));

      courant = velocity_mag  * delta_t_by_delta_x;
      courant_x = velocity_x * delta_t_by_delta_x;
      courant_y = velocity_y * delta_t_by_delta_x;

      file_courant<<courant<<std::endl;
      file_courant_x<<courant_x<<std::endl;
      file_courant_y<<courant_y<<std::endl;

      if(courant < courant_min)
      {
        courant_min = courant;
      }

      if(courant > courant_max)
      {
        courant_max = courant;
      }

      if(courant_x < courant_x_min)
      {
        courant_x_min = courant_x;
      }

      if(courant_x> courant_x_max)
      {
        courant_x_max = courant_x;
      }

            if(courant_y < courant_y_min)
      {
        courant_y_min = courant_y;
      }

      if(courant_y> courant_y_max)
      {
        courant_y_max = courant_y;
      }

   }

   file_courant_minmax<<"Min:"<<courant_min<<" "<<"Max:"<<courant_max<<std::endl;
   file_courant_x_minmax<<"Min:"<<courant_x_min<<" "<<"Max:"<<courant_x_max<<std::endl;
   file_courant_y_minmax<<"Min:"<<courant_y_min<<" "<<"Max:"<<courant_y_max<<std::endl;

   file_courant.close();
   file_courant_x.close();
   file_courant_y.close();

  //  file_courant_minmax.close();
  //  file_courant_x_minmax.close();
  //  file_courant_y_minmax.close();
}



// Function to output velocity, phase fraction and pressure to data files at regular pre-set intervals
void PV_coupling::alpha_and_velocity_output_vector_field_to_file(std::vector<double> x_distance, std::vector<double> y_distance, int iteration_no, int output_interval)
{
     if((iteration_no % output_interval) == 0)
    { 
      std::string str_num_vx = std::to_string(iteration_no) + "_vx.txt";
      std::string str_num_vy = std::to_string(iteration_no) + "_vy.txt";
      std::string str_num_v_mag = std::to_string(iteration_no) + "_v_mag.txt";
      std::string str_num_alpha = std::to_string(iteration_no) + "_alpha.txt";
      std::string str_num_vec_plot = std::to_string(iteration_no) + "_vec_plot.txt";
      std::string str_num_vx_horizontal = std::to_string(iteration_no) + "_vx_horizontal.txt";
      std::string str_num_vx_vertical = std::to_string(iteration_no) + "_vx_vertical.txt";
      std::string str_num_vy_horizontal = std::to_string(iteration_no) + "_vy_horizontal.txt";
      std::string str_num_vy_vertical = std::to_string(iteration_no) + "_vy_vertical.txt";

      std::string str_num_v_mag_left = std::to_string(iteration_no) + "v_mag_left.txt";
      std::string str_num_v_mag_right = std::to_string(iteration_no) + "v_mag_right.txt";
      std::string str_num_v_mag_bottom = std::to_string(iteration_no) + "v_mag_bottom.txt";
      std::string str_num_v_mag_horizontal = std::to_string(iteration_no) + "v_mag_horizontal.txt";
      std::string str_num_v_mag_vertical = std::to_string(iteration_no) + "v_mag_vertical.txt";

      std::string str_num_pressure = std::to_string(iteration_no) + "pressure.txt";
      std::string str_num_pressure_vertical = std::to_string(iteration_no) + "pressure_vertical.txt";
      std::string str_num_pressure_horizontal = std::to_string(iteration_no) + "pressure_horizontal.txt";
      std::string str_num_pressure_left = std::to_string(iteration_no) + "pressure_left.txt";
      std::string str_num_pressure_right = std::to_string(iteration_no) + "pressure_right.txt";
      std::string str_num_pressure_bottom = std::to_string(iteration_no) + "pressure_bottom.txt";

      std::string str_num_alpha_vertical = std::to_string(iteration_no) + "alpha_vertical.txt";
      std::string str_num_alpha_horizontal = std::to_string(iteration_no) + "alpha_horizontal.txt";
      std::string str_num_alpha_left = std::to_string(iteration_no) + "alpha_left.txt";
      std::string str_num_alpha_right = std::to_string(iteration_no)+ "alpha_right.txt";
      std::string str_num_alpha_bottom = std::to_string(iteration_no) + "alpha_bottom.txt";

        std::ofstream vector_field_profiles_x(str_num_vx);
        for(int i=0; i<y_distance.size(); i++)
         {
              for(int j=0; j<x_distance.size(); j++)
              {  
                vector_field_profiles_x<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity.vector_field_values_x(j + i*x_distance.size())<<std::endl;
                //std::cout<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity.vector_field_values_x(j + i*x_distance.size())<<std::endl;
              }
         }

        std::ofstream vector_field_profiles_y(str_num_vy);
        for(int i=0; i<y_distance.size(); i++)
         {
              for(int j=0; j<x_distance.size(); j++)
              {
                vector_field_profiles_y<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<velocity.vector_field_values_y(j + i*x_distance.size())<<std::endl;
              }
         }

        std::ofstream vector_field_profiles_xy(str_num_v_mag);
        // for(int i=0; i<y_distance.size(); i++)
        //  {
        //       for(int j=0; j<x_distance.size(); j++)
        //       {
        //          double vx = velocity.vector_field_values_x(j + i*x_distance.size());
        //          double vy = velocity.vector_field_values_y(j + i*x_distance.size());
        //          double v_mag = sqrt(vx*vx +  vy*vy);

        //         vector_field_profiles_xy<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<v_mag<<std::endl;
        //       }
        //  }

        std::ofstream pressure_profiles(str_num_pressure); 
        for(int i=0; i<y_distance.size(); i++)
         {
              for(int j=0; j<x_distance.size(); j++)
              {  
                pressure_profiles<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<pressure.scalar_field_values(j + i*x_distance.size())<<std::endl;
              }
         }

        std::ofstream alpha_profiles(str_num_alpha);
        for(int i=0; i<y_distance.size(); i++)
         {
              for(int j=0; j<x_distance.size(); j++)
              {  
                alpha_profiles<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<alpha.scalar_field_values(j + i*x_distance.size())<<std::endl;
                 //alpha_profiles<<y_distance[j]<<"\t"<<alpha.scalar_field_values(j)<<std::endl;
              }
         }

        std::ofstream vector_plots(str_num_vec_plot);
        // for(int i=0; i<y_distance.size(); i++)
        //  {
        //       for(int j=0; j<x_distance.size(); j++)
        //       {
        //          double vx = velocity.vector_field_values_x(j + i*x_distance.size());
        //          double vy = velocity.vector_field_values_y(j + i*x_distance.size());
        //          double v_mag = sqrt(vx*vx +  vy*vy);

        //         vector_plots<<x_distance[j]<<"\t"<<y_distance[i]<<"\t"<<pressure.scalar_field_values(j + i*x_distance.size())<<"\t"<<vx/v_mag<<"\t"<<vy/v_mag<<std::endl;
        //       }
        //  }

         std::ofstream vx_horizontal(str_num_vx_horizontal);
         std::ofstream vy_horizontal(str_num_vy_horizontal);
         std::ofstream velocity_mag_horizontal(str_num_v_mag_horizontal);

        // for(int j=0; j<x_distance.size(); j++)
        //  {
        //     double vx_left = velocity.vector_field_values_x(((y_distance.size()/2) - 1.0)*x_distance.size() + j);
        //     double vx_right = velocity.vector_field_values_x((y_distance.size()/2)*x_distance.size() + j); 
        //     double vx_mid = (vx_left + vx_right)/2.0;

        //     double vy_left = velocity.vector_field_values_y(((y_distance.size()/2) - 1.0)*x_distance.size() + j);
        //     double vy_right = velocity.vector_field_values_y((y_distance.size()/2)*x_distance.size() + j); 
        //     double vy_mid = (vy_left + vy_right)/2.0;

        //     double v_mag_horz = sqrt(vx_mid*vx_mid + vy_mid*vy_mid);

        //     vx_horizontal<<x_distance[j]<<"\t"<<vx_mid<<std::endl;
        //     vy_horizontal<<x_distance[j]<<"\t"<<vy_mid<<std::endl;
        //     velocity_mag_horizontal<<x_distance[j]<<"\t"<<v_mag_horz<<std::endl;
        //  }

        std::ofstream alpha_horizontal(str_num_alpha_horizontal);
        // for(int j=0; j<x_distance.size(); j++)
        //  {
        //     double alpha_left = alpha.scalar_field_values(((y_distance.size()/2) - 1.0)*x_distance.size() + j);
        //     double alpha_right = alpha.scalar_field_values((y_distance.size()/2)*x_distance.size() + j); 
        //     double alpha_mid = (alpha_left + alpha_right)/2.0;

        //     alpha_horizontal<<x_distance[j]<<"\t"<<alpha_mid<<std::endl;
        //  }

        std::ofstream  pressure_horizontal(str_num_pressure_horizontal);
        // for(int j=0; j<x_distance.size(); j++)
        //  {
        //     double  pressure_left =  pressure.scalar_field_values(((y_distance.size()/2) - 1.0)*x_distance.size() + j);
        //     double  pressure_right =  pressure.scalar_field_values((y_distance.size()/2)*x_distance.size() + j); 
        //     double  pressure_mid = (pressure_left +  pressure_right)/2.0;

        //     pressure_horizontal<<x_distance[j]<<"\t"<<pressure_mid<<std::endl;
        //  }
 
         std::ofstream vx_vertical(str_num_vx_vertical);
         std::ofstream vy_vertical(str_num_vy_vertical);
         std::ofstream velocity_mag_vertical(str_num_v_mag_vertical);
        // for(int j=0; j<y_distance.size(); j++)
        //  {
        //     double vx_left = velocity.vector_field_values_x(((x_distance.size()/2) - 1.0) + (y_distance.size() * j));
        //     double vx_right = velocity.vector_field_values_x(((x_distance.size()/2)) + (y_distance.size() * j));
        //     double vx_mid = (vx_left + vx_right)/2.0;

        //     double vy_left = velocity.vector_field_values_y(((x_distance.size()/2) - 1.0) + (y_distance.size() * j));
        //     double vy_right = velocity.vector_field_values_y(((x_distance.size()/2)) + (y_distance.size() * j));
        //     double vy_mid = (vy_left + vy_right)/2.0;

        //     double v_mag_vert = sqrt(vx_mid*vx_mid + vy_mid*vy_mid);

        //     vx_vertical<<y_distance[j]<<"\t"<<vx_mid<<std::endl;
        //     vy_vertical<<y_distance[j]<<"\t"<<vy_mid<<std::endl;
        //     velocity_mag_vertical<<y_distance[j]<<"\t"<<v_mag_vert<<std::endl;
        //  }

        std::ofstream alpha_vertical(str_num_alpha_vertical);
        for(int j=0; j<y_distance.size(); j++)
         {
          //  double alpha_left = alpha.scalar_field_values(((x_distance.size()/2) - 1.0) + (y_distance.size() * j));
          //  double alpha_right = alpha.scalar_field_values(((x_distance.size()/2)) + (y_distance.size() * j));
          //  double alpha_mid = (alpha_left + alpha_right)/2.0;

            //alpha_vertical<<y_distance[j]<<"\t"<<alpha_mid<<std::endl;
         }

        std::ofstream pressure_vertical(str_num_pressure_vertical);
        for(int j=0; j<y_distance.size(); j++)
         {
           // double  pressure_left =  pressure.scalar_field_values(((x_distance.size()/2) - 1.0) + (y_distance.size() * j));
           // double  pressure_right =  pressure.scalar_field_values(((x_distance.size()/2))*y_distance.size() + j); 
           // double  pressure_mid = (pressure_left +  pressure_right)/2.0;

            //pressure_vertical<<y_distance[j]<<"\t"<< pressure_mid<<std::endl;
         }

        
        std::ofstream velocity_mag_left(str_num_v_mag_left);
        for(int j=0; j<y_distance.size(); j++)
         {
           // double v_x = velocity.vector_field_values_x(((x_distance.size()/5) + 3) + (x_distance.size()*j));
          //  double v_y = velocity.vector_field_values_y(((x_distance.size()/5) + 3) + (x_distance.size()*j)); 
           // double v_mid = sqrt(v_x*v_x + v_y*v_y);

           // velocity_mag_left<<y_distance[j]<<"\t"<<v_mid<<std::endl;
         }

        std::ofstream alpha_left(str_num_alpha_left);
        for(int j=0; j<y_distance.size(); j++)
         {
          //  double alpha_val = alpha.scalar_field_values(((x_distance.size()/5) + 3) + (x_distance.size()*j));
          //  alpha_left<<y_distance[j]<<"\t"<<alpha_val<<std::endl;
         }

        std::ofstream pressure_left(str_num_pressure_left);
        for(int j=0; j<y_distance.size(); j++)
         {
          //  double pressure_val = pressure.scalar_field_values(((x_distance.size()/5) + 3) + (x_distance.size()*j));
            //pressure_left<<y_distance[j]<<"\t"<<pressure_val<<std::endl;
         }


        std::ofstream velocity_mag_right(str_num_v_mag_right);
        for(int j=0; j<y_distance.size(); j++)
         {
           // double v_x = velocity.vector_field_values_x(((x_distance.size()*0.8) + 1) + (x_distance.size()*j));
            //double v_y = velocity.vector_field_values_y(((x_distance.size()*0.8) + 1) + (x_distance.size()*j)); 
            //double v_mid = sqrt(v_x*v_x + v_y*v_y);

           // velocity_mag_right<<y_distance[j]<<"\t"<<v_mid<<std::endl;
         }

        std::ofstream alpha_right(str_num_alpha_right);
        for(int j=0; j<y_distance.size(); j++)
         {
            //double alpha_val = alpha.scalar_field_values(((x_distance.size()*0.8) + 1) + (x_distance.size()*j));
           // alpha_right<<y_distance[j]<<"\t"<<alpha_val<<std::endl;
         }


        std::ofstream pressure_right(str_num_pressure_right);
        for(int j=0; j<y_distance.size(); j++)
         {
            //double pressure_val = pressure.scalar_field_values(((x_distance.size()*0.8) + 1) + (x_distance.size()*j));
           // pressure_right<<y_distance[j]<<"\t"<<pressure_val<<std::endl;
         }

        std::ofstream velocity_mag_bottom(str_num_v_mag_bottom);
        for(int j=0; j<x_distance.size(); j++)
         {
           //  double v_x = velocity.vector_field_values_x((((y_distance.size()*0.2) - 5)*x_distance.size()) + j);
           //  double v_y = velocity.vector_field_values_y((((y_distance.size()/5) - 5)*x_distance.size()) + j);
           //  double v_mid = sqrt(v_x*v_x + v_y*v_y);

            // velocity_mag_bottom<<x_distance[j]<<"\t"<<v_mid<<std::endl;
         }

        std::ofstream alpha_bottom(str_num_alpha_bottom);
        for(int j=0; j<x_distance.size(); j++)
         {
            // double alpha_val = alpha.scalar_field_values((((y_distance.size()/5) - 5)*x_distance.size()) + j);
            // alpha_bottom<<x_distance[j]<<"\t"<<alpha_val<<std::endl;
         }

        std::ofstream pressure_bottom(str_num_pressure_bottom);
        for(int j=0; j<x_distance.size(); j++)
         {
            // double pressure_val = pressure.scalar_field_values((((y_distance.size()/5) - 5)*x_distance.size()) + j);
            // pressure_bottom<<x_distance[j]<<"\t"<<pressure_val<<std::endl;
         }

        vector_field_profiles_x.close();
        vector_field_profiles_y.close();
        vector_field_profiles_xy.close();
        alpha_profiles.close();
        vector_plots.close();
        vx_horizontal.close();
        vx_vertical.close();
        vy_horizontal.close();
        vy_vertical.close();
         velocity_mag_left.close();
         velocity_mag_right.close();
         velocity_mag_bottom.close();
         velocity_mag_horizontal.close();
         velocity_mag_vertical.close();
         pressure_profiles.close();
         alpha_vertical.close();
         alpha_horizontal.close();
        alpha_left.close();
         alpha_right.close();
         alpha_bottom.close();
         pressure_vertical.close();
         pressure_horizontal.close();
         pressure_left.close();
         pressure_right.close();     
         pressure_bottom.close();
    }
} 

