#include<iostream>
#include<vector>
#include<fstream>
#include<Eigen/Dense>

#include "vector_field.hh"
#include "vector.hh"
#include "face.hh"
#include "cell.hh"
#include "boundary.hh"
#include "mesh.hh"

#include "scalar_field.hh"
#include "scalar_boundary_field.hh"
#include "vector_boundary_field.hh"
#include "pv_coupling.hh"

int main (void)
{   
  /* rows = Number of cells in Vertical Y-Direction
     columns = Number of Cells in Horizontal X-Direction
     alpha_u = under-relaxation factor
     delta_time = Time step
     rho_1, rho_2 = Densities of the fluids 1 and 2 respectively
     nu_1, nu_2 = Kinematic viscosities of fluids 1 and 2 respectively
     mu_1, mu_2 = Dynamic viscosities of fluids 1 and 2 respectively
     output_interval = Number of iterations after which results of velocity, phase fraction and pressure are to be output in text files (inside bin folder)
  */
  
  int rows = 40, columns = 40, total_cells = rows*columns, output_interval = 100; 
 
 // Declaration and initialization
  double delta_time = 0.000005, nu_val = 0.001, alpha_u = 0.7, initial_pressure = 0.0, alpha_initial = 0.0, initial_rho = 0.0, initial_mu = 0.0;
  double rho_1 = 1000.0, rho_2 = 1.0, nu_1 = 0.000001, nu_2 = 0.0000148;

  double mu_1 = nu_1*rho_1;
  double mu_2 = nu_2*rho_2;

  double x_length = 0.1, y_length = 0.1, x_w_max = 0.025, y_w_max = 0.05;
  double delta_x = x_length/columns;
  double delta_y = y_length/rows;

  std::vector<double> temp_ap_coeffs;
  std::vector<double> pressure_corrections_for_flux;
  std::vector<double> pressure_corrections_for_velocity_x(total_cells);
  std::vector<double> pressure_corrections_for_velocity_y(total_cells);
  std::vector<double> cell_fluxes_sum;

  std::ofstream div_u("div_u_cell.txt");

  std::vector<double> x_distance (columns);
  std::vector<double> y_distance (rows);
  
  //Initialization of velocity
  std::vector<double> const_vel = {0.0, 0.0, 0.0};
  Vector const_velocity(const_vel[0], const_vel[1], const_vel[2]);

  PV_coupling cavity(total_cells, nu_val, initial_pressure, alpha_initial, initial_rho, initial_mu, const_velocity); 

  //Mesh generation and calculation of mesh parameters
  Mesh m("../points.txt","../faces.txt","../cells.txt","../boundary.txt");

  std::vector<Cell> list_of_cells = m.return_list_of_all_cells();
  std::vector<Face> list_of_faces = m.return_list_of_all_faces();
  
  for(int i=0; i<columns; i++)
  {
     Vector d_x = list_of_cells[i].get_cell_centre();
     x_distance[i] = d_x[0];
  }

  for(int j=0; j<rows; j++)
  {
    Vector d_y = list_of_cells[j*columns].get_cell_centre();
    y_distance[j] = d_y[1];
  }

  //Scalar Boundary Conditions
  std::vector<std::string> boundary_types_scalar = {"neumann","neumann","neumann","neumann", "neumann"} ;
  std::vector<double> boundary_values_scalar = {0.0, 0.0, 0.0, 0.0, 0.0};

  //Vector Boudnary Conditions
  std::vector<std::string> boundary_types_vector = {"dirichlet","dirichlet","dirichlet","dirichlet","dirichlet"};
  std::vector<std::vector<double>> boundary_values_vector = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  std::vector<Boundary> list_of_boundaries = m.return_list_of_all_boundaries();

  Scalar_boundary_field pressure_boundary (list_of_boundaries, boundary_types_scalar, boundary_values_scalar);
  Scalar_boundary_field alpha_boundary(list_of_boundaries, boundary_types_scalar, boundary_values_scalar);
  Vector_boundary_field velocity_boundary(list_of_boundaries, boundary_types_vector, boundary_values_vector);

  //File object declarations for Residual and Convergence monitoring
  std::ofstream sum_initial_residual_ux("sum_residual_initial_ux.txt");
  std::ofstream sum_initial_residual_uy("sum_residual_initial_uy.txt");

  std::ofstream sum_final_residual_ux("sum_residual_final_ux.txt");
  std::ofstream sum_final_residual_uy("sum_residual_final_uy.txt");

  std::ofstream sum_initial_pressure_residual("sum_initial_pressure_residual.txt");
  std::ofstream sum_final_pressure_residual("sum_final_pressure_residual.txt");

  std::ofstream pressure_convergence_initial_h("pressure_convergence.txt");
  std::ofstream x_velocity_convergence_initial("x_velocity_convergence.txt");
  std::ofstream y_velocity_convergence_initial("y_velocity_convergence.txt");
  std::ofstream x_velocity_convergence_final("x_velocity_convergence_final.txt");
  std::ofstream y_velocity_convergence_final("y_velocity_convergence_final.txt");

  std::ofstream file_courant_minmax("courant_minmax.txt");
  std::ofstream file_courant_x_minmax("courant_x_minmax.txt");
  std::ofstream file_courant_y_minmax("courant_y_minmax.txt");

  int iteration_no = 0;

  // Initialization of Phase-Fraction Fields
  cavity.set_alpha_scalar_initial_fields(list_of_cells, rho_1, rho_2, nu_1, nu_2, mu_1, mu_2, x_w_max, y_w_max);

   while(iteration_no <50000000)
   {
     std::cout<<"Iteration Number : "<<iteration_no<<std::endl;
     iteration_no = iteration_no +1; 

     //cavity.display_courant_number_details(list_of_cells, file_courant_minmax, file_courant_x_minmax, file_courant_y_minmax);


     //Sum of Divergence of Velocity for all cells computed to check blowing up of simulation   
     cavity.velocity_compute_div_u(list_of_cells, total_cells, list_of_faces); 

     //Phase Continuity Equation Discretization

     //Temporal Derivative Term
     cavity.alpha_rate_of_change_discretization(list_of_cells, delta_time);

     //Convection Term
     cavity.alpha_convection_discretization(list_of_faces, list_of_boundaries, alpha_boundary, velocity_boundary, list_of_cells);

     //Numerically Solving Phase Continuity Equation
     cavity.alpha_combine_and_solve_matrices(list_of_cells);

     //Updating Density and Viscosity
     cavity.alpha_update_rho_nu(list_of_cells, rho_1, rho_2, nu_1, nu_2, mu_1, mu_2);

     
     //Momentum Equation Discretization

     //Temporal Derivative Term
     cavity.velocity_compute_rate_of_change_matrix(list_of_cells, delta_time);

     //Diffusion Term
     cavity.velocity_compute_diffusion_matrix(list_of_faces, list_of_boundaries, velocity_boundary);

     //Convection Term
     cavity.velocity_compute_convection_matrix(list_of_faces, list_of_boundaries, velocity_boundary);

     // Source Term (Gravity Term)
     cavity.velocity_compute_source_matrix(list_of_cells);

     //Combining all discretized Matrices
     cavity.velocity_combine_a_matrices();
     cavity.velocity_combine_b_matrices();

     //Under-relaxing cobined matrix
     cavity.velocity_under_relaxation(list_of_cells, alpha_u); 

     //Initial velocity residual computation
     cavity.velocity_calculate_initial_residuals(x_distance, y_distance, iteration_no, output_interval, sum_initial_residual_ux, sum_initial_residual_uy);

     //Solving for velocity (Momentum predictor)  
     cavity.velocity_solve_matrices(list_of_cells);

     //Velocity Convergence  
     cavity.velocity_plot_convergence_initial_x(x_velocity_convergence_initial, iteration_no);
     cavity.velocity_plot_convergence_initial_y(y_velocity_convergence_initial, iteration_no);

     //Final Velocity Residuals
     cavity.velocity_calculate_final_residuals(x_distance, y_distance, iteration_no, output_interval, sum_final_residual_ux, sum_final_residual_uy);

      for(int i = 0; i < list_of_cells.size(); i++)
      {
        cavity.velocity_a_matrix_combined(i,i) *= alpha_u;
      }

    temp_ap_coeffs = cavity.velocity_store_ap_coefficients();
     
    //Pressure Equation (Pressure Corrector)

    // Computing sum of fluxes for all cells (to be used as source term for pressure equation)
    cavity.velocity_set_face_and_cell_fluxes(list_of_cells, list_of_faces, list_of_boundaries, velocity_boundary);  

    //DIffusion Term
    cavity.pressure_compute_diffusion_matrix(list_of_faces, temp_ap_coeffs, list_of_boundaries, iteration_no, pressure_boundary);  

    //Source term
    cavity.pressure_compute_source_matrix(list_of_cells, list_of_faces);     

    //Pressure Initial Residuals                
    cavity.pressure_calculate_initial_residuals_p(x_distance, y_distance, iteration_no, output_interval, sum_initial_pressure_residual);    

    //Solving Pressure Corrector Equation                        
    cavity.pressure_combine_and_solve_matrices(list_of_cells);

    //Pressure Convergence Initial
    cavity.pressure_plot_convergence_initial(pressure_convergence_initial_h, iteration_no);
     
    //Pressure Final Residuals
    cavity.pressure_calculate_final_residuals_p(x_distance, y_distance, iteration_no, output_interval, sum_final_pressure_residual);
     
    //Flux Correction
    cavity.pressure_compute_flux_correction(list_of_cells, list_of_faces, temp_ap_coeffs, list_of_boundaries);

    Scalar_field obj = cavity.retrieve_pressure_field();

    // Computation of Gauss Gradient for Velocity Correction
    Vector_field grad_p = obj.compute_gauss_gradient(list_of_cells, list_of_faces);   

    //Velocity corrections
    cavity.velocity_correct_cell_centre_velocities(list_of_cells, grad_p, temp_ap_coeffs);

    //Pressure under-relaxation
    cavity.pressure_under_relax();

    //cavity.alpha_output_scalar_fields_to_file(x_distance, y_distance, list_of_cells, iteration_no);

    //Function to output alpha, velocity and pressure fields into text files at regular intervals 
    cavity.alpha_and_velocity_output_vector_field_to_file(x_distance, y_distance, iteration_no, output_interval);
  
  }
 
  // Functions to output final matrix coefficients to files
  cavity.velocity_output_vector_matrix_coefficients_to_file(total_cells);
  
  cavity.pressure_output_scalar_matrix_coefficients_to_file(total_cells);

  cavity.alpha_output_scalar_matrix_coefficients_to_file(total_cells);  

  div_u.close();

  sum_initial_residual_ux.close();
  sum_initial_residual_uy.close();
  sum_final_residual_ux.close();
  sum_final_residual_uy.close();
  sum_initial_pressure_residual.close();
  sum_final_pressure_residual.close(); 
  pressure_convergence_initial_h.close();
  x_velocity_convergence_initial.close();
  y_velocity_convergence_initial.close();
  x_velocity_convergence_final.close();
  y_velocity_convergence_final.close();

  file_courant_minmax.close();
  file_courant_x_minmax.close();
  file_courant_y_minmax.close();

  return 0;
}
