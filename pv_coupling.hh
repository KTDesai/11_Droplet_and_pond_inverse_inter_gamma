#ifndef PV_COUPLING_HH
#define PV_COUPLING_HH

#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<vector>
#include<cmath>

#include "cell.hh"
#include "boundary.hh"
#include "scalar_field.hh"
#include "scalar_boundary_field.hh"
#include "vector_boundary_field.hh"
#include "vector_field.hh"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;

class PV_coupling
{
  public:

  Scalar_field nu;
  Vector_field velocity;
  Scalar_field pressure;
  Scalar_field alpha;
  Scalar_field rho;
  Scalar_field mu;
  
  Eigen::MatrixXd velocity_a_matrix_rate_of_change;
  Eigen::VectorXd velocity_b_vector_rate_of_change_ux;
  Eigen::VectorXd velocity_b_vector_rate_of_change_uy;

  Eigen::VectorXd velocity_b_vector_source_ux;
  Eigen::VectorXd velocity_b_vector_source_uy;

  Eigen::MatrixXd velocity_a_matrix_diffusion;
  Eigen::VectorXd velocity_b_vector_diffusion_ux;
  Eigen::VectorXd velocity_b_vector_diffusion_uy;

  Eigen::MatrixXd velocity_a_matrix_convection;
  Eigen::VectorXd velocity_b_vector_convection_ux;
  Eigen::VectorXd velocity_b_vector_convection_uy;

  Eigen::MatrixXd velocity_a_matrix_combined;
  Eigen::VectorXd velocity_b_vector_combined_ux;
  Eigen::VectorXd velocity_b_vector_combined_uy;

  Eigen::VectorXd velocity_initial_residuals_ux;
  Eigen::VectorXd velocity_initial_residuals_uy;
  Eigen::VectorXd velocity_final_residuals_ux;
  Eigen::VectorXd velocity_final_residuals_uy;


  Eigen::MatrixXd pressure_a_matrix_rate_of_change;
  Eigen::VectorXd pressure_b_vector_rate_of_change;

  Eigen::VectorXd pressure_b_vector_source;

  Eigen::MatrixXd pressure_a_matrix_diffusion;
  Eigen::VectorXd pressure_b_vector_diffusion;

  Eigen::MatrixXd pressure_a_matrix_convection;
  Eigen::VectorXd pressure_b_vector_convection;

  Eigen::MatrixXd pressure_a_matrix_combined;
  Eigen::VectorXd pressure_b_vector_combined;

  Eigen::VectorXd pressure_initial_residuals_p;
  Eigen::VectorXd pressure_final_residuals_p;

  Eigen::MatrixXd alpha_a_matrix_rate_of_change;
  Eigen::VectorXd alpha_b_vector_rate_of_change;

  Eigen::MatrixXd alpha_a_matrix_convection;
  Eigen::VectorXd alpha_b_vector_convection;

  Eigen::VectorXd alpha_b_vector_source;

  Eigen::MatrixXd alpha_a_matrix_combined;
  Eigen::VectorXd alpha_b_vector_combined;

  PV_coupling(int, double, double, double, double, double, Vector);

  void velocity_update_a_matrix_rate_of_change(double, int);
  void velocity_update_b_vector_rate_of_change_ux(double, int);
  void velocity_update_b_vector_rate_of_change_uy(double, int);

  void velocity_update_b_vector_source_ux(double, int);
  void velocity_update_b_vector_source_uy(double, int);

  void velocity_update_a_matrix_diffusion(double, int, int);
  void velocity_update_b_vector_diffusion_ux(double, int);
  void velocity_update_b_vector_diffusion_uy(double, int);

  void velocity_update_a_matrix_convection(double, int, int);
  void velocity_update_b_vector_convection_ux(double, int);
  void velocity_update_b_vector_convection_uy(double, int);

  void velocity_combine_a_matrices();
  void velocity_combine_b_matrices();

  void velocity_reset_rate_of_change();
  void velocity_reset_source();
  void velocity_reset_diffusion();
  void velocity_reset_convection();
  void velocity_reset_combined();

  void velocity_compute_rate_of_change_matrix(std::vector<Cell>&, double&);
  void velocity_compute_source_matrix(std::vector<Cell>&);
  void velocity_compute_diffusion_matrix(std::vector<Face>&, std::vector<Boundary>&, Vector_boundary_field&);
  void velocity_compute_convection_matrix(std::vector<Face>&, std::vector<Boundary>&, Vector_boundary_field&);
  void velocity_under_relaxation(std::vector<Cell>&, double&);
  void velocity_solve_matrices(std::vector<Cell> );

  void velocity_calculate_initial_residuals(std::vector<double>&, std::vector<double>&, int&, int&, std::ofstream &, std::ofstream &);
  void velocity_calculate_final_residuals(std::vector<double>&, std::vector<double>&, int&, int&, std::ofstream &, std::ofstream &);

  std::vector<double> velocity_store_ap_coefficients();

  void velocity_output_vector_matrix_coefficients_to_file(double);
  void velocity_output_vector_field_to_file(std::vector<double>, std::vector<double>);

  void velocity_correct_cell_fluxes(std::vector<Cell>, std::vector<double>);
  void velocity_correct_cell_centre_velocities(std::vector<Cell>&, Vector_field &obj , std::vector<double>&);

  //std::vector<double> set_face_and_cell_fluxes(std::vector<Cell> &);
  void velocity_set_face_and_cell_fluxes(std::vector<Cell> &,std::vector<Face> &, std::vector<Boundary>&, Vector_boundary_field&);

  void velocity_plot_convergence_initial_x(std::ofstream &, int&);
  void velocity_plot_convergence_initial_y(std::ofstream &, int&);
  void velocity_plot_convergence_final_x(std::ofstream &, int&);
  void velocity_plot_convergence_final_y(std::ofstream &, int&);

  void pressure_update_a_matrix_rate_of_change(double, int);
  void pressure_update_b_vector_rate_of_change(double, int);

  void pressure_update_b_vector_source(double, int);

  void pressure_update_a_matrix_diffusion(double, int, int);
  void pressure_update_b_vector_diffusion(double, int);

  void pressure_update_a_matrix_convection(double, int, int);
  void pressure_update_b_vector_convection(double, int);

  void pressure_combine_a_and_b_matrices();

  void pressure_reset_rate_of_change();
  void pressure_reset_source();
  void pressure_reset_diffusion();
  void pressure_reset_convection();
  void pressure_reset_combined();

  void pressure_compute_source_matrix(std::vector<Cell>&, std::vector<Face>&);

  void pressure_compute_diffusion_matrix(std::vector<Face>&, std::vector<double>&, std::vector<Boundary>&, int&, Scalar_boundary_field&);

  void pressure_compute_convection_matrix(std::vector<Cell>, std::vector<Boundary>, Vector_field obj_2);
  void pressure_combine_and_solve_matrices(std::vector<Cell>&);

  void pressure_under_relax();

  void pressure_calculate_initial_residuals_p(std::vector<double>&, std::vector<double>&, int&, int&, std::ofstream &);
  void pressure_calculate_final_residuals_p(std::vector<double>&, std::vector<double>&, int&, int&,  std::ofstream &);

  void pressure_compute_flux_correction(std::vector<Cell>&, std::vector<Face>&, std::vector<double>&, std::vector<Boundary>&);
  void pressure_compute_velocity_correction_terms(std::vector<Cell>&, std::vector<double>&, std::vector<double>&, std::vector<Boundary>);

  void pressure_output_scalar_matrix_coefficients_to_file(double);
  void pressure_output_scalar_field_to_file(std::vector<double>, std::vector<double>, std::vector<Cell>);
  void pressure_plot_convergence_initial(std::ofstream &, int&);

  void pressure_display_pressure_values(int);

  Scalar_field retrieve_pressure_field();

  void set_alpha_scalar_initial_fields(std::vector<Cell>&, double&, double&, double&, double&, double&, double&, double&, double&);

  void alpha_update_a_matrix_rate_of_change(double, int);
  void alpha_update_b_vector_rate_of_change(double, int);

  void alpha_update_b_vector_source(double, int);

  void alpha_update_a_matrix_convection(double, int, int);
  void alpha_update_b_vector_convection(double, int);

  void alpha_combine_a_and_b_matrices();

  void alpha_reset_rate_of_change();
  void alpha_reset_source();
  void alpha_reset_convection();
  void alpha_reset_combined();

  void alpha_rate_of_change_discretization(std::vector<Cell>&, double&);
  void alpha_convection_discretization(std::vector<Face>&, std::vector<Boundary>&, Scalar_boundary_field&, Vector_boundary_field&, std::vector<Cell> &);

  void alpha_combine_and_solve_matrices(std::vector<Cell> list_of_cells);

  void alpha_update_rho_nu(std::vector<Cell>&, double&, double&, double&, double&, double&, double&);

  void alpha_output_scalar_fields_to_file(std::vector<double>&, std::vector<double>&, std::vector<Cell>&, int&);
  void alpha_output_scalar_matrix_coefficients_to_file(double);

  void velocity_compute_diffusion_matrix_old(std::vector<Face>, std::vector<Boundary>, Vector_boundary_field);
  void velocity_compute_convection_matrix_old(std::vector<Face>, std::vector<Boundary>, Vector_boundary_field);

  void velocity_compute_div_u(std::vector<Cell> &, int&, std::vector<Face>&);

  void display_courant_number_details(std::vector<Cell>&, std::ofstream &,std::ofstream &, std::ofstream &);

  void alpha_and_velocity_output_vector_field_to_file(std::vector<double> x_distance, std::vector<double> y_distance, int iteration_no, int);

};

#endif

















