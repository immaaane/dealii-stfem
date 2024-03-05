#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

namespace dealii
{
  void
  tensorproduct_add(BlockVectorType          &c,
                    FullMatrix<Number> const &A,
                    VectorType const         &b,
                    int                       block_offset = 0)
  {
    const unsigned int n_blocks = A.m();
    for (unsigned int i = 0; i < n_blocks; ++i)
      if (A(i, 0) != 0.0)
        c.block(block_offset + i).add(A(i, 0), b);
  }

  BlockVectorType
  operator*(const FullMatrix<Number> &A, VectorType const &b)
  {
    const unsigned int n_blocks = A.m();
    BlockVectorType    c(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      c.block(i) = b;
    c = 0.0;
    tensorproduct_add(c, A, b);
    return c;
  }


  void
  tensorproduct_add(BlockVectorType          &c,
                    FullMatrix<Number> const &A,
                    BlockVectorType const    &b,
                    int                       block_offset = 0)
  {
    const unsigned int n_blocks = A.m();
    for (unsigned int i = 0; i < n_blocks; ++i)
      for (unsigned int j = 0; j < n_blocks; ++j)
        if (A(i, j) != 0.0)
          c.block(block_offset + i).add(A(i, j), b.block(block_offset + j));
  }

  BlockVectorType
  operator*(const FullMatrix<Number> &A, BlockVectorType const &b)
  {
    BlockVectorType c = b;
    c                 = 0.0;
    tensorproduct_add(c, A, b);
    return c;
  }

  template <typename Number, typename SystemMatrixType>
  class SystemMatrix
  {
  public:
    SystemMatrix(TimerOutput              &timer,
                 const SystemMatrixType   &K,
                 const SystemMatrixType   &M,
                 const FullMatrix<Number> &Alpha_,
                 const FullMatrix<Number> &Beta_)
      : timer(timer)
      , K(K)
      , M(M)
      , Alpha(Alpha_)
      , Beta(Beta_)
      , alpha_is_zero(Alpha.all_zero())
      , beta_is_zero(Beta.all_zero())
    {}
    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vmult");

      const unsigned int n_blocks = src.n_blocks();

      dst = 0.0;
      VectorType tmp;
      K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(j, i) != 0.0)
              dst.block(j).add(Alpha(j, i), tmp);
        }

      M.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          M.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(j, i) != 0.0)
              dst.block(j).add(Beta(j, i), tmp);
        }
    }

    void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "Tvmult");

      const unsigned int n_blocks = src.n_blocks();

      dst = 0.0;
      VectorType tmp;
      K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(i, j) != 0.0)
              dst.block(j).add(Alpha(i, j), tmp);
        }

      M.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          M.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(i, j) != 0.0)
              dst.block(j).add(Beta(i, j), tmp);
        }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    void
    vmult_add(BlockVectorType &dst, const VectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vmult");

      const unsigned int n_blocks = dst.n_blocks();

      VectorType tmp;
      if (!alpha_is_zero)
        {
          K.initialize_dof_vector(tmp);
          K.vmult(tmp, src);
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(j, 0) != 0.0)
              dst.block(j).add(Alpha(j, 0), tmp);
        }

      if (!beta_is_zero)
        {
          M.initialize_dof_vector(tmp);
          M.vmult(tmp, src);
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(j, 0) != 0.0)
              dst.block(j).add(Beta(j, 0), tmp);
        }
    }

    void
    vmult(BlockVectorType &dst, const VectorType &src) const
    {
      dst = 0.0;
      vmult_add(dst, src);
    }

    void
    initialize_dof_vector(VectorType &vec) const
    {
      K.initialize_dof_vector(vec);
    }

    void
    initialize_dof_vector(BlockVectorType &vec) const
    {
      vec.reinit(Alpha.m());
      for (unsigned int i = 0; i < vec.n_blocks(); ++i)
        this->initialize_dof_vector(vec.block(i));
    }

  private:
    TimerOutput              &timer;
    const SystemMatrixType   &K;
    const SystemMatrixType   &M;
    const FullMatrix<Number> &Alpha;
    const FullMatrix<Number> &Beta;

    // Only used for nx1: small optimization to avoid unnecessary vmult
    bool alpha_is_zero;
    bool beta_is_zero;
  };

  template <int dim, typename Number>
  class MatrixFreeOperator
  {
  public:
    MatrixFreeOperator(const Mapping<dim>              &mapping,
                       const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<Number> &constraints,
                       const Quadrature<dim>           &quadrature,
                       const double                     mass_matrix_scaling,
                       const double                     laplace_matrix_scaling)
      : mass_matrix_scaling(mass_matrix_scaling)
      , laplace_matrix_scaling(laplace_matrix_scaling)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;

      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);
    }

    void
    initialize_dof_vector(VectorType &vec) const
    {
      matrix_free.initialize_dof_vector(vec);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix_free.cell_loop(
        &MatrixFreeOperator::do_cell_integral_range, this, dst, src, true);
    }

    void
    compute_system_matrix(SparseMatrixType &sparse_matrix) const
    {
      MatrixFreeTools::compute_matrix(
        matrix_free,
        matrix_free.get_affine_constraints(),
        sparse_matrix,
        &MatrixFreeOperator::do_cell_integral_local,
        this);
    }

  private:
    using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

    void
    do_cell_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FECellIntegrator integrator(matrix_free);

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          integrator.reinit(cell);

          // gather
          integrator.read_dof_values(src);

          do_cell_integral_local(integrator);

          // scatter
          integrator.distribute_local_to_global(dst);
        }
    }

    void
    do_cell_integral_local(FECellIntegrator &integrator) const
    {
      // evaluate
      if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);
      else if (mass_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::values);
      else if (laplace_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::gradients);

      // quadrature
      for (unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          if (mass_matrix_scaling != 0.0)
            integrator.submit_value(mass_matrix_scaling *
                                      integrator.get_value(q),
                                    q);
          if (laplace_matrix_scaling != 0.0)
            integrator.submit_gradient(laplace_matrix_scaling *
                                         integrator.get_gradient(q),
                                       q);
        }

      // integrate
      if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
      else if (mass_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::values);
      else if (laplace_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::gradients);
    }

    MatrixFree<dim, Number> matrix_free;

    double mass_matrix_scaling;
    double laplace_matrix_scaling;
  };
} // namespace dealii
