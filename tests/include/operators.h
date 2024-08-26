// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/base/subscriptor.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "compute_block_matrix.h"
#include "types.h"
namespace dealii
{
  template <typename Number>
  void
  tensorproduct_add(BlockVectorT<Number>     &c,
                    FullMatrix<Number> const &A,
                    VectorT<Number> const    &b,
                    int                       block_offset = 0)
  {
    const unsigned int n_blocks = A.m();
    AssertDimension(A.n(), 1);
    for (unsigned int i = 0; i < n_blocks; ++i)
      if (A(i, 0) != 0.0)
        c.block(block_offset + i).add(A(i, 0), b);
  }

  template <typename Number>
  BlockVectorT<Number>
  operator*(const FullMatrix<Number> &A, VectorT<Number> const &b)
  {
    const unsigned int   n_blocks = A.m();
    BlockVectorT<Number> c(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      c.block(i).reinit(b);
    tensorproduct_add(c, A, b);
    return c;
  }

  template <typename Number>
  void
  tensorproduct_add(BlockVectorT<Number>       &c,
                    FullMatrix<Number> const   &A,
                    BlockVectorT<Number> const &b,
                    int                         block_offset = 0)
  {
    const unsigned int n_blocks = A.n();
    const unsigned int m_blocks = A.m();
    for (unsigned int i = 0; i < m_blocks; ++i)
      for (unsigned int j = 0; j < n_blocks; ++j)
        if (A(i, j) != 0.0)
          c.block(block_offset + i).add(A(i, j), b.block(block_offset + j));
  }

  template <typename Number>
  void
  tensorproduct_add(MutableBlockVectorSliceT<Number> &c,
                    FullMatrix<Number> const         &A,
                    BlockVectorSliceT<Number> const  &b,
                    int                               block_offset = 0)
  {
    const unsigned int n_blocks = A.n();
    const unsigned int m_blocks = A.m();
    for (unsigned int i = 0; i < m_blocks; ++i)
      for (unsigned int j = 0; j < n_blocks; ++j)
        if (A(i, j) != 0.0)
          c[block_offset + i].get().add(A(i, j), b[block_offset + j].get());
  }

  template <typename Number>
  BlockVectorT<Number>
  operator*(const FullMatrix<Number> &A, BlockVectorT<Number> const &b)
  {
    BlockVectorT<Number> c(A.m());
    for (unsigned int i = 0; i < A.m(); ++i)
      c.block(i).reinit(b.block(i));
    tensorproduct_add(c, A, b);
    return c;
  }

  template <typename Number,
            typename SystemMatrixTypeK,
            typename SystemMatrixTypeM = SystemMatrixTypeK>
  class SystemMatrixBase : public Subscriptor
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrixBase(TimerOutput              &timer,
                     const SystemMatrixTypeK  &K,
                     const SystemMatrixTypeM  &M,
                     const FullMatrix<Number> &Alpha_,
                     const FullMatrix<Number> &Beta_)
      : timer(timer)
      , K(K)
      , M(M)
      , Alpha(Alpha_)
      , Beta(Beta_)
      , alpha_is_zero(Alpha.all_zero())
      , beta_is_zero(Beta.all_zero())
    {
      AssertDimension(Alpha.m(), Beta.m());
      AssertDimension(Alpha.n(), Beta.n());
    }
    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType &dst, BlockVectorType const &src) const = 0;

    virtual void
    vmult_slice(BlockVectorType &dst, BlockVectorType const &src) const
    {
      dst = 0.0;
      vmult_slice_add(dst, src);
    }

    virtual types::global_dof_index
    m() const
    {
      return Alpha.m() * std::max(K.m(), M.m());
    }

    virtual types::global_dof_index
    n() const
    {
      AssertDimension(m(), Alpha.n() * std::max(K.m(), M.m()));
      return m();
    }

    virtual Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const = 0;
    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const = 0;

  protected:
    TimerOutput              &timer;
    const SystemMatrixTypeK  &K;
    const SystemMatrixTypeM  &M;
    const FullMatrix<Number> &Alpha;
    const FullMatrix<Number> &Beta;

    // Only used for nx1: small optimization to avoid unnecessary vmult
    bool alpha_is_zero;
    bool beta_is_zero;
  };


  template <typename Number, typename SystemMatrixType>
  class SystemMatrix final : public SystemMatrixBase<Number, SystemMatrixType>
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrix(TimerOutput              &timer,
                 const SystemMatrixType   &K,
                 const SystemMatrixType   &M,
                 const FullMatrix<Number> &Alpha_,
                 const FullMatrix<Number> &Beta_)
      : SystemMatrixBase<Number, SystemMatrixType>(timer, K, M, Alpha_, Beta_)
    {}

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");

      const unsigned int n_blocks = src.n_blocks();
      AssertDimension(this->Alpha.m(), n_blocks);
      dst = 0.0;
      VectorType tmp;
      this->K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          this->K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(j, i) != 0.0)
              dst.block(j).add(this->Alpha(j, i), tmp);

          this->M.vmult(tmp, src.block(i));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(j, i) != 0.0)
              dst.block(j).add(this->Beta(j, i), tmp);
        }
    }

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "Tvmult");

      const unsigned int n_blocks = src.n_blocks();

      dst = 0.0;
      VectorType tmp;
      this->K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          this->K.vmult(tmp, src.block(i));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(i, j) != 0.0)
              dst.block(j).add(this->Alpha(i, j), tmp);

          this->M.vmult(tmp, src.block(i));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(i, j) != 0.0)
              dst.block(j).add(this->Beta(i, j), tmp);
        }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType       &dst,
                    BlockVectorType const &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");

      const unsigned int n_blocks = dst.n_blocks();

      VectorType tmp;
      this->K.initialize_dof_vector(tmp);
      if (!this->alpha_is_zero)
        {
          this->K.vmult(tmp, src.block(0));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(j, 0) != 0.0)
              dst.block(j).add(this->Alpha(j, 0), tmp);
        }

      if (!this->beta_is_zero)
        {
          this->M.vmult(tmp, src.block(0));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(j, 0) != 0.0)
              dst.block(j).add(this->Beta(j, 0), tmp);
        }
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const override
    {
      BlockVectorType vec(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          vec.block(i) = this->K.get_matrix_diagonal()->get_vector();
          vec.block(i).sadd(this->Alpha(i, i),
                            this->Beta(i, i),
                            this->M.get_matrix_diagonal()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const override
    {
      BlockVectorType vec(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          vec.block(i) = this->K.get_matrix_diagonal_inverse()->get_vector();
          vec.block(i).sadd(
            1. / this->Alpha(i, i),
            1. / this->Beta(i, i),
            this->M.get_matrix_diagonal_inverse()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    virtual types::global_dof_index
    m() const override
    {
      return this->Alpha.m() * this->M.m();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int = 1) const
    {
      this->K.initialize_dof_vector(vec);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(this->Alpha.m());
      for (unsigned int i = 0; i < vec.n_blocks(); ++i)
        this->initialize_dof_vector(vec.block(i));
    }
  };


  template <typename Number, typename StokesMatrixType, typename MassMatrixType>
  class SystemMatrixStokes final
    : public SystemMatrixBase<Number, StokesMatrixType, MassMatrixType>
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrixStokes(TimerOutput              &timer,
                       const StokesMatrixType   &K,
                       const MassMatrixType     &M,
                       const FullMatrix<Number> &Alpha_,
                       const FullMatrix<Number> &Beta_,
                       const BlockSlice         &blk_slice_)
      : SystemMatrixBase<Number, StokesMatrixType, MassMatrixType>(timer,
                                                                   K,
                                                                   M,
                                                                   Alpha_,
                                                                   Beta_)
      , blk_slice(blk_slice_)
    {}


    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      AssertDimension(this->Alpha.m(), src.n_blocks());
      dst = 0.0;
      BlockVectorType tmp(2);
      this->K.initialize_dof_vector(tmp);
      BlockVectorType tmp_src(tmp.n_blocks());
      auto const     &n_timesteps_at_once = blk_slice.n_timesteps_at_once();
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            auto slice =
              blk_slice.get_variable(const_cast<BlockVectorType &>(src),
                                     it,
                                     id);
            swap(tmp_src, slice);
            // Stokes
            this->K.vmult(tmp, tmp_src);
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                for (unsigned int jv = 0; jv < blk_slice.n_variables(); ++jv)
                  scatter(
                    dst, tmp, this->Alpha, blk_slice, jt, jv, jd, it, 0, id);

            // \partial_t v
            this->M.vmult(tmp.block(0), tmp_src.block(0));
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                scatter(dst, tmp, this->Beta, blk_slice, 0, jt, jd, it, id);

            swap(slice, tmp_src);
          }
    }

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      AssertDimension(this->Alpha.m(), src.n_blocks());
      dst = 0.0;
      BlockVectorType tmp(2);
      this->K.initialize_dof_vector(tmp);
      BlockVectorType tmp_src(tmp.n_blocks());

      auto const &n_timesteps_at_once = blk_slice.n_timesteps_at_once();
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            auto slice =
              blk_slice.get_variable(const_cast<BlockVectorType &>(src),
                                     it,
                                     id);
            swap(tmp_src, slice);
            //  Stokes
            this->K.vmult(tmp, tmp_src);
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                for (unsigned int v = 0; v < blk_slice.n_variables(); ++v)
                  scatter(dst, tmp, this->Alpha, blk_slice, v, it, id, jt, jd);

            // \partial_t v
            this->M.vmult(tmp.block(0), tmp_src.block(0));
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                scatter(dst, tmp, this->Beta, blk_slice, 0, it, id, jt, jd);

            swap(slice, tmp_src);
          }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType       &dst,
                    BlockVectorType const &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      AssertDimension(this->Alpha.n(), 1);
      AssertDimension(src.n_blocks(), blk_slice.n_variables());

      BlockVectorType tmp(2);
      this->K.initialize_dof_vector(tmp);
      BlockVectorType tmp_src(tmp.n_blocks());

      for (unsigned int it = 0; it < blk_slice.n_timesteps_at_once(); ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            // Stokes
            if (!this->alpha_is_zero)
              {
                this->K.vmult(tmp, src);
                for (unsigned int v = 0; v < blk_slice.n_variables(); ++v)
                  scatter(dst, tmp, this->Alpha, blk_slice, it, v, id, 0, 0, 0);
              }
            // \partial_t v
            if (!this->beta_is_zero)
              {
                this->M.vmult(tmp.block(0), src.block(0));
                scatter(dst, tmp, this->Beta, blk_slice, it, 0, id, 0, 0, 0);
              }
          }
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const override
    {
      Assert(false, ExcNotImplemented());
      return std::make_shared<DiagonalMatrix<BlockVectorType>>();
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const override
    {
      Assert(false, ExcNotImplemented());
      return std::make_shared<DiagonalMatrix<BlockVectorType>>();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int variable) const
    {
      this->K.initialize_dof_vector(vec, variable);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          auto const &[tsp, v, td] = blk_slice.decompose(i);
          initialize_dof_vector(vec.block(i), v);
        }
    }

  private:
    void
    scatter(BlockVectorType          &dst,
            const BlockVectorType    &tmp,
            const FullMatrix<Number> &matrix,
            const BlockSlice         &blk_slice,
            unsigned int              jt,
            unsigned int              jv,
            unsigned int              jd,
            unsigned int              it,
            unsigned int              iv,
            unsigned int              id) const
    {
      unsigned int j = blk_slice.index(jt, jv, jd);
      unsigned int i = blk_slice.index(it, iv, id);
      if (std::abs(matrix(j, i)) > 10 * std::numeric_limits<Number>::epsilon())
        dst.block(j).add(matrix(j, i), tmp.block(jv));
    }


    void
    scatter(BlockVectorType          &dst,
            const BlockVectorType    &tmp,
            const FullMatrix<Number> &matrix,
            const BlockSlice         &blk_slice,
            unsigned int              v,
            unsigned int              jt,
            unsigned int              jd,
            unsigned int              it,
            unsigned int              id) const
    {
      scatter(dst, tmp, matrix, blk_slice, jt, v, jd, it, v, id);
    }

    BlockSlice blk_slice;
  };


  template <int dim>
  class Coefficient final : public Function<dim>
  {
    double             c1, c2, c3;
    bool               distorted;
    Point<dim>         lower_left;
    Point<dim>         step_size;
    Table<dim, double> distortion;

    template <typename number>
    double
    get_coefficient(number px, number py) const
    {
      if (py >= 0.2)
        {
          if (px < 0.2)
            return c2;
          else
            return c3;
        }
      return c1;
    }


  public:
    Coefficient(Parameters<dim> const &params,
                double                 c1_ = 1.0,
                double                 c2_ = 9.0,
                double                 c3_ = 16.0)
      : c1(c1_)
      , c2(c2_)
      , c3(c3_)
      , distorted(params.distort_coeff != 0.0)
      , lower_left(params.hyperrect_lower_left)
    {
      if (distorted)
        {
          auto const &subdivisions = params.subdivisions;
          if constexpr (dim == 2)
            distortion = Table<2, double>(subdivisions[0], subdivisions[1]);
          else
            distortion = Table<3, double>(subdivisions[0],
                                          subdivisions[1],
                                          subdivisions[2]);
          std::vector<double>    tmp(distortion.n_elements());
          boost::random::mt19937 rng(boost::random::mt19937::default_seed);
          boost::random::uniform_real_distribution<> uniform_distribution(
            1 - params.distort_coeff, 1 + params.distort_coeff);
          std::generate(tmp.begin(), tmp.end(), [&]() {
            return uniform_distribution(rng);
          });
          distortion.fill(tmp.begin());
          auto const extent =
            params.hyperrect_upper_right - params.hyperrect_lower_left;
          for (int i = 0; i < dim; ++i)
            step_size[i] = extent[i] / subdivisions[i];
        }
    }

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return get_coefficient(p[0], p[1]);
    }

    template <typename number>
    number
    value(const Point<dim, number> &p) const
    {
      number value;
      auto   v = value.begin();
      if constexpr (dim == 2)
        for (auto px = p[0].begin(), py = p[1].begin(); px != p[0].end();
             ++px, ++py, ++v)
          {
            *v = get_coefficient(*px, *py);
            if (distorted)
              *v *= distortion(
                static_cast<unsigned>((*px - lower_left[0]) / step_size[0]),
                static_cast<unsigned>((*py - lower_left[1]) / step_size[1]));
          }
      else
        for (auto px = p[0].begin(), py = p[1].begin(), pz = p[2].begin();
             px != p[0].end();
             ++px, ++py, ++pz, ++v)
          {
            *v = get_coefficient(*px, *py);
            if (distorted)
              *v *= distortion(
                static_cast<unsigned>((*px - lower_left[0]) / step_size[0]),
                static_cast<unsigned>((*py - lower_left[1]) / step_size[1]),
                static_cast<unsigned>((*pz - lower_left[2]) / step_size[2]));
          }
      return value;
    }
  };

  template <int dim, int n_components, typename Number>
  class MatrixFreeOperator
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;
    MatrixFreeOperator(const Mapping<dim>              &mapping,
                       const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<Number> &constraints,
                       const Quadrature<dim>           &quadrature,
                       const double                     mass_matrix_scaling,
                       const double                     laplace_matrix_scaling)
      : mass_matrix_scaling(mass_matrix_scaling)
      , laplace_matrix_scaling(laplace_matrix_scaling)
      , has_mass_coefficient(false)
      , has_laplace_coefficient(false)
    {
      mass_matrix_coefficient.clear();
      laplace_matrix_coefficient.clear();
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags =
        update_values | update_gradients | update_quadrature_points;

      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);

      compute_diagonal();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec) const
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
    compute_system_matrix(
      SparseMatrixType                        &sparse_matrix,
      dealii::AffineConstraints<Number> const *constraints = nullptr) const
    {
      if (constraints == nullptr)
        constraints = &matrix_free.get_affine_constraints();
      compute_matrix(matrix_free,
                     *constraints,
                     sparse_matrix,
                     &MatrixFreeOperator::do_cell_integral_local,
                     this);
      // MatrixFreeTools::compute_matrix(
      //   matrix_free,
      //   matrix_free.get_affine_constraints(),
      //   sparse_matrix,
      //   &MatrixFreeOperator::do_cell_integral_local,
      //   this);
    }

    std::shared_ptr<DiagonalMatrix<VectorType>> const &
    get_matrix_diagonal() const
    {
      return diagonal;
    }

    std::shared_ptr<DiagonalMatrix<VectorType>> const &
    get_matrix_diagonal_inverse() const
    {
      return diagonal_inverse;
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler().n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    void
    evaluate_coefficient(const Coefficient<dim> &coefficient_fun)
    {
      FECellIntegrator   integrator(matrix_free);
      const unsigned int n_cells = matrix_free.n_cell_batches();
      if (mass_matrix_scaling != 0.0)
        mass_matrix_coefficient.reinit(n_cells, integrator.n_q_points);
      if (laplace_matrix_scaling != 0.0)
        laplace_matrix_coefficient.reinit(n_cells, integrator.n_q_points);

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          integrator.reinit(cell);
          for (const unsigned int q : integrator.quadrature_point_indices())
            {
              if (mass_matrix_scaling != 0.0)
                mass_matrix_coefficient(cell, q) =
                  coefficient_fun.value(integrator.quadrature_point(q));
              if (laplace_matrix_scaling != 0.0)
                laplace_matrix_coefficient(cell, q) =
                  coefficient_fun.value(integrator.quadrature_point(q));
            }
        }
      if (!mass_matrix_coefficient.empty())
        has_mass_coefficient = true;
      if (!laplace_matrix_coefficient.empty())
        has_laplace_coefficient = true;
    }

  private:
    using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;

    void
    compute_diagonal()
    {
      diagonal         = std::make_shared<DiagonalMatrix<VectorType>>();
      diagonal_inverse = std::make_shared<DiagonalMatrix<VectorType>>();
      VectorType &diagonal_inv_vector = diagonal_inverse->get_vector();
      VectorType &diagonal_vector     = diagonal->get_vector();
      initialize_dof_vector(diagonal_inv_vector);
      initialize_dof_vector(diagonal_vector);
      MatrixFreeTools::compute_diagonal(
        matrix_free,
        diagonal_vector,
        &MatrixFreeOperator::do_cell_integral_local,
        this);
      diagonal_inv_vector = diagonal_vector;
      auto constexpr tol  = std::sqrt(std::numeric_limits<Number>::epsilon());
      for (auto &i : diagonal_inv_vector)
        i = std::abs(i) > tol ? 1. / i : 1.;
    }

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
      unsigned int const cell = integrator.get_current_cell_index();
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
            integrator.submit_value((has_mass_coefficient ?
                                       mass_matrix_coefficient(cell, q) :
                                       mass_matrix_scaling) *
                                      integrator.get_value(q),
                                    q);
          if (laplace_matrix_scaling != 0.0)
            integrator.submit_gradient((has_laplace_coefficient ?
                                          laplace_matrix_coefficient(cell, q) :
                                          laplace_matrix_scaling) *
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

    std::shared_ptr<DiagonalMatrix<VectorType>> diagonal;
    std::shared_ptr<DiagonalMatrix<VectorType>> diagonal_inverse;

    MatrixFree<dim, Number> matrix_free;

    Number mass_matrix_scaling;
    Number laplace_matrix_scaling;

    bool                              has_mass_coefficient    = false;
    bool                              has_laplace_coefficient = false;
    Table<2, VectorizedArray<Number>> mass_matrix_coefficient;
    Table<2, VectorizedArray<Number>> laplace_matrix_coefficient;
  };
  template <int dim, typename Number>
  using MatrixFreeOperatorScalar = MatrixFreeOperator<dim, 1, Number>;
  template <int dim, typename Number>
  using MatrixFreeOperatorVector = MatrixFreeOperator<dim, dim, Number>;

  template <int dim, typename Number>
  class StokesMatrixFreeOperator
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    StokesMatrixFreeOperator(
      const Mapping<dim>                                   &mapping,
      const std::vector<const DoFHandler<dim> *>           &dof_handlers,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const std::vector<Quadrature<dim>>                   &quadrature,
      const double                                          viscosity_)
      : viscosity(viscosity_)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      // additional_data.mapping_update_flags =
      //   update_values | update_gradients | update_quadrature_points;
      additional_data.mapping_update_flags = update_values | update_gradients;
      matrix_free.reinit(
        mapping, dof_handlers, constraints, quadrature, additional_data);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(2);
      initialize_dof_vector(vec.block(0), 0);
      initialize_dof_vector(vec.block(1), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(
      const std::vector<std::reference_wrapper<VectorT<Number2>>> &vec) const
    {
      initialize_dof_vector(vec[0].get(), 0);
      initialize_dof_vector(vec[1].get(), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int variable) const
    {
      matrix_free.initialize_dof_vector(vec, variable);
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      matrix_free.cell_loop(&StokesMatrixFreeOperator::do_cell_integral_range,
                            this,
                            dst,
                            src,
                            true);
    }

    void
    compute_system_matrix(
      BlockSparseMatrixType                                 &sparse_matrix,
      std::vector<const dealii::AffineConstraints<Number> *> constraints =
        std::vector<const dealii::AffineConstraints<Number> *>()) const
    {
      if (constraints.empty())
        constraints = std::vector<const dealii::AffineConstraints<Number> *>{
          &matrix_free.get_affine_constraints(0),
          &matrix_free.get_affine_constraints(1)};
      compute_matrix(matrix_free,
                     constraints,
                     sparse_matrix,
                     &StokesMatrixFreeOperator::do_cell_integral_local,
                     this);
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler(0).n_dofs() +
             matrix_free.get_dof_handler(1).n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

  private:
    using FECellIntegratorP = FEEvaluation<dim, -1, 0, 1, Number>;
    using FECellIntegratorU = FEEvaluation<dim, -1, 0, dim, Number>;

    void
    do_cell_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      BlockVectorType                             &dst,
      const BlockVectorType                       &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FECellIntegratorU velocity(matrix_free, 0);
      FECellIntegratorP pressure(matrix_free, 1);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          velocity.reinit(cell);
          velocity.read_dof_values(src.block(0));
          pressure.reinit(cell);
          pressure.read_dof_values(src.block(1));

          do_cell_integral_local(velocity, pressure);

          velocity.distribute_local_to_global(dst.block(0));
          pressure.distribute_local_to_global(dst.block(1));
        }
    }


    void
    do_cell_integral_local(FECellIntegratorU &velocity,
                           FECellIntegratorP &pressure) const
    {
      velocity.evaluate(EvaluationFlags::gradients);
      pressure.evaluate(EvaluationFlags::values);

      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          auto grad_u = velocity.get_gradient(q);
          auto div_u  = velocity.get_divergence(q);
          auto p      = pressure.get_value(q);
          pressure.submit_value(div_u, q);
          grad_u *= viscosity;
          for (unsigned int i = 0; i < dim; ++i)
            grad_u[i][i] -= p;
          velocity.submit_gradient(grad_u, q);
        }

      velocity.integrate(EvaluationFlags::gradients);
      pressure.integrate(EvaluationFlags::values);
    }

    MatrixFree<dim, Number> matrix_free;

    Number viscosity = 1.0;
  };
} // namespace dealii
