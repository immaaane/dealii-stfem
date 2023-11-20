#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>

using namespace dealii;

template <typename Number>
class SystemMatrix
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  SystemMatrix(const SparseMatrix<Number> &K,
               const SparseMatrix<Number> &M,
               const FullMatrix<Number>   &A_inv)
    : K(K)
    , M(M)
    , A_inv(A_inv)
  {}

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    const unsigned int n_blocks = src.n_blocks();

    VectorType tmp;
    tmp.reinit(src.block(0));
    for (unsigned int i = 0; i < n_blocks; ++i)
      K.vmult(dst.block(i), src.block(i));

    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        M.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          dst.block(j).add(A_inv(j, i), tmp);
      }
  }

private:
  const SparseMatrix<Number> &K;
  const SparseMatrix<Number> &M;
  const FullMatrix<Number>   &A_inv;
};



template <typename Number>
class Preconditioner
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  template <int dim>
  Preconditioner(const SparseMatrix<Number> &K,
                 const SparseMatrix<Number> &M,
                 const FullMatrix<Number>   &A_inv,
                 const DoFHandler<dim>      &dof_handler)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> my_indices(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(my_indices);

        indices.emplace_back(my_indices);
      }

    std::vector<FullMatrix<Number>> K_blocks, M_blocks;

    SparseMatrixTools::restrict_to_full_matrices(K,
                                                 K.get_sparsity_pattern(),
                                                 indices,
                                                 K_blocks);
    SparseMatrixTools::restrict_to_full_matrices(M,
                                                 M.get_sparsity_pattern(),
                                                 indices,
                                                 M_blocks);

    blocks.resize(K_blocks.size());

    for (unsigned int i = 0; i < blocks.size(); ++i)
      {
        const auto &K = K_blocks[i];
        const auto &M = M_blocks[i];
        auto       &B = blocks[i];

        B = FullMatrix<Number>(K.m() * A_inv.m(), K.n() * A_inv.n());

        for (unsigned int i = 0; i < A_inv.m(); ++i)
          for (unsigned int j = 0; j < A_inv.n(); ++j)
            for (unsigned int k = 0; k < K.m(); ++k)
              for (unsigned int l = 0; l < K.n(); ++l)
                B(k + i * K.m(), l + j * K.n()) =
                  A_inv(i, j) * M(k, l) + ((i == j) ? K(k, l) : 0.0);

        B.gauss_jordan();
      }
  }

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    dst = 0.0;

    Vector<Number> dst_local;
    Vector<Number> src_local;

    const unsigned int n_blocks = src.n_blocks();

    for (unsigned int i = 0; i < blocks.size(); ++i)
      {
        // gather
        src_local.reinit(blocks[i].m());
        dst_local.reinit(blocks[i].m());

        for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
          for (unsigned int j = 0; j < indices[i].size(); ++j, ++c)
            src_local[c] = src.block(b)[indices[i][j]];

        // patch solver
        blocks[i].vmult(dst_local, src_local);

        // scatter
        for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
          for (unsigned int j = 0; j < indices[i].size(); ++j, ++c)
            dst.block(b)[indices[i][j]] += dst_local[c];
      }
  }

private:
  std::vector<std::vector<types::global_dof_index>> indices;
  std::vector<FullMatrix<Number>>                   blocks;
};



template <int dim>
void
test()
{
  using Number     = double;
  using VectorType = BlockVector<Number>;

  DoFHandler<dim> dof_handler;

  SolverControl           solver_control;
  SolverGMRES<VectorType> solver(solver_control);

  VectorType x, rhs;

  SparseMatrix<Number> K, M;  // TODO: fill
  FullMatrix<Number>   A_inv; //

  SystemMatrix<Number>   matrix(K, M, A_inv);
  Preconditioner<Number> preconditioner(K, M, A_inv, dof_handler);

  solver.solve(matrix, x, rhs, preconditioner);
}



int
main()
{
  return 0;

  const int dim = 2;

  test<dim>();
}