#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>

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
  using VectorType = BlockVector<Number>;

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    (void)dst;
    (void)src;
  }

private:
};



template <int dim>
void
test()
{
  using Number     = double;
  using VectorType = BlockVector<Number>;

  SolverControl           solver_control;
  SolverGMRES<VectorType> solver(solver_control);

  VectorType x, rhs;

  SparseMatrix<Number> K, M;  // TODO: fill
  FullMatrix<Number>   A_inv; //

  SystemMatrix<Number>   matrix(K, M, A_inv);
  Preconditioner<Number> preconditioner;

  solver.solve(matrix, x, rhs, preconditioner);
}



int
main()
{
  return 0;

  const int dim = 2;

  test<dim>();
}