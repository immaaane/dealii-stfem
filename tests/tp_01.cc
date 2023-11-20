#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>

using namespace dealii;

template <typename Number>
class SystemMatrix
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

  SystemMatrix<Number>   matrix;
  Preconditioner<Number> preconditioner;

  solver.solve(matrix, x, rhs, preconditioner);
}

int
main()
{
  const int dim = 2;

  test<dim>();
}