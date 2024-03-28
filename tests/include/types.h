#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

using Number = double;
template <typename Number>
using VectorT = dealii::LinearAlgebra::distributed::Vector<Number>;
template <typename Number>
using BlockVectorT = dealii::LinearAlgebra::distributed::BlockVector<Number>;
using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
using SparsityPatternType = dealii::TrilinosWrappers::SparsityPattern;

enum class ProblemType : unsigned int
{
  heat = 1,
  wave = 2,
};
static std::unordered_map<std::string, ProblemType> const str_to_problem_type =
  {{"heat", ProblemType::heat}, {"wave", ProblemType::wave}};
