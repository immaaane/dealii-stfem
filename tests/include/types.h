#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

using Number          = double;
using VectorType      = dealii::LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<double>;
using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
using SparsityPatternType = dealii::TrilinosWrappers::SparsityPattern;
