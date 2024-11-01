// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once
#include <deal.II/base/types.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <functional>
#include <vector>

using Number = double;
template <typename Number>
using VectorT = dealii::LinearAlgebra::distributed::Vector<Number>;
template <typename Number>
using BlockVectorT = dealii::LinearAlgebra::distributed::BlockVector<Number>;
using SparseMatrixType         = dealii::TrilinosWrappers::SparseMatrix;
using SparsityPatternType      = dealii::TrilinosWrappers::SparsityPattern;
using BlockSparseMatrixType    = dealii::TrilinosWrappers::BlockSparseMatrix;
using BlockSparsityPatternType = dealii::TrilinosWrappers::BlockSparsityPattern;

template <typename Number>
using BlockVectorSliceT =
  std::vector<std::reference_wrapper<const VectorT<Number>>>;
template <typename Number>
using MutableBlockVectorSliceT =
  std::vector<std::reference_wrapper<VectorT<Number>>>;

constexpr dealii::types::manifold_id cylindrical_manifold_id = 0;
constexpr dealii::types::manifold_id tfi_manifold_id         = 1;

enum class ProblemType : unsigned int
{
  heat   = 1,
  wave   = 2,
  stokes = 3,
};

enum class CoarseningType : int
{
  space_or_time  = 0,
  space_and_time = 1,
};

static std::unordered_map<std::string, ProblemType> const str_to_problem_type =
  {{"heat", ProblemType::heat},
   {"wave", ProblemType::wave},
   {"stokes", ProblemType::stokes}};

static std::unordered_map<std::string,
                          dealii::MGTransferGlobalCoarseningTools::
                            PolynomialCoarseningSequenceType> const
  str_to_polynomial_coarsening_type = {
    {"bisect",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       bisect},
    {"decrease_by_one",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       decrease_by_one},
    {"go_to_one",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       go_to_one}};

static std::unordered_map<std::string, CoarseningType> const
  str_to_coarsening_type = {{"space_and_time", CoarseningType::space_and_time},
                            {"space_or_time", CoarseningType::space_or_time}};
