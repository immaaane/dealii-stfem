#pragma once
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

namespace dealii
{
  namespace SparseMatrixTools
  {
    template <typename SparseMatrixType,
              typename SparsityPatternType,
              typename Number>
    void
    restrict_to_full_matrices_(
      const SparseMatrixType                                  &system_matrix,
      const SparsityPatternType                               &sparsity_pattern,
      const std::vector<std::vector<types::global_dof_index>> &row_indices,
      const std::vector<std::vector<types::global_dof_index>> &col_indices,
      std::vector<FullMatrix<Number>>                         &blocks,
      const VectorT<Number> &scaling = VectorT<Number>())
    {
      // 0) determine which rows are locally owned and which ones are remote
      const auto local_size = internal::get_local_size(system_matrix);
      const auto prefix_sum = Utilities::MPI::partial_and_total_sum(
        local_size, internal::get_mpi_communicator(system_matrix));
      IndexSet locally_owned_dofs(std::get<1>(prefix_sum));
      locally_owned_dofs.add_range(std::get<0>(prefix_sum),
                                   std::get<0>(prefix_sum) + local_size);

      std::vector<dealii::types::global_dof_index> ghost_indices_vector;

      for (const auto &i : row_indices)
        ghost_indices_vector.insert(ghost_indices_vector.end(),
                                    i.begin(),
                                    i.end());

      std::sort(ghost_indices_vector.begin(), ghost_indices_vector.end());

      IndexSet locally_active_dofs(std::get<1>(prefix_sum));
      locally_active_dofs.add_indices(ghost_indices_vector.begin(),
                                      ghost_indices_vector.end());

      locally_active_dofs.subtract_set(locally_owned_dofs);

      // 1) collect remote rows of sparse matrix
      const auto locally_relevant_matrix_entries =
        internal::extract_remote_rows<Number>(system_matrix,
                                              sparsity_pattern,
                                              locally_active_dofs,
                                              internal::get_mpi_communicator(
                                                system_matrix));


      // 2) loop over all cells and "revert" assembly
      blocks.clear();
      blocks.resize(row_indices.size());

      for (unsigned int c = 0; c < row_indices.size(); ++c)
        {
          if (row_indices[c].empty() || col_indices[c].empty())
            continue;

          const auto &local_dof_row_indices = row_indices[c];
          const auto &local_dof_col_indices = col_indices[c];
          auto       &cell_matrix           = blocks[c];

          // allocate memory
          const unsigned int n_local_rows = row_indices[c].size();
          const unsigned int n_local_cols = col_indices[c].size();

          cell_matrix = FullMatrix<Number>(n_local_rows, n_local_cols);

          // loop over all entries of the restricted element matrix and
          // do different things if rows are locally owned or not and
          // if column entries of that row exist or not
          for (unsigned int i = 0; i < n_local_rows; ++i)
            for (unsigned int j = 0; j < n_local_cols; ++j)
              {
                if (locally_owned_dofs.is_element(
                      local_dof_row_indices[i])) // row is local
                  {
                    cell_matrix(i, j) =
                      sparsity_pattern.exists(local_dof_row_indices[i],
                                              local_dof_col_indices[j]) ?
                        system_matrix(local_dof_row_indices[i],
                                      local_dof_col_indices[j]) :
                        0;
                  }
                else // row is ghost
                  {
                    Assert(locally_active_dofs.is_element(
                             local_dof_row_indices[i]),
                           ExcInternalError());

                    const auto &row_entries = locally_relevant_matrix_entries
                      [locally_active_dofs.index_within_set(
                        local_dof_row_indices[i])];

                    const auto ptr = std::lower_bound(
                      row_entries.begin(),
                      row_entries.end(),
                      std::pair<types::global_dof_index, Number>{
                        local_dof_col_indices[j], /*dummy*/ 0.0},
                      [](const auto a, const auto b) {
                        return a.first < b.first;
                      });

                    if (ptr != row_entries.end() &&
                        local_dof_col_indices[j] == ptr->first)
                      cell_matrix(i, j) = ptr->second;
                    else
                      cell_matrix(i, j) = 0.0;
                  }
                if (scaling.size() != 0)
                  cell_matrix(i, j) *= scaling[local_dof_row_indices[i]];
              }
        }
    }
    template <typename BlockSparseMatrixType,
              typename BlockSparsityPatternType,
              typename Number>
    Table<2, std::vector<FullMatrix<Number>>>
    restrict_to_full_block_matrices_(
      const BlockSparseMatrixType    &system_matrix,
      const BlockSparsityPatternType &sparsity_pattern,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
        &row_indices,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
                                 &col_indices,
      const BlockVectorT<Number> &scaling = BlockVectorT<Number>(),
      const Table<2, bool>       &mask    = Table<2, bool>())
    {
      Table<2, std::vector<FullMatrix<Number>>> blocks_(
        sparsity_pattern.n_block_rows(), sparsity_pattern.n_block_cols());
      for (unsigned int i = 0; i < sparsity_pattern.n_block_rows(); ++i)
        for (unsigned int j = 0; j < sparsity_pattern.n_block_cols(); ++j)
          {
            if (mask.empty() || mask(i, j))
              restrict_to_full_matrices_(system_matrix.block(i, j),
                                         sparsity_pattern.block(i, j),
                                         row_indices[i],
                                         col_indices[j],
                                         blocks_(i, j),
                                         scaling.block(i));
          }
      return blocks_;
    }

    template <typename BlockSparseMatrixType,
              typename BlockSparsityPatternType,
              typename Number>
    void
    restrict_to_full_matrices_(
      const BlockSparseMatrixType    &system_matrix,
      const BlockSparsityPatternType &sparsity_pattern,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
        &row_indices,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
                                      &col_indices,
      std::vector<FullMatrix<Number>> &blocks,
      const BlockVectorT<Number>      &scaling = BlockVectorT<Number>(),
      const Table<2, bool>            &mask    = Table<2, bool>())
    {
      Table<2, std::vector<FullMatrix<Number>>> blocks_ =
        restrict_to_full_block_matrices_(system_matrix,
                                         sparsity_pattern,
                                         row_indices,
                                         col_indices,
                                         scaling,
                                         mask);
      blocks.resize(blocks_(0, 0).size());
      for (unsigned int ii = 0; ii < blocks.size(); ++ii)
        {
          auto        &block  = blocks[ii];
          unsigned int n_rows = 0, n_cols = 0;
          for (unsigned int i = 0; i < blocks_.size(0); ++i)
            n_rows += row_indices[i][ii].size();
          for (unsigned int i = 0; i < blocks_.size(1); ++i)
            n_cols += col_indices[i][ii].size();

          block = FullMatrix<Number>(n_rows, n_cols);
          for (unsigned int i = 0; i < blocks_.size(0); ++i)
            for (unsigned int j = 0; j < blocks_.size(1); ++j)
              if (mask.empty() || mask(i, j))
                {
                  auto const &block_block = blocks_(i, j)[ii];
                  block.fill(block_block,
                             i == 0 ? 0 : blocks_(i - 1, j)[ii].m(),
                             j == 0 ? 0 : blocks_(i, j - 1)[ii].n(),
                             0,
                             0);
                }
        }
    }
  } // namespace SparseMatrixTools

  template <int dim, typename Number>
  void
  make_block_sparsity_pattern_block(
    const DoFHandler<dim>             &dof_row,
    const DoFHandler<dim>             &dof_col,
    TrilinosWrappers::SparsityPattern &sparsity,
    const AffineConstraints<Number>   &constraints_row,
    const AffineConstraints<Number>   &constraints_col,
    const bool                         keep_constrained_dofs,
    const types::subdomain_id          subdomain_id)
  {
    const types::global_dof_index n_dofs_row = dof_row.n_dofs();
    const types::global_dof_index n_dofs_col = dof_col.n_dofs();
    (void)n_dofs_row;
    (void)n_dofs_col;
    Assert(sparsity.n_rows() == n_dofs_row,
           ExcDimensionMismatch(sparsity.n_rows(), n_dofs_row));
    Assert(sparsity.n_cols() == n_dofs_col,
           ExcDimensionMismatch(sparsity.n_cols(), n_dofs_col));
    // See DoFTools::make_sparsity_pattern
    if (const auto *triangulation =
          dynamic_cast<const parallel::DistributedTriangulationBase<dim> *>(
            &dof_row.get_triangulation()))
      Assert((subdomain_id == numbers::invalid_subdomain_id) ||
               (subdomain_id == triangulation->locally_owned_subdomain()),
             ExcMessage(
               "For distributed Triangulation objects and associated "
               "DoFHandler objects, asking for any subdomain other than the "
               "locally owned one does not make sense."));

    std::vector<types::global_dof_index> dofs_row_on_this_cell;
    dofs_row_on_this_cell.reserve(
      dof_row.get_fe_collection().max_dofs_per_cell());
    std::vector<types::global_dof_index> dofs_col_on_this_cell;
    dofs_col_on_this_cell.reserve(
      dof_col.get_fe_collection().max_dofs_per_cell());

    // See DoFTools::make_sparsity_pattern
    for (const auto &cell_row : dof_row.active_cell_iterators())
      if (((subdomain_id == numbers::invalid_subdomain_id) ||
           (subdomain_id == cell_row->subdomain_id())) &&
          cell_row->is_locally_owned())
        {
          typename DoFHandler<dim>::active_cell_iterator cell_col =
            cell_row->as_dof_handler_iterator(dof_col);
          const unsigned int dofs_per_cell_row =
            cell_row->get_fe().n_dofs_per_cell();
          dofs_row_on_this_cell.resize(dofs_per_cell_row);
          cell_row->get_dof_indices(dofs_row_on_this_cell);
          const unsigned int dofs_per_cell_col =
            cell_col->get_fe().n_dofs_per_cell();
          dofs_col_on_this_cell.resize(dofs_per_cell_col);
          cell_col->get_dof_indices(dofs_col_on_this_cell);
          // See DoFTools::make_sparsity_pattern
          constraints_row.add_entries_local_to_global(dofs_row_on_this_cell,
                                                      constraints_col,
                                                      dofs_col_on_this_cell,
                                                      sparsity,
                                                      keep_constrained_dofs);
        }
  }

  template <std::size_t N, std::size_t... Is>
  struct even_sequence_helper : even_sequence_helper<N - 1, 2 * (N - 1), Is...>
  {};

  template <std::size_t... Is>
  struct even_sequence_helper<0, Is...>
  {
    using type = std::index_sequence<Is...>;
  };

  template <std::size_t N>
  using even_sequence = typename even_sequence_helper<(N + 1) / 2>::type;

  template <std::size_t N, std::size_t... Is>
  struct odd_sequence_helper
    : odd_sequence_helper<N - 1, 2 * (N - 1) + 1, Is...>
  {};

  template <std::size_t... Is>
  struct odd_sequence_helper<0, Is...>
  {
    using type = std::index_sequence<Is...>;
  };

  template <std::size_t N>
  using odd_sequence = typename odd_sequence_helper<N / 2>::type;

  std::vector<unsigned int>
  interleave(const std::vector<unsigned int> &a,
             const std::vector<unsigned int> &b)
  {
    std::vector<unsigned int> interleaved;
    interleaved.reserve(a.size() + b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      {
        interleaved.push_back(a[i]);
        interleaved.push_back(b[i]);
      }
    return interleaved;
  }

  template <std::size_t... Is>
  std::vector<unsigned int>
  make_index_vector(std::index_sequence<Is...>)
  {
    return {Is...};
  }
  template <typename T1, typename T2>
  auto
  forward_as_pair(T1 &&first, T2 &&second)
  {
    return std::pair<T1 &&, T2 &&>(std::forward<T1>(first),
                                   std::forward<T2>(second));
  }
  template <int dim,
            typename VectorizedArrayType,
            bool is_face,
            typename... CellEvalTypes,
            std::size_t... Is>
  auto
  generate_tuple_from_vector(
    std::vector<
      std::unique_ptr<FEEvaluationData<dim, VectorizedArrayType, is_face>>>
      &phi,
    std::index_sequence<Is...>)
  {
    static_assert(sizeof...(Is) == sizeof...(CellEvalTypes),
                  "Index sequence size must match number of CellEvalTypes");
    Assert(phi.size() >= sizeof...(Is),
           ExcLowerRange(phi.size(), sizeof...(Is)));
    return std::forward_as_tuple(static_cast<CellEvalTypes &>(*phi[Is])...);
  }

  template <int dim,
            typename VectorizedArrayType,
            bool is_face,
            typename... FaceEvalTypes,
            std::size_t... Is>
  auto
  generate_pair_tuple_from_vector(
    std::vector<
      std::unique_ptr<FEEvaluationData<dim, VectorizedArrayType, is_face>>>
      &phi,
    std::index_sequence<Is...>)
  {
    return std::forward_as_tuple(
      forward_as_pair(static_cast<FaceEvalTypes &>(*phi[Is * 2]),
                      static_cast<FaceEvalTypes &>(*phi[Is * 2 + 1]))...);
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename... FaceEvalTypes,
            std::size_t... Is>
  auto
  create_face_phi_vector(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const std::pair<unsigned int, unsigned int>        &range,
    const std::vector<unsigned int> &first_selected_component,
    const std::index_sequence<Is...>)
  {
    std::vector<
      std::unique_ptr<FEEvaluationData<dim, VectorizedArrayType, true>>>
      phi;
    phi.reserve(2 * sizeof...(FaceEvalTypes));
    (
      [&]() {
        if (!MatrixFreeTools::internal::is_fe_nothing<true>(
              matrix_free,
              range,
              Is,
              Is,
              first_selected_component[Is],
              FaceEvalTypes::static_fe_degree,
              FaceEvalTypes::static_n_q_points_1d,
              true) &&
            !MatrixFreeTools::internal::is_fe_nothing<true>(
              matrix_free,
              range,
              Is,
              Is,
              first_selected_component[Is],
              FaceEvalTypes::static_fe_degree,
              FaceEvalTypes::static_n_q_points_1d,
              false))
          {
            phi.emplace_back(std::make_unique<FaceEvalTypes>(
              matrix_free,
              range,
              true,
              Is,
              Is,
              first_selected_component[Is * 2]));
            phi.emplace_back(std::make_unique<FaceEvalTypes>(
              matrix_free,
              range,
              false,
              Is,
              Is,
              first_selected_component[Is * 2 + 1]));
          }
      }(),
      ...);
    return phi;
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename... FaceEvalTypes,
            std::size_t... Is>
  auto
  create_boundary_phi_vector(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const std::pair<unsigned int, unsigned int>        &range,
    const std::vector<unsigned int> &first_selected_component,
    const std::index_sequence<Is...>)
  {
    std::vector<
      std::unique_ptr<FEEvaluationData<dim, VectorizedArrayType, true>>>
      phi;
    phi.reserve(sizeof...(FaceEvalTypes));

    (
      [&]() {
        if (!MatrixFreeTools::internal::is_fe_nothing<true>(
              matrix_free,
              range,
              Is,
              Is,
              first_selected_component[Is],
              FaceEvalTypes::static_fe_degree,
              FaceEvalTypes::static_n_q_points_1d,
              true))
          phi.emplace_back(std::make_unique<FaceEvalTypes>(
            matrix_free, range, true, Is, Is, first_selected_component[Is]));
      }(),
      ...);
    return phi;
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename... CellEvalTypes,
            std::size_t... Is>
  auto
  create_cell_phi_vector(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const std::pair<unsigned int, unsigned int>        &range,
    const MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, VectorizedArrayType, false> &data,
    const std::index_sequence<Is...>)
  {
    static_assert(sizeof...(Is) == sizeof...(CellEvalTypes),
                  "Index sequence size must match number of CellEvalTypes");
    std::vector<
      std::unique_ptr<FEEvaluationData<dim, VectorizedArrayType, false>>>
      phi;
    phi.reserve(sizeof...(CellEvalTypes));

    (
      [&]() {
        if (!MatrixFreeTools::internal::is_fe_nothing<false>(
              matrix_free,
              range,
              data.dof_numbers[Is],
              data.quad_numbers[Is],
              data.first_selected_components[Is],
              CellEvalTypes::static_fe_degree,
              CellEvalTypes::static_n_q_points_1d))
          phi.emplace_back(std::make_unique<CellEvalTypes>(
            matrix_free,
            range,
            data.dof_numbers[Is],
            data.quad_numbers[Is],
            data.first_selected_components[Is]));
      }(),
      ...);
    return phi;
  }


  template <int dim,
            typename VectorizedArrayType,
            bool is_face,
            typename... EvalTypes,
            std::size_t... Is>
  void
  reinit_phi(std::vector<std::unique_ptr<
               FEEvaluationData<dim, VectorizedArrayType, is_face>>> &phi,
             const unsigned int                                       batch,
             std::index_sequence<Is...>)
  {
    static_assert(sizeof...(Is) == sizeof...(EvalTypes),
                  "Index sequence size must match number of EvalTypes");
    Assert(phi.size() >= sizeof...(Is),
           ExcLowerRange(phi.size(), sizeof...(Is)));
    (static_cast<EvalTypes &>(*phi[Is]).reinit(batch), ...);
    (void)batch;
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes,
            typename... FaceEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const dealii::AffineConstraints<Number>            &constraints,
    BlockMatrixType                                    &matrix,
    const std::function<void(CellEvalTypes &...)>      &cell_operation,
    const std::function<void(std::pair<FaceEvalTypes &, FaceEvalTypes &>...)>
                                                  &face_operation,
    const std::function<void(FaceEvalTypes &...)> &boundary_operation)
  {
    std::vector<const dealii::AffineConstraints<Number> *> constraints_{
      &constraints};
    compute_matrix(matrix_free,
                   constraints_,
                   matrix,
                   cell_operation,
                   face_operation,
                   boundary_operation);
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes,
            typename... FaceEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType>           &matrix_free,
    const std::vector<const dealii::AffineConstraints<Number> *> &constraints,
    BlockMatrixType                                              &matrix,
    const std::function<void(CellEvalTypes &...)> &cell_operation,
    const std::function<void(std::pair<FaceEvalTypes &, FaceEvalTypes &>...)>
                                                  &face_operation,
    const std::function<void(FaceEvalTypes &...)> &boundary_operation)
  {
    auto cell_eval_index    = std::index_sequence_for<CellEvalTypes...>{};
    auto face_eval_index    = std::index_sequence_for<FaceEvalTypes...>{};
    auto in_face_eval_index = even_sequence<sizeof...(FaceEvalTypes)>{};
    auto ex_face_eval_index = odd_sequence<sizeof...(FaceEvalTypes)>{};
    MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, VectorizedArrayType, false>
        data_cell;

    auto const index_vector = make_index_vector(cell_eval_index);
    data_cell.dof_numbers   = index_vector;
    data_cell.quad_numbers  = index_vector;
    data_cell.n_components.resize(index_vector.size());
    for (size_t i = 0; i < index_vector.size(); ++i)
      {
        data_cell.n_components[i] =
          matrix_free.get_dof_handler(index_vector[i]).get_fe().n_components();
      }
    data_cell.first_selected_components =
      std::vector<unsigned int>(index_vector.size(), 0);
    data_cell.batch_type = std::vector<unsigned int>(index_vector.size(), 0);

    data_cell.op_create =
      [&](const std::pair<unsigned int, unsigned int> &range) {
        return create_cell_phi_vector<dim,
                                      Number,
                                      VectorizedArrayType,
                                      CellEvalTypes...>(matrix_free,
                                                        range,
                                                        data_cell,
                                                        cell_eval_index);
      };

    data_cell.op_reinit = [&cell_eval_index](auto &phi, const unsigned batch) {
      reinit_phi<dim, VectorizedArrayType, false, CellEvalTypes...>(
        phi, batch, cell_eval_index);
    };

    if (cell_operation)
      data_cell.op_compute = [&](auto &phi) {
        auto phi_tuple =
          generate_tuple_from_vector<dim,
                                     VectorizedArrayType,
                                     false,
                                     CellEvalTypes...>(phi, cell_eval_index);
        std::apply(cell_operation, phi_tuple);
      };


    MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, VectorizedArrayType, true>
        data_face;

    data_face.dof_numbers  = interleave(index_vector, index_vector);
    data_face.quad_numbers = interleave(index_vector, index_vector);
    data_face.n_components.resize(index_vector.size());
    for (size_t i = 0; i < index_vector.size(); ++i)
      {
        data_face.n_components[i] =
          matrix_free.get_dof_handler(index_vector[i]).get_fe().n_components();
      }
    std::vector<unsigned int> first_selected_components(index_vector.size(), 0);
    data_face.first_selected_components =
      interleave(first_selected_components, first_selected_components);
    data_face.batch_type =
      interleave(std::vector<unsigned int>(index_vector.size(), 1),
                 std::vector<unsigned int>(index_vector.size(), 2));

    data_face.op_create =
      [&](const std::pair<unsigned int, unsigned int> &range) {
        return create_face_phi_vector<dim,
                                      Number,
                                      VectorizedArrayType,
                                      FaceEvalTypes...>(
          matrix_free,
          range,
          data_face.first_selected_components,
          face_eval_index);
      };

    data_face.op_reinit = [&in_face_eval_index,
                           &ex_face_eval_index](auto          &phi,
                                                const unsigned batch) {
      reinit_phi<dim, VectorizedArrayType, true, FaceEvalTypes...>(
        phi, batch, in_face_eval_index);
      reinit_phi<dim, VectorizedArrayType, true, FaceEvalTypes...>(
        phi, batch, ex_face_eval_index);
    };

    if (face_operation)
      data_face.op_compute = [&](auto &phi) {
        auto phi_tuple =
          generate_pair_tuple_from_vector<dim,
                                          VectorizedArrayType,
                                          true,
                                          FaceEvalTypes...>(phi,
                                                            face_eval_index);
        std::apply(face_operation, phi_tuple);
      };

    MatrixFreeTools::internal::
      ComputeMatrixScratchData<dim, VectorizedArrayType, true>
        data_boundary;

    data_boundary.dof_numbers  = index_vector;
    data_boundary.quad_numbers = index_vector;
    data_boundary.n_components.resize(index_vector.size());
    for (size_t i = 0; i < index_vector.size(); ++i)
      {
        data_boundary.n_components[i] =
          matrix_free.get_dof_handler(index_vector[i]).get_fe().n_components();
      }
    data_boundary.first_selected_components =
      std::vector<unsigned int>(index_vector.size(), 0);
    data_boundary.batch_type =
      std::vector<unsigned int>(index_vector.size(), 1);

    data_boundary.op_create =
      [&](const std::pair<unsigned int, unsigned int> &range) {
        return create_boundary_phi_vector<dim,
                                          Number,
                                          VectorizedArrayType,
                                          FaceEvalTypes...>(
          matrix_free,
          range,
          data_boundary.first_selected_components,
          face_eval_index);
      };

    data_boundary.op_reinit = [&face_eval_index](auto          &phi,
                                                 const unsigned batch) {
      reinit_phi<dim, VectorizedArrayType, true, FaceEvalTypes...>(
        phi, batch, face_eval_index);
    };

    if (boundary_operation)
      data_boundary.op_compute = [&](auto &phi) {
        auto phi_tuple =
          generate_tuple_from_vector<dim,
                                     VectorizedArrayType,
                                     true,
                                     FaceEvalTypes...>(phi, face_eval_index);
        std::apply(boundary_operation, phi_tuple);
      };
    MatrixFreeTools::internal::compute_matrix(
      matrix_free, constraints, data_cell, data_face, data_boundary, matrix);
  }


  template <typename CLASS,
            int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number>                    &constraints,
    BlockMatrixType                                    &matrix,
    void (CLASS::*cell_operation)(CellEvalTypes &...) const,
    const CLASS *owning_class)
  {
    std::vector<const dealii::AffineConstraints<Number> *> constraints_{
      &constraints};
    std::function<void(CellEvalTypes & ...)> cell_op_func =
      [&owning_class, &cell_operation](CellEvalTypes &...evals) {
        (owning_class->*cell_operation)(evals...);
      };

    std::function<void()> face_op_func{};
    std::function<void()> boundary_op_func{};

    compute_matrix(matrix_free,
                   constraints_,
                   matrix,
                   cell_op_func,
                   face_op_func,
                   boundary_op_func);
  }

  template <typename CLASS,
            int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType>           &matrix_free,
    const std::vector<const dealii::AffineConstraints<Number> *> &constraints,
    BlockMatrixType                                              &matrix,
    void (CLASS::*cell_operation)(CellEvalTypes &...) const,
    const CLASS *owning_class)
  {
    std::function<void(CellEvalTypes & ...)> cell_op_func =
      [&owning_class, &cell_operation](CellEvalTypes &...evals) {
        (owning_class->*cell_operation)(evals...);
      };

    std::function<void()> face_op_func{};
    std::function<void()> boundary_op_func{};

    compute_matrix(matrix_free,
                   constraints,
                   matrix,
                   cell_op_func,
                   face_op_func,
                   boundary_op_func);
  }

  template <typename CLASS,
            int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes,
            typename... FaceEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number>                    &constraints,
    BlockMatrixType                                    &matrix,
    void (CLASS::*cell_operation)(CellEvalTypes &...) const,
    void (CLASS::*face_operation)(
      std::pair<FaceEvalTypes &, FaceEvalTypes &>...) const,
    void (CLASS::*boundary_operation)(FaceEvalTypes &...) const,
    const CLASS *owning_class)
  {
    std::vector<const dealii::AffineConstraints<Number> *> constraints_{
      &constraints};
    compute_matrix(matrix_free,
                   constraints_,
                   matrix,
                   cell_operation,
                   face_operation,
                   boundary_operation,
                   owning_class);
  }

  template <typename CLASS,
            int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockMatrixType,
            typename... CellEvalTypes,
            typename... FaceEvalTypes>
  void
  compute_matrix(
    const MatrixFree<dim, Number, VectorizedArrayType>           &matrix_free,
    const std::vector<const dealii::AffineConstraints<Number> *> &constraints,
    BlockMatrixType                                              &matrix,
    void (CLASS::*cell_operation)(CellEvalTypes &...) const,
    void (CLASS::*face_operation)(
      std::pair<FaceEvalTypes &, FaceEvalTypes &>...) const,
    void (CLASS::*boundary_operation)(FaceEvalTypes &...) const,
    const CLASS *owning_class)
  {
    compute_matrix(
      matrix_free,
      constraints,
      matrix,
      [&](CellEvalTypes &...evals) {
        (owning_class->*cell_operation)(evals...);
      },
      [&](std::pair<FaceEvalTypes &, FaceEvalTypes &>... face_evals) {
        (owning_class->*face_operation)(face_evals...);
      },
      [&](FaceEvalTypes &...boundary_evals) {
        (owning_class->*boundary_operation)(boundary_evals...);
      });
  }


  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_block_matrix_block(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number>                    &constraints,
    MatrixType                                         &matrix,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)>
                      &cell_operation,
    const unsigned int dof_r_no                   = 0,
    const unsigned int dof_c_no                   = 0,
    const unsigned int quad_r_no                  = 0,
    const unsigned int quad_c_no                  = 0,
    const unsigned int first_selected_r_component = 0,
    const unsigned int first_selected_c_component = 0)
  {}

} // namespace dealii
