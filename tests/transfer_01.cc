#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/multigrid/mg_base.h>

using namespace dealii;

template <typename BlockVectorType>
class MGTransferST : public MGTransferBase<BlockVectorType>
{
public:
  using Number = typename BlockVectorType::value_type;

  void
  prolongate(const unsigned int,
             BlockVectorType &,
             const BlockVectorType &) const final
  {
    AssertThrow(false, ExcNotImplemented());
  }

  void
  prolongate_and_add(const unsigned int     to_level,
                     BlockVectorType       &dst,
                     const BlockVectorType &src) const final
  {
    const auto &prolongation_matrix = prolongation_matrices[to_level - 1];

    Vector<Number> dst_local(dst.n_blocks());
    Vector<Number> src_local(src.n_blocks());

    for (const auto i : dst.block(0).locally_owned_elements())
      {
        for (unsigned int j = 0; j < src_local.size(); ++j)
          src_local[j] = src.block(j)[i];

        prolongation_matrix.vmult(dst_local, src_local);

        for (unsigned int j = 0; j < dst_local.size(); ++j)
          dst.block(j)[i] = dst_local[j];
      }
  }

  void
  restrict_and_add(const unsigned int     from_level,
                   BlockVectorType       &dst,
                   const BlockVectorType &src) const final
  {
    const auto &prolongation_matrix = prolongation_matrices[from_level - 1];

    Vector<Number> dst_local(dst.n_blocks());
    Vector<Number> src_local(src.n_blocks());

    for (const auto i : dst.block(0).locally_owned_elements())
      {
        for (unsigned int j = 0; j < src_local.size(); ++j)
          src_local[j] = src.block(j)[i];

        prolongation_matrix.Tvmult(dst_local, src_local);

        for (unsigned int j = 0; j < dst_local.size(); ++j)
          dst.block(j)[i] = dst_local[j];
      }
  }

  template <int dim>
  void
  copy_to_mg(const DoFHandler<dim>          &dof_handler,
             MGLevelObject<BlockVectorType> &dst,
             const BlockVectorType          &src) const
  {
    (void)dof_handler;

    const unsigned int min_level = 0;
    const unsigned int max_level = prolongation_matrices.size();

    dst.resize(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        const unsigned int n_blocks =
          (l == max_level) ? prolongation_matrices[max_level - 1].n() :
                             prolongation_matrices[max_level].m();

        dst[l].reinit(n_blocks);

        for (unsigned int b = 0; b < n_blocks; ++b)
          dst[l].block(b).reinit(partitioners[l]);
      }

    dst[max_level] = src;
  }

  template <int dim>
  void
  copy_from_mg(const DoFHandler<dim>                &dof_handler,
               BlockVectorType                      &dst,
               const MGLevelObject<BlockVectorType> &src) const
  {
    (void)dof_handler;

    const unsigned int max_level = prolongation_matrices.size();

    dst = src[max_level];
  }

private:
  std::vector<FullMatrix<Number>> prolongation_matrices; // TODO: fill
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
    partitioners; // TODO: fill
};

int
main(int argc, char **argv)
{
  return 0; // TODO: just compile

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 2;
  using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;

  DoFHandler<dim> dof_handler;

  BlockVectorType                dst, src;
  MGLevelObject<BlockVectorType> mg_dst, mg_src;

  MGTransferST<BlockVectorType> transfer;

  transfer.prolongate_and_add(0, dst, src);
  transfer.restrict_and_add(0, dst, src);

  transfer.copy_to_mg(dof_handler, mg_dst, src);
  transfer.copy_from_mg(dof_handler, dst, mg_src);
}
