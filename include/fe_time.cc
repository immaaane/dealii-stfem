#include "fe_time.h"

namespace dealii
{
  bool
  is_space_lvl(MGType mg)
  {
    return mg == MGType::h || mg == MGType::p;
  };
  bool
  is_time_lvl(MGType mg)
  {
    return mg == MGType::tau || mg == MGType::k;
  };

  unsigned int
  create_next_polynomial_coarsening_degree(
    const unsigned int previous_fe_degree,
    const MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
                &p_sequence,
    unsigned int k_min)
  {
    switch (p_sequence)
      {
        case MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          bisect:
          return std::max(previous_fe_degree / 2, 0u);
        case MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          decrease_by_one:
          return std::max(previous_fe_degree - 1, 0u);
        case MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          go_to_one:
          return k_min;
        default:
          DEAL_II_NOT_IMPLEMENTED();
          return 0u;
      }
  }

  std::vector<unsigned int>
  get_poly_mg_sequence(
    unsigned int const k_max,
    unsigned int const k_min,
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType const
      p_seq)
  {
    std::vector<unsigned int> degrees{k_max};
    if (degrees.back() == k_min)
      return degrees;
    while (degrees.back() > k_min)
      degrees.push_back(dealii::create_next_polynomial_coarsening_degree(
        degrees.back(), p_seq, k_min));

    std::reverse(degrees.begin(), degrees.end());
    return degrees;
  }

  std::vector<MGType>
  get_mg_sequence(unsigned int              n_sp_lvl,
                  std::vector<unsigned int> k_seq,
                  std::vector<unsigned int> p_seq,
                  unsigned int const        n_timesteps_at_once,
                  unsigned int const        n_timesteps_at_once_min,
                  MGType                    lower_lvl,
                  CoarseningType            coarsening_type,
                  bool                      time_before_space,
                  bool                      use_p_multigrid_space,
                  bool                      zip_from_back)
  {
    Assert(n_sp_lvl >= 1, ExcLowerRange(n_sp_lvl, 1));
    Assert(k_seq.size() >= 1, ExcLowerRange(k_seq.size(), 1));
    unsigned int n_k_lvl = k_seq.size() - 1;
    unsigned int n_t_lvl = log2(n_timesteps_at_once / n_timesteps_at_once_min);
    MGType       upper_lvl = (lower_lvl == MGType::k) ? MGType::tau : MGType::k;
    MGType       lower_lvl_s = (lower_lvl == MGType::k) ? MGType::p : MGType::h;
    MGType       upper_lvl_s = (lower_lvl == MGType::k) ? MGType::h : MGType::p;
    unsigned int n_ll        = (lower_lvl == MGType::k) ? n_k_lvl : n_t_lvl;
    unsigned int n_ul        = (lower_lvl == MGType::k) ? n_t_lvl : n_k_lvl;
    unsigned int n_p_lvl     = (use_p_multigrid_space) ? p_seq.size() - 1 : 0;
    unsigned int n_ll_s = (lower_lvl == MGType::k) ? n_p_lvl : n_sp_lvl - 1;
    unsigned int n_ul_s = (lower_lvl == MGType::k) ? n_sp_lvl - 1 : n_p_lvl;
    std::vector<MGType> time_levels(n_ll, lower_lvl);
    time_levels.insert(time_levels.end(), n_ul, upper_lvl);
    std::vector<MGType> space_levels(n_ll_s, lower_lvl_s);
    space_levels.insert(space_levels.end(), n_ul_s, upper_lvl_s);
    std::vector<MGType> mg_type_level;
    auto append_levels = [&](const auto &first, const auto &second) {
      mg_type_level.insert(mg_type_level.end(), first.begin(), first.end());
      mg_type_level.insert(mg_type_level.end(), second.begin(), second.end());
    };
    auto append_levels_reverse = [&](const auto &first, const auto &second) {
      mg_type_level.insert(mg_type_level.end(), first.rbegin(), first.rend());
      mg_type_level.insert(mg_type_level.end(), second.rbegin(), second.rend());
    };
    if (coarsening_type == CoarseningType::space_or_time)
      if (zip_from_back)
        append_levels_reverse(time_before_space ? time_levels : space_levels,
                              time_before_space ? space_levels : time_levels);
      else
        append_levels(time_before_space ? time_levels : space_levels,
                      time_before_space ? space_levels : time_levels);
    else
      {
        size_t time_size = time_levels.size(), space_size = space_levels.size();
        size_t max_levels = std::max(time_size, space_size);
        auto   get_index =
          [&](const std::vector<MGType> &levels, size_t i, bool reverse) {
            return reverse ? levels[levels.size() - 1 - i] : levels[i];
          };
        for (size_t i = 0; i < max_levels; ++i)
          {
            if (i < (time_before_space ? time_size : space_size))
              mg_type_level.push_back(
                get_index(time_before_space ? time_levels : space_levels,
                          i,
                          zip_from_back));
            if (i < (time_before_space ? space_size : time_size))
              mg_type_level.push_back(
                get_index(time_before_space ? space_levels : time_levels,
                          i,
                          zip_from_back));
          }
        if (zip_from_back)
          std::reverse(mg_type_level.begin(), mg_type_level.end());
      }
    return mg_type_level;
  }

  std::vector<unsigned int>
  get_precondition_stmg_types(std::vector<MGType> const &mg_type_level,
                              CoarseningType             coarsening_type,
                              bool                       time_before_space,
                              [[maybe_unused]] bool      zip_from_back,
                              SupportedSmoothers         smoother)
  {
    std::vector<unsigned int> ret(mg_type_level.size() + 1,
                                  static_cast<unsigned int>(smoother));
    if (coarsening_type == CoarseningType::space_or_time)
      return ret;
    for (size_t i = 0; i < mg_type_level.size() - 1; ++i)
      // maybe replace by time_before_space == zip_from_back
      if (time_before_space ?
            is_space_lvl(mg_type_level[i]) &&
              is_time_lvl(mg_type_level[i + 1]) :
            is_time_lvl(mg_type_level[i]) && is_space_lvl(mg_type_level[i + 1]))
        ret[i]     = static_cast<unsigned int>(smoother),
        ret[i + 1] = static_cast<unsigned int>(SupportedSmoothers::Identity),
        ++i;
    return ret;
  }

  Quadrature<1>
  get_time_quad(TimeStepType type, unsigned int const r)
  {
    if (type == TimeStepType::DG)
      return QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right);
    else if (type == TimeStepType::CGP)
      return QGaussLobatto<1>(r + 1);
    else
      return Quadrature<1>();
  }

  std::vector<Polynomials::Polynomial<double>>
  get_time_basis(TimeStepType type, unsigned int const r)
  {
    auto quad_time = get_time_quad(type, r);
    return Polynomials::generate_complete_Lagrange_basis(
      quad_time.get_points());
  }

  std::vector<size_t>
  get_fe_q_permutation(FE_Q<1> const &fe_time)
  {
    size_t const        n_dofs = fe_time.n_dofs_per_cell();
    std::vector<size_t> permutation(n_dofs, 0);
    std::iota(permutation.begin() + 1, permutation.end() - 1, 2);
    permutation.back() = 1;
    return permutation;
  }

  template std::array<FullMatrix<double>, 5>
  get_fe_time_weights_wave(TimeStepType,
                           FullMatrix<double> const &,
                           FullMatrix<double> const &,
                           FullMatrix<double> const &,
                           FullMatrix<double> const &,
                           unsigned int);
  template std::array<FullMatrix<float>, 5>
  get_fe_time_weights_wave(TimeStepType,
                           FullMatrix<float> const &,
                           FullMatrix<float> const &,
                           FullMatrix<float> const &,
                           FullMatrix<float> const &,
                           unsigned int);

  template FullMatrix<double>
  get_time_evaluation_matrix(
    std::vector<Polynomials::Polynomial<double>> const &,
    unsigned int);
  template FullMatrix<float>
  get_time_evaluation_matrix(
    std::vector<Polynomials::Polynomial<double>> const &,
    unsigned int);

  template std::array<FullMatrix<double>, 4>
  get_fe_time_weights(TimeStepType,
                      unsigned int const,
                      double,
                      unsigned int,
                      double);
  template std::array<FullMatrix<float>, 4>
  get_fe_time_weights(TimeStepType,
                      unsigned int const,
                      double,
                      unsigned int,
                      double);

  template std::vector<std::array<FullMatrix<double>, 5>>
  get_fe_time_weights_wave(TimeStepType,
                           double,
                           unsigned int,
                           double,
                           std::vector<MGType> const &,
                           std::vector<unsigned int> const &);
  template std::vector<std::array<FullMatrix<float>, 5>>
  get_fe_time_weights_wave(TimeStepType,
                           double,
                           unsigned int,
                           double,
                           std::vector<MGType> const &,
                           std::vector<unsigned int> const &);

  template std::array<FullMatrix<double>, 4>
  split_lhs_rhs(std::array<FullMatrix<double>, 2> const &time_weights);
  template std::array<FullMatrix<float>, 4>
  split_lhs_rhs(std::array<FullMatrix<float>, 2> const &time_weights);

  template std::array<FullMatrix<double>, 4>
    split_lhs_rhs(std::array<FullMatrix<double>, 3>);
  template std::array<FullMatrix<float>, 4>
    split_lhs_rhs(std::array<FullMatrix<float>, 3>);

  template std::array<FullMatrix<double>, 3>
  get_dg_weights(unsigned int const, double const);

  template std::array<FullMatrix<double>, 2>
  get_cg_weights(unsigned int const, double const);

  template std::array<FullMatrix<float>, 3>
  get_dg_weights(unsigned int const, double const);

  template std::array<FullMatrix<float>, 2>
  get_cg_weights(unsigned int const, double const);

  template FullMatrix<double>
  get_time_projection_matrix(TimeStepType,
                             unsigned int const,
                             unsigned int const,
                             unsigned int const);
  template FullMatrix<float>
  get_time_projection_matrix(TimeStepType,
                             unsigned int const,
                             unsigned int const,
                             unsigned int const);

  template FullMatrix<double>
  get_time_prolongation_matrix(TimeStepType,
                               unsigned int const,
                               unsigned int const);
  template FullMatrix<float>
  get_time_prolongation_matrix(TimeStepType,
                               unsigned int const,
                               unsigned int const);

  template FullMatrix<double>
  get_time_restriction_matrix(TimeStepType,
                              unsigned int const,
                              unsigned int const);
  template FullMatrix<float>
  get_time_restriction_matrix(TimeStepType,
                              unsigned int const,
                              unsigned int const);

  template std::array<FullMatrix<double>, 4>
  get_fe_time_weights_stokes(TimeStepType,
                             unsigned int const,
                             double,
                             unsigned int,
                             double);
  template std::array<FullMatrix<float>, 4>
  get_fe_time_weights_stokes(TimeStepType,
                             unsigned int const,
                             double,
                             unsigned int,
                             double);

  template std::array<FullMatrix<double>, 4>
  get_fe_time_weights_2variable_evolutionary(TimeStepType,
                                             unsigned int const,
                                             double,
                                             unsigned int,
                                             double);
  template std::array<FullMatrix<float>, 4>
  get_fe_time_weights_2variable_evolutionary(TimeStepType,
                                             unsigned int const,
                                             double,
                                             unsigned int,
                                             double);


  template void
  swap(MutableBlockVectorSliceT<double> &, BlockVectorT<double> &);
  template void
  swap(BlockVectorT<double> &, MutableBlockVectorSliceT<double> &);

  template void
  equ(BlockVectorT<double> &, BlockVectorSliceT<double> const &);
  template void
  equ(BlockVectorT<double> &, MutableBlockVectorSliceT<double> const &);

  template void
  swap(MutableBlockVectorSliceT<float> &, BlockVectorT<float> &);
  template void
  swap(BlockVectorT<float> &, MutableBlockVectorSliceT<float> &);

  template void
  equ(BlockVectorT<float> &, BlockVectorSliceT<float> const &);
  template void
  equ(BlockVectorT<float> &, MutableBlockVectorSliceT<float> const &);
} // namespace dealii
