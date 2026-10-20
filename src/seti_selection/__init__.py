"""SETI temporal selection-function simulator."""
__version__ = "1.19.0"

from .instrument import spectral_efficiency, median_spectral_efficiency
from .temporal import (
    source_temporal_completeness,
    source_temporal_completeness_from_mjd,
    required_active_fraction_from_eirp,
    pair_pipeline_efficiency_bounds,
)
from .inference import n_eff, eta_upper_exact

from .pipeline_calibration import (
    Margot2023InjectionCalibration,
    two_scan_marginal_recovery_benchmark_envelope,
)
from .beam import (
    gaussian_beam_gain, half_power_radius, effective_eirp_threshold, eirp_margin,
    gbt_fwhm_arcmin, gaussian_beam_gain_chromatic, effective_eirp_threshold_chromatic,
)
from .gcns import (
    angular_separation_arcmin, parse_gcns_fixed_width_line,
    load_gcns_fixed_width, load_pointings, match_catalog_arrays,
    summarize_sources, reconstruct_from_local_gcns,
)
from .margot2023 import (
    DRIFT_MAX_ABS_HZ_S,
    MargotSelectionResult,
    MargotMultiEpochResult,
    aggregate_empirical_completeness,
    aggregate_empirical_multiepoch_completeness,
    pair_efficiency_scenarios,
)
from .population import (
    circular_top_hat_solid_angle,
    equivalent_top_hat_half_angle,
    random_static_alignment_probability,
    ideal_top_hat_directivity,
    eirp_from_radiated_power,
    scan_region_inclusion_probability,
    uniform_scan_duty_cycle,
    drift_support_fraction,
    normalized_quadrature_weights,
    population_average_source_completeness,
    apply_visibility_factor,
    UniformScannerGeometry,
)
from .weak_scintillation import (
    weak_modulation_index,
    fresnel_scale_m,
    weak_scintillation_timescale_s,
    lognormal_gain_survival,
    gaussian_gain_survival,
    required_pair_survival,
)
from .ism_model_audit import (
    model_ratio_summary, max_abs_difference, weak_across_band, weak_margin_factor,
)
from .general_selection import (
    population_average_from_joint_samples,
    mixture_population_average,
    bernoulli_zero_loglikelihood,
    bernoulli_zero_probability,
    eta_upper_bernoulli_exact,
    poisson_multiplicity_zero_loglikelihood,
    mu_upper_poisson_multiplicity,
    bernoulli_loglikelihood_series,
    eta_upper_series,
    summarize_bernoulli_selection,
    BernoulliSelectionSummary,
    bernoulli_zero_probability_global_mixture,
    eta_upper_bernoulli_global_mixture,
)

# v1.18 foundation retained in v1.19: exact public file-level Price et al. L-band cadence reconstruction.
from .price2020_exact_schedule import PriceExactCadence, parse_price_lband_cadence_matrix

from .system_occupancy import (
    build_system_labels, collapse_source_completeness_to_systems, angular_separation_arcsec,
)
