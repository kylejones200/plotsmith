# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Waterfall chart support (`plot_waterfall`)
- Waffle chart support (`plot_waffle`)
- Dumbbell chart support (`plot_dumbbell`)
- Range chart support (`plot_range`)
- Lollipop chart support (`plot_lollipop`)
- Slope chart support (`plot_slope`)
- Metric display support (`plot_metric`)
- Box plot support (`plot_box`)
- Violin plot support (`plot_violin`)
- Enhanced scatter plot with color/size mapping (`plot_scatter`)
- Correlation heatmap (`plot_correlation`)
- Forecast comparison plotting (`plot_forecast_comparison`)
- Custom exception hierarchy (`PlotSmithError`, `ValidationError`, `PlotError`, `TaskError`)
- Validation functions for all new chart types
- Vectorized operations for improved performance
- Shared helper functions to reduce code duplication

### Changed
- Improved vectorization in drawing functions
- Enhanced error messages with better context
- Code quality improvements (Pythonic patterns, reduced duplication)

### Fixed
- Removed duplicate numpy imports
- Fixed code duplication between dumbbell and range charts
- Improved type hints and type safety

## [0.1.5] - 2025-01-XX

### Added
- Initial release with core plotting functionality
- Time series plotting with confidence bands
- Scatter plots, histograms, bar charts, heatmaps
- Residual analysis plots
- Model comparison plots
- Backtest visualization
- 4-layer architecture with strict boundaries

[Unreleased]: https://github.com/kylejones200/plotsmith/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/kylejones200/plotsmith/releases/tag/v0.1.5


