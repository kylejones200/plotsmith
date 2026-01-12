# API Stability Policy

## Overview

PlotSmith follows semantic versioning (SemVer) and provides API stability guarantees to help users plan upgrades and migrations.

## Version Numbering

PlotSmith uses semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes that require code modifications
- **MINOR**: New features and functionality (backward compatible)
- **PATCH**: Bug fixes and minor improvements (backward compatible)

## Stability Guarantees

### Stable APIs (Public)

The following APIs are considered stable and will maintain backward compatibility within a major version:

1. **Workflow Functions** (`plotsmith.workflows`):
   - All `plot_*` functions (e.g., `plot_timeseries`, `plot_bar`, `plot_heatmap`)
   - `figure()` and `small_multiples()` helper functions
   - Function signatures and return types
   - Default parameter values

2. **View Objects** (`plotsmith.objects`):
   - All View dataclasses (e.g., `SeriesView`, `BarView`, `ScatterView`)
   - View object structure and required fields
   - `FigureSpec` dataclass

3. **Styling Helpers** (`plotsmith.primitives`):
   - `minimal_axes()`, `tidy_axes()`
   - `event_line()`, `direct_label()`, `note()`, `accent_point()`
   - `ACCENT` constant

### Internal APIs (Unstable)

The following are considered internal and may change without notice:

1. **Task Classes** (`plotsmith.tasks`):
   - Task class implementations
   - Internal task methods
   - Task execution details

2. **Primitive Functions** (`plotsmith.primitives.draw`):
   - Low-level drawing functions (e.g., `draw_series`, `draw_bar`)
   - Internal styling functions
   - Matplotlib integration details

3. **Validation Functions** (`plotsmith.objects.validate`):
   - Internal validation logic
   - Validation function signatures

## Deprecation Policy

### Deprecation Process

1. **Announcement**: Deprecated features are announced in the CHANGELOG.md
2. **Warning Period**: Deprecated features emit `DeprecationWarning` for at least one minor version
3. **Removal**: Deprecated features are removed in the next major version

### Current Deprecations

See `CHANGELOG.md` for current deprecations.

## Migration Guides

When breaking changes occur, migration guides are provided in:
- `CHANGELOG.md` - Detailed change descriptions
- Release notes on GitHub
- Documentation updates

## Exceptions

Breaking changes may occur without a major version bump for:
- Security vulnerabilities
- Critical bug fixes that require API changes
- Removal of clearly marked experimental features

## Recommendations

1. **Pin Major Versions**: Use `plotsmith>=1.0.0,<2.0.0` to get bug fixes while avoiding breaking changes
2. **Monitor CHANGELOG**: Review CHANGELOG.md before upgrading
3. **Test Upgrades**: Test your code after upgrading minor/patch versions
4. **Report Issues**: Report any unexpected breaking changes

## Support Timeline

- **Current Version**: Full support
- **Previous Major Version**: Security fixes only (6 months)
- **Older Versions**: No support

## Questions?

If you have questions about API stability or migration, please:
- Open an issue on GitHub
- Check the documentation
- Review the CHANGELOG.md

