Performance Guide
=================

Performance characteristics and optimization tips for PlotSmith.

Performance Characteristics
---------------------------

PlotSmith is designed for efficiency with vectorized operations where possible. However, performance depends on several factors:

**Data Size**
- Small datasets (< 1,000 points): Very fast (< 10ms)
- Medium datasets (1,000 - 10,000 points): Fast (< 100ms)
- Large datasets (10,000 - 100,000 points): Moderate (< 1s)
- Very large datasets (> 100,000 points): Consider sampling

**Chart Type**
- Simple plots (line, scatter): Fastest
- Statistical plots (box, violin): Moderate (requires computation)
- Heatmaps: Moderate (depends on matrix size)
- Complex charts (waterfall, waffle): Slower (more computation)

Optimization Tips
-----------------

**1. Sample Large Datasets**

For very large datasets, consider sampling:

.. code-block:: python

   import pandas as pd
   from plotsmith import plot_timeseries
   
   # Original large dataset
   large_data = pd.Series(...)  # 1M+ points
   
   # Sample for plotting
   if len(large_data) > 10000:
       sample_data = large_data.sample(10000)
       fig, ax = plot_timeseries(sample_data)
   else:
       fig, ax = plot_timeseries(large_data)

**2. Use Appropriate Backends**

For non-interactive use (CI, scripts), use Agg backend:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # Non-interactive, faster
   from plotsmith import plot_timeseries

**3. Close Figures Explicitly**

Always close figures to free memory:

.. code-block:: python

   fig, ax = plot_timeseries(data)
   # ... use plot ...
   plt.close(fig)  # Free memory

**4. Batch Operations**

When creating multiple plots, reuse figure objects:

.. code-block:: python

   from plotsmith import figure
   
   fig, axes = figure(nrows=2, ncols=2)
   # Plot to each axis
   plt.close(fig)

**5. Avoid Redundant Computations**

Cache expensive computations:

.. code-block:: python

   # Compute once
   correlation_matrix = df.corr()
   
   # Plot multiple times
   fig1, ax1 = plot_heatmap(correlation_matrix)
   fig2, ax2 = plot_heatmap(correlation_matrix, cmap='viridis')

Memory Usage
------------

PlotSmith uses memory efficiently, but large plots can consume significant memory:

- **Base memory**: ~10-50 MB
- **Per plot**: ~1-5 MB (depends on data size)
- **Large plots**: Up to 100+ MB for very large datasets

To reduce memory usage:

1. Close figures when done
2. Sample large datasets
3. Use vectorized operations
4. Avoid storing many figure objects

Benchmarking
------------

PlotSmith includes performance benchmarks. Run them with:

.. code-block:: bash

   pytest tests/test_performance.py --benchmark-only

This will show timing information for various operations.

Performance Regression Testing
-------------------------------

The CI pipeline includes performance benchmarks. If performance degrades significantly, the benchmarks will fail.

For local performance testing:

.. code-block:: bash

   pytest tests/test_performance.py -m slow --benchmark-compare

Known Performance Considerations
---------------------------------

1. **Waterfall Charts**: Require cumulative calculations - slower for many categories
2. **Waffle Charts**: Require grid computation - slower for large grids
3. **Violin Plots**: Require kernel density estimation - slower for large datasets
4. **Heatmaps**: Annotation slows down large matrices significantly

For production use with very large datasets, consider:
- Data preprocessing and sampling
- Asynchronous plotting
- Caching plot results
- Using specialized visualization libraries for extreme scales

