Quick Start
===========

Basic Usage
-----------

Here's a simple example to get started with PlotSmith:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from plotsmith import plot_timeseries

   # Create a simple time series
   dates = pd.date_range("2024-01-01", periods=50, freq="D")
   values = 10 + np.cumsum(np.random.randn(50) * 0.5)
   series = pd.Series(values, index=dates, name="Sample Data")

   # Plot it
   fig, ax = plot_timeseries(
       series,
       title="My First PlotSmith Plot",
       xlabel="Date",
       ylabel="Value"
   )

Time Series with Confidence Bands
-----------------------------------

.. code-block:: python

   from plotsmith import plot_timeseries
   import pandas as pd

   dates = pd.date_range("2023-01-01", periods=100, freq="D")
   values = 50 + 10 * np.sin(2 * np.pi * np.arange(100) / 30)
   
   lower = pd.Series(values - 5, index=dates)
   upper = pd.Series(values + 5, index=dates)
   mean_series = pd.Series(values, index=dates, name="Forecast")

   bands = {"95% Confidence Interval": (lower, upper)}

   fig, ax = plot_timeseries(
       mean_series,
       bands=bands,
       title="Forecast with Confidence Bands"
   )

Residual Analysis
-----------------

.. code-block:: python

   from plotsmith import plot_residuals
   import numpy as np

   actual = np.array([1, 2, 3, 4, 5])
   predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

   fig, ax = plot_residuals(actual, predicted, plot_type="scatter")

Heatmap
-------

.. code-block:: python

   from plotsmith import plot_heatmap
   import pandas as pd

   # Correlation matrix
   corr_matrix = df.corr()
   fig, ax = plot_heatmap(
       corr_matrix,
       annotate=True,
       cmap="RdYlGn",
       vmin=-1,
       vmax=1
   )

Next Steps
----------

- See :doc:`architecture` for an overview of the 4-layer architecture
- Check out :doc:`examples/index` for comprehensive examples
- Browse the :doc:`api/index` for the full API reference

