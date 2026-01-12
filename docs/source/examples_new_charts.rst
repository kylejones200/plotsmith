New Chart Types Examples
========================

This document provides examples for all the new chart types added to PlotSmith.

Waterfall Chart
---------------

Waterfall charts show how an initial value is affected by a series of positive or negative changes.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_waterfall

    df = pd.DataFrame({
        'Category': ['Revenue', 'COGS', 'Operating Expenses', 'Taxes', 'Net Profit'],
        'Value': [100000, -50000, -20000, -10000, 0],
    })

    fig, ax = plot_waterfall(
        df,
        categories_col='Category',
        values_col='Value',
        title='Financial Waterfall'
    )

With measure types:

.. code-block:: python

    df = pd.DataFrame({
        'Category': ['Year beginning', 'Profit1', 'Loss1', 'Q1', 'Profit2', 'Loss2', 'Q2'],
        'Value': [100, 50, -20, 0, 40, -10, 0],
        'measure': ['absolute', 'relative', 'relative', 'total', 'relative', 'relative', 'total'],
    })

    fig, ax = plot_waterfall(
        df,
        categories_col='Category',
        values_col='Value',
        measure_col='measure'
    )

Waffle Chart
------------

Waffle charts display parts of a whole using a grid of squares.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_waffle

    df = pd.DataFrame({
        'Party': ['A', 'B', 'C'],
        'Seats': [10, 20, 15]
    })

    fig, ax = plot_waffle(
        df,
        category_col='Party',
        value_col='Seats',
        title='Election Results'
    )

Dumbbell Chart
--------------

Dumbbell charts compare two values for each category.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_dumbbell

    df = pd.DataFrame({
        'country': ['France', 'Germany', 'Greece'],
        '1952': [67.41, 67.5, 65.86],
        '2002': [79.59, 78.67, 78.25],
    })

    fig, ax = plot_dumbbell(
        df,
        categories_col='country',
        values1_col='1952',
        values2_col='2002',
        title='Life Expectancy Change'
    )

Range Chart
-----------

Range charts show the range between two values for each category.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_range

    df = pd.DataFrame({
        'category': ['A', 'B', 'C'],
        'min': [10, 20, 15],
        'max': [15, 25, 20]
    })

    fig, ax = plot_range(
        df,
        categories_col='category',
        values1_col='min',
        values2_col='max'
    )

Lollipop Chart
--------------

Lollipop charts combine the benefits of bar charts and scatter plots.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_lollipop

    df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [10, 25, 15, 30]
    })

    fig, ax = plot_lollipop(
        df,
        categories_col='Category',
        values_col='Value',
        title='Category Values'
    )

Slope Chart
-----------

Slope charts show changes between two time points.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_slope

    # Long format
    df = pd.DataFrame({
        'year': [2020, 2020, 2020, 2023, 2023, 2023],
        'country': ['USA', 'Canada', 'Mexico', 'USA', 'Canada', 'Mexico'],
        'gdp': [20.94, 1.64, 1.07, 22.99, 1.99, 1.27],
    })

    fig, ax = plot_slope(
        data_frame=df,
        x='year',
        y='gdp',
        group='country',
        title='GDP Change'
    )

    # Wide format
    df_wide = pd.DataFrame({
        'Year': [2020, 2021],
        'USA': [10, 12],
        'Canada': [12, 15]
    })

    fig, ax = plot_slope(
        df=df_wide,
        categories_col='Year',
        values_cols=['USA', 'Canada']
    )

Metric Display
--------------

Metric displays show a single value with optional delta.

.. code-block:: python

    from plotsmith import plot_metric

    fig, ax = plot_metric(
        title='Sales',
        value=12345,
        delta=123,
        prefix='$',
        title='Monthly Sales'
    )

Box Plot
--------

Box plots show the distribution of data across categories.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_box

    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })

    fig, ax = plot_box(
        df,
        x='category',
        y='value',
        title='Distribution by Category'
    )

Violin Plot
-----------

Violin plots combine box plots with kernel density estimation.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_violin

    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })

    fig, ax = plot_violin(
        df,
        x='category',
        y='value',
        show_means=True
    )

Enhanced Scatter Plot
---------------------

Scatter plots with color and size mapping.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_scatter

    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'size': [10, 20, 30, 40, 50]
    })

    # With color mapping
    fig, ax = plot_scatter(df, x='x', y='y', color='category')

    # With size mapping
    fig, ax = plot_scatter(df, x='x', y='y', size='size')

    # With both
    fig, ax = plot_scatter(df, x='x', y='y', color='category', size='size')

Correlation Heatmap
-------------------

Correlation heatmaps visualize correlation matrices.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from plotsmith import plot_correlation

    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
    })

    fig, ax = plot_correlation(
        df,
        method='pearson',
        annotate=True,
        title='Feature Correlations'
    )

Forecast Comparison
-------------------

Compare multiple forecast models against actual values.

.. code-block:: python

    import pandas as pd
    from plotsmith import plot_forecast_comparison

    actual = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))

    forecasts = {
        'Model1': pd.Series([1.1, 2.1, 2.9, 4.1, 4.9], index=actual.index),
        'Model2': pd.Series([0.9, 2.2, 3.1, 3.9, 5.1], index=actual.index),
    }

    fig, ax = plot_forecast_comparison(
        actual,
        forecasts,
        title='Forecast Comparison'
    )

With confidence intervals:

.. code-block:: python

    intervals = {
        'Model1': (
            pd.Series([0.9, 1.9, 2.7, 3.9, 4.7], index=actual.index),
            pd.Series([1.3, 2.3, 3.1, 4.3, 5.1], index=actual.index),
        )
    }

    fig, ax = plot_forecast_comparison(
        actual,
        forecasts,
        intervals=intervals
    )


