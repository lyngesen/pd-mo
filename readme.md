# Pareto Front convexity measure

Given a discrete Pareto Front $\mathcal{Y}_N \subset \mathbb{R}^2$ we define two polygons $P^I$ and $P^N$ using the reference points $y^I$ (ideal point) and $y^N$ (Nadir point) as follows:
- $P^I = \bigcup_{y \in \mathcal{Y}_N} Box(y, y^I)$
- $P^N = \bigcup_{y \in \mathcal{Y}_N} Box(y^N, y)$

We define the convexity measures: $M^I$ as the area of $P^I$ divided by the area of $conv(M^I)$ and $M^N$ as the area of $M^N$ divided by the area of $conv(M^N)$.

We combine these to a single measure $M = M^I + 1-M^N$

# Usage

Download the the project and create a new python enviroment with `python3 -m venv .venvname`. In the activated shell run `pip install -r requirements.txt` installing the dependencies `matplotlib, numpy` and `shapely`. The library `shapely` is used for operations on polygons (works only for $p=2$ dimensions).

Sets of points are represented with the `PointList` class from the file `classes.pointclass`. This class has a lot of methods where only a few are used such as `.get_supported()`.

A pointset is defined by an iterable of vectors. e.g. `Y = PointList([(1,2),(3,2),(4,2)])`.

See the file `main.py` on how to call the Pareto Measure. To get the measure $M \in (0,1)$ use the function `M = convexity_measure(Y,  figname = figname) # with plots saved as figname (slower)` the figname should end with `.png` or `.pdf`.

# Some plots

![](./figures/test_convexity_0.png)
![](./figures/test_convexity_1.png)
![](./figures/test_convexity_2.png)
