from src.pdmo.classes.pointclass import PointList, Point
import numpy as np
from src.pdmo.utils.convexity_measure import convexity_measure


def test_convex_hull():
    Y_list = []
    Y = PointList([Point((y1,y2)) for y1,y2 in zip(np.linspace(0, 100, 20), np.linspace(100, 0, 20))])
    Y_list.append(Y)
    Ynew = PointList([y for y in Y if not y <= Point((60,60))] + [Point((60,60))])
    Y_list.append(Ynew)
    Ynew2 = PointList([y for y in Y if not Point((40,40)) <= y]+ [Point((40,40))])
    Y_list.append(Ynew2)
    # Points on the unit circle in the first quadrant
    Ynew3 = PointList([Point((np.cos(t), np.sin(t))) for t in np.linspace(0, np.pi/2, 20)])
    # Points on the unit circle in the third quadrant
    Y_list.append(Ynew3)
    Ynew4 = PointList([Point((-np.cos(t), -np.sin(t))) for t in np.linspace(0, np.pi/2, 20)])
    Y_list.append(Ynew4)
    
    for i,Y in enumerate(Y_list):
        print(f"Testing convex hull for $Y^{i}$ with {len(Y.points)} points")
        figname = f'figures/test_convexity_{i}.' + 'png' # or 'pdf'
        print('Saving convexity measure plots', figname)
        M = convexity_measure(Y,  figname = figname) # with plots saved as figname (slower)
        M = convexity_measure(Y) # without plots (fast)
        print(f"Convexity measure for $Y^{i}$: {M:.2f}")


if __name__ == "__main__":
    test_convex_hull()

