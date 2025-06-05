# public libary imports
import numpy as np
from matplotlib import pyplot as plt
import json

# local imports
from src.pdmo.classes.pointclass import PointList, Point
from src.pdmo.utils.convexity_measure import convexity_measure, get_M_N_measure


def test_convex_hull():
    for i,Y in enumerate(Y_list):
        print(f"Testing convex hull for $Y^{i}$ with {len(Y.points)} points")
        figname = f'figures/test_convexity_{i}.' + 'png' # or 'pdf'
        print('Saving convexity measure plots', figname)
        M = convexity_measure(Y,  figname = figname) # with plots saved as figname (slower)
        M = convexity_measure(Y) # without plots (fast)
        print(f"Convexity measure for $Y^{i}$: {M:.2f}")

def calculate_M_N_with_plot():

    result_dict = {}

    i= 0
    Y = Y_list[0]  # Example: using the first PointList from Y_list
    for i, Y in enumerate(Y_list):
        fig, ax = plt.subplots(figsize=(4, 2))
        filename = f'figures/Y_{i}_plot.png'
        Y.plot(ax=ax, color='black',marker='o')
        fig.savefig(filename)
        M_N = get_M_N_measure(Y)
        M = convexity_measure(Y)
        print(f"Convexity measure M_N for $Y^{i}$: {M_N:.2f}")
        print(f"Convexity measure M for $Y^{i}$: {M_N:.2f}")
        result_dict[f'Y_{i}'] = {'M_N': M_N, 'M': M, 'filename': filename}

    with open('convexity_measure_results.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    # add to readme.md file
    search_string = "<!---RESULT_TABLE-->"

    with open('README.md', 'r') as f:
        lines = f.readlines()
        lines_out = lines[:lines.index(search_string + '\n') + 1]
        # lines_out.append("<table>")
        
    for key, value in result_dict.items():
        # format <tr><td><img src="figures/Y_0_plot.png"></td><td>1</td><td>2</td></tr>
        lines_out.append(f'\n<tr><td><img src="{value["filename"]}"></td><td>{1-value["M_N"]:.2f}</td><td>{value["M"]:.2f}</td></tr>')

    lines_out.append("</table>")
    with open('README.md', 'w') as f:
        f.writelines(lines_out)

if __name__ == "__main__":

    Y_list = []

    Y = PointList.from_json('instances/tests/modc_Yn_save_test_1.json')
    Y_list.append(Y)
    Y = PointList.from_json('instances/tests/modc_Yn_save_test_0.json')
    Y_list.append(Y)
    Y = PointList.from_json('instances/subproblems_local_sets/sp-2-10-l_1.json')
    Y_list.append(Y)
    Y = PointList.from_json('instances/subproblems_local_sets/sp-2-100-l_1.json')
    Y_list.append(Y)
    Y = PointList.from_json('instances/subproblems_local_sets/sp-2-100-u_1.json')
    Y_list.append(Y)
    Y = PointList.from_json('instances/subproblems_local_sets/sp-2-100-m_1.json')
    Y_list.append(Y)
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
    


    # test_convex_hull()

    calculate_M_N_with_plot()

