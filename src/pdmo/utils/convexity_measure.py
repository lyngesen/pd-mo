from src.pdmo.classes.plotter import Plotter

def convexity_measure(Y, figname : str = ''):
    ''' Calculate the convexity measure of a set of points Y.
    args:
    Y (PointList): A list of points.
    figname (str): If given, the function will plot the convex hull, hypervolume shape, and difference.
    '''

    # Y = Y.get_supported() # uncomment to use only supported points
    if figname:
        P = Plotter(ncols = 3, nrows= 2, figsize=(10,7))

    for row, ref_name in enumerate(['ideal', 'nadir']):
        if ref_name == 'ideal':
            ref = Y.get_ideal()
        elif ref_name == 'nadir':
            ref = Y.get_nadir()
        else:
            raise ValueError(f"Unknown reference point: {ref_name}")

        # T = (y_N[0] - y_I[0]) * (y_N[1] - y_I[1])
        # print(f"Area of rectangle: {T}")

        conv = Y._convex_hull_with_ref(ref=ref)

        # print(f"Convex hull: {conv}")
        # print(f"Convex hull area: {conv.area}")


        # this is the reference shape defined by the union of boxes between points of Y and the reference point
        hypervolume_shape = Y._hypervolume_shape(ref=ref)

        # calculate the difference between the convex hull and the hypervolume shape
        difference = conv.difference(hypervolume_shape)

        T = conv.area # for normalization

        if figname: # plot if figname is given
            P.plot(conv, ax=P.axs[row,0])
            ax_n = list(P.axs)[row]
            for ax in ax_n:
                ax.set_xticks([])
                ax.set_yticks([])
                P.plot(Y, l ='$\mathcal{Y}_N$ ', ax=ax)
                ref.plot(ax=ax, color='red', l='$y^r$')


            P.plot(difference, ax=P.axs[row,2])
            P.plot(hypervolume_shape, ax=P.axs[row,1])
            P.axs[row,0].set_title(f'Convex hull area {conv.area/T:.2f}')
            P.axs[row,1].set_title(f'Hypervolume area {hypervolume_shape.area/T:.2f}')

        if ref_name == 'ideal':
            M_N = difference.area/T
        if ref_name == 'nadir':
            M_I = difference.area/T

    if figname: # only if figname is given
        P.axs[0,0].set_ylabel('Reference nadir point')
        P.axs[1,0].set_ylabel('Reference ideal point')

        P.axs[0,2].set_title(f'difference {M_N:.2f} = $M^N$')
        P.axs[1,2].set_title(f'difference {M_I:.2f} = $M^I$')
        P.axs[1,1].set_xlabel(f"Ration M^I/M^N = {M_I/M_N:.2f}$ \n" + "$(1-M^N)/2 + M^I/2 = $" + f"{(1-M_N)/2 + M_I/2:.2f}" + "$((M^I) + (1-M^N))/2=$"+ f"{M_I/2 + (1-M_N)/2:.2f}$")
        # P.fig.savefig(f'docs/figures/tests/convexity_measure/measure_id{figname}.pdf')

        P.fig.savefig(figname)

    def combining_MI_MN(M_I, M_N):
        # should be between 0 and 1. 1 For convex, 0 for concave
        return (1 - M_N) / 2 + M_I / 2

    return combining_MI_MN(M_I, M_N)

