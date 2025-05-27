import matplotlib.pyplot as plt
# from classes.geom import AsmoPoint, LineString, Polygon, Bound, Rectangle, Point, Line
# from classes.problem import Solution, SolutionList
from src.pdmo.classes.pointclass import Point
import shapely
import matplotlib as mpl
from shapely import Polygon

class Plotter:
    plot_dir = './docs/figures/'

    def __init__(self, nrows=1, ncols=1, figsize = (10,4), ax_mosaic: list[list[str]]|None = None ):
        self.filename = 'testing.pdf'
        if ax_mosaic:
            self.axs = ax_mosaic
            self.fig = plt.figure(figsize=figsize)
            self.axs = self.fig.subplot_mosaic(ax_mosaic)
            return
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            self.axs = [self.axs]
        self.axs : list[plt.Axes] = self.axs

    def add_legend(self, ax:int = -1, loc = 'upper right' , fontsize=8, markerscale=2):
        if ax == -1:
            for axi in self.axs:
                axi.legend(loc=loc, fontsize=fontsize, markerscale=markerscale)
        else:
            # check if ax is a list
            self.axs[ax].legend(loc=loc, fontsize=fontsize, markerscale=markerscale)

    def plot(self, geomobject,  ax:int|tuple|str=0, fill_figure=True,show_area = False, **kwargs):
        name = kwargs.pop('name', None)
        if 'color' in kwargs:
            if isinstance(kwargs['color'],int):
                kwargs['color'] = self.get_color(kwargs['color'])
        # set defaults of full_figure and show_area to False
        elif geomobject.__class__.__name__.endswith("MultiLineString"):
            for line in geomobject.geoms:
                x, y = line.xy
                self.axs[ax].plot(x, y, **kwargs)
        elif isinstance(geomobject, Point) or geomobject.__class__.__name__.endswith("AsmoPoint"):
            self.axs[ax].plot(geomobject[0], geomobject[1], 'o', **kwargs)
            # label not working l='name'
            if name:
                self.axs[ax].text(geomobject[0]*1.0, geomobject[1]*1.0, s= name, ha='left', va='bottom', color = 'black')
        elif isinstance(geomobject, Point):
            self.axs[ax].plot(geomobject.x, geomobject.y, 'o', label='Point', **kwargs)
        elif geomobject.__class__.__name__.endswith("Rectangle") or geomobject.__class__.__name__.endswith("Bound"):
            self.plot(geomobject.geom, ax, fill_figure,show_area,name=name, **kwargs)
            centroid = geomobject.geom.centroid
            # if name:
                # self.axs[ax].text(centroid.x, centroid.y, s=name, ha='left', va='bottom', color = 'black')
        # elif isinstance(geomobject, Polygon):
            # x, y = geomobject.exterior.xy
            # self.axs[ax].plot(x, y, label='Polygon', **kwargs)
        # if multipolygon
        elif geomobject.__class__.__name__.endswith("MultiPolygon"):
            for polygon in geomobject.geoms:
                x, y = polygon.exterior.xy
                self.plot(polygon, ax, fill_figure=fill_figure, **kwargs)
                # self.axs[ax].plot(x, y, label='Polygon', **kwargs)
        elif isinstance(geomobject, Polygon) or geomobject.__class__.__name__.endswith("Polygon"):
            x, y = geomobject.exterior.xy
            # self.axs[ax].plot(x, y, **kwargs)

            axis = self.axs[ax] if isinstance(ax, int) else ax

            # Hatch the area
            if fill_figure:
                color = kwargs.get('color', 'black')
                hatch = kwargs.pop('hatch', 'xxx')
                if 'color' in kwargs:
                    del kwargs['color']
                axis.fill(x, y,label = name, hatch=hatch, edgecolor = color, facecolor='none', **kwargs)
            
            # Calculate the centroid for placing the label
            centroid = geomobject.centroid
            area = geomobject.area
            
            # Add the label with the area
            if show_area:
                axis.text(centroid.x, centroid.y, f'Area: {area:.2f}', ha='center', va='center', color = 'black')
        # plot multipolygon
        elif geomobject.__class__.__name__.endswith("MultiPolygon") or geomobject.__class__.__name__.endswith("GeometryCollection"):
            for polygon in geomobject.geoms:
                # the fill_figure is not passed correctly to the function call
                self.plot(polygon, ax, fill_figure= fill_figure, **kwargs)
        elif geomobject.__class__.__name__.endswith("SearchArea"):
            self.plot(geomobject.geom, ax,name=name, fill_figure=fill_figure, **kwargs)
        elif geomobject.__class__.__name__.endswith("Line"):
            x, y = geomobject.geom.xy
            self.axs[ax].plot(x, y, label=name, **kwargs)
            if name:
                self.axs[ax].text(geomobject.geom.centroid.x, geomobject.geom.centroid.y, s=name, ha='left', va='bottom', color = 'black')
        elif geomobject.__class__.__name__.endswith("Solution"):
            self.axs[ax].plot(geomobject.val[0], geomobject.val[1], 'o', label=f'_Solution: {geomobject.classification}', **kwargs)
        elif geomobject.__class__.__name__.endswith("SolutionList"):
            # if empty return
            if len(geomobject) == 0:
                return
            # plot with a single call to ax.plot, and add name if supplied
            all_xy = [solution.val for solution in geomobject]
            if name:
                self.axs[ax].scatter(*zip(*all_xy), label =name, **kwargs)
            else:
                self.axs[ax].scatter(*zip(*all_xy), **kwargs)
            # for solution in geomobject:
                # self.plot(solution, ax, **kwargs)
        elif geomobject.__class__.__name__.endswith("PointList"):
            geomobject.plot(ax=ax, **kwargs)
        elif geomobject.__class__.__name__.endswith("Rectangle"):
            geomobject.plot(ax=ax, **kwargs)
        else:
            print(f"Unsupported geometry type: {type(geomobject)}")
            raise TypeError("Unsupported geometry type")

    def show(self):
        # for ax in self.axs.flat:
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.legend()
        self.fig.show()

    def zoom(self, geomobject,ax=0, buffer = 10, buffer_relative = False):
        ''' Set the axis limits to the bounding box of the geometry '''
        # buffer is padding
        # should set axis limits to the bounding box of the geometry
        bounding_box = shapely.envelope(geomobject)

        if buffer_relative: # use relative where buffer is in percentage - different for each axis
            buffer = buffer/100
            self.axs[ax].set_xlim([bounding_box.bounds[0] - buffer*bounding_box.bounds[0], bounding_box.bounds[2] + buffer*bounding_box.bounds[2]])
            self.axs[ax].set_ylim([bounding_box.bounds[1] - buffer*bounding_box.bounds[1], bounding_box.bounds[3] + buffer*bounding_box.bounds[3]])
        else:
            self.axs[ax].set_xlim([bounding_box.bounds[0] - buffer, bounding_box.bounds[2] + buffer])
            # self.axs[ax].set_xlim([bounding_box.bounds[0] , bounding_box.bounds[2]])
            self.axs[ax].set_ylim([bounding_box.bounds[1] - buffer, bounding_box.bounds[3] + buffer])
            # self.axs[ax].set_ylim([bounding_box.bounds[1], bounding_box.bounds[3]])

    def get_color(self,i:int):
        # get a color from a discrete set of colors using the index
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        return colors[i]

    def get_colormap(self, cmapname:str):
        # get a colormap from a discrete set of colors using the index
        cmap = mpl.colormaps[cmapname]
        return cmap
    def set_size(self, w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=self.fig.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
              


    def save(self, filename = None, as_multiple_plots = False):
        if filename:
            self.filename = filename
        self.fig.savefig(self.plot_dir + self.filename)
