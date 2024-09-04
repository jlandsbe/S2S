"""Metrics for generic plotting.

Functions
---------
plot_metrics(history,metric)
plot_metrics_panels(history, settings)
plot_map(x, clim=None, title=None, text=None, cmap='RdGy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
import palettable
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import metrics
from shapely.errors import ShapelyDeprecationWarning
import warnings
import regions
import base_directories
import os
import matplotlib.colors as mcol
from scipy.stats import linregress
import matplotlib.patheffects as pe

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()
if dir_settings["data_directory"].split("/")[1] != "Users":
    ct.config["data_dir"] = "/scratch/jlandsbe/cartopy_maps"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150
plt.style.use("seaborn-v0_8")
dpiFig = 300


def get_mycolormap():
    import matplotlib.colors as clr
    import palettable

    cmap = palettable.scientific.sequential.get_map("Buda_20",).colors
    del cmap[0:1]
    del cmap[-4:]

    cmap.append((256*.85, 256*.8, 256*.65))
    cmap.append((256*.9, 256*.85, 256*.8))
    cmap.append((256*.9, 256*.9, 256*.9))
    cmap.append((256, 256, 256))

    cmap = cmap[::-1]
    cmap = np.divide(cmap, 256.)
    return clr.LinearSegmentedColormap.from_list('custom', cmap, N=256)


def savefig(filename, dpi=300):
    for fig_format in (".png", ".pdf"):
        plt.savefig(filename + fig_format,
                    bbox_inches="tight",
                    dpi=dpi)


def get_qual_cmap():
    cmap = palettable.colorbrewer.qualitative.Accent_7.mpl_colormap
    cmap = ListedColormap(cmap(np.linspace(0, 1, 11)))
    cmap2 = cmap.colors
    cmap2[6, :] = cmap.colors[0, :]
    cmap2[2:6, :] = cmap.colors[5:1:-1, :]
    cmap2[1, :] = (.95, .95, .95, 1)
    cmap2[0, :] = (1, 1, 1, 1)
    cmap2[5, :] = cmap2[6, :]
    cmap2[6, :] = [0.7945098, 0.49647059, 0.77019608, 1.]
    cmap2 = np.append(cmap2, [[.2, .2, .2, 1]], axis=0)
    cmap2 = np.delete(cmap2, 0, 0)

    return ListedColormap(cmap2)


def plot_targets(target_train, target_val):
    plt.figure(figsize=(10, 2.5), dpi=125)
    plt.subplot(1, 2, 1)
    plt.hist(target_train, np.arange(0, 8, .1))
    plt.title('Training targets')
    plt.subplot(1, 2, 2)
    plt.hist(target_val, np.arange(0, 8, .1))
    plt.title('Validation targets')
    plt.show()


def drawOnGlobe(ax, map_proj, data, lats, lons, cmap='coolwarm', vmin=None, vmax=None, inc=None, cbarBool=True,
                contourMap=[], contourVals=[], fastBool=False, extent='both', alpha=1., landfacecolor="None"):

    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons)  # fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    # data_cyc = data
    # lons_cyc = lons

    #     ax.set_global()
    #     ax.coastlines(linewidth = 1.2, color='black')
    #     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')

    # ADD COASTLINES
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor=landfacecolor,
        edgecolor='k',
        linewidth=.5,
    )
    ax.add_feature(land_feature)

    # ADD COUNTRIES
    # country_feature = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='admin_0_countries',
    #     scale='50m',
    #     facecolor='None',
    #     edgecolor = 'k',
    #     linewidth=.25,
    #     alpha=.25,
    # )
    # ax.add_feature(country_feature)

    #     ax.GeoAxes.patch.set_facecolor('black')

    if fastBool:
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, alpha=alpha)
    #         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading='auto')

    if (np.size(contourMap) != 0):
        contourMap_cyc, __ = add_cyclic_point(contourMap, coord=lons)  # fixes white line by adding point
        ax.contour(lons_cyc, lats, contourMap_cyc, contourVals, transform=data_crs, colors='fuchsia')

    if cbarBool:
        cb = plt.colorbar(image, shrink=.45, orientation="horizontal", pad=.02, extend=extent)
        cb.ax.tick_params(labelsize=6)
    else:
        cb = None

    image.set_clim(vmin, vmax)

    return cb, image


def add_cyclic_point(data, coord=None, axis=-1):
    # had issues with cartopy finding utils so copied for myself

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both', length=4, width=2, which='major', color='dimgrey')
#     ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)


def summarize_errors(metrics_dict):
    marker_size = 15
    alpha = .8

    x_plot = metrics.eval_function(metrics_dict["error_climo"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '--', markersize=marker_size, label='climatology baseline',
             color="gray", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_network"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='network',
             color="orange", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_corr"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='corr. baseline',
             color="cornflowerblue", alpha=alpha/3.)

    x_plot = metrics.eval_function(metrics_dict["error_globalcorr"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='global corr. baseline',
             color="seagreen", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_random"])
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='random baseline',
             color="gray", alpha=alpha)

    plt.ylabel('MAE (K)')
    plt.xlabel('number of analogues averaged')
    plt.xlim(0, np.max(metrics_dict["analogue_vector"])*1.01)

    plt.ylim(.1, 1.)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title('MAE')


def summarize_skill_score(metrics_dict, settings):
    marker_size = 15
    alpha = .8
    error_type = settings["error_calc"]
    x_plot = metrics.eval_function(metrics_dict["error_climo"])
    x_climatology_baseline = x_plot.copy()
    plt.axhline(y=0, linewidth=1, linestyle='-', color="k", alpha=alpha)

    if error_type == "field":
        plt.plot(metrics_dict["analogue_vector"], 1-(np.repeat(np.array([x_plot]), len(metrics_dict["analogue_vector"]))), '.-', markersize=marker_size, label='climatology',
             color="black", alpha=alpha)
        

    
    x_plot = metrics.eval_function(metrics_dict["error_maxskill"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='max skill',
             color="yellowgreen", alpha=alpha)
    if not np.isnan(metrics_dict["error_customcorr"]).any(): 
        x_plot = metrics.eval_function(metrics_dict["error_customcorr"])
        x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
        plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='N hemisphere',
                color="palevioletred", alpha=alpha)
    
    x_plot = metrics.eval_function(metrics_dict["error_corr"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='region corr.',
             color="lightskyblue", alpha=alpha)
    

    x_plot = metrics_dict["error_persist"]
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='persistence',
             color="cornflowerblue", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_globalcorr"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='global corr.',
             color="mediumblue", alpha=alpha)

    x_plot = metrics.eval_function(metrics_dict["error_random"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='random',
             color="gray", alpha=alpha)
    
    x_plot = metrics.eval_function(metrics_dict["error_network"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='masked analog',
             color="orange", alpha=alpha)

    plt.ylabel('skill score')
    plt.xlabel('number of analogues averaged')
    plt.xlim(0, np.max(metrics_dict["analogue_vector"])*1.01)

    plt.ylim(-0.5, 1.0)
    plt.grid(False)
    plt.legend(fontsize=8)
    if error_type == "field":
        plt.title('Anomaly Correlation Coefficient')
    elif settings["percentiles"] != None:
        plt.title('Classification Accuracy')
    else:
        plt.title('MAE Skill Score')

def skill_score_helper(x, clima, error_type):
    if error_type == "field":
        return x
    else:
        return x/clima
def plot_state_masks(fig, settings, weights_train, lat, lon, region_bool=True, climits=None, central_longitude=215.,
                      title_text=None, subplot=(1, 1, 1), cmap=None, use_text=True, edgecolor="turquoise", 
                      cbarBool=True):
    if cmap is None:
        cmap = get_mycolormap()

    if settings["maskout_landocean_input"] == "ocean":
        landfacecolor = "k"
    else:
        landfacecolor = "None"

    for channel in [0, 1]:

        if climits is None:
            cmin = 0.  # np.min(weights_train[:])
            cmax = np.max(weights_train[:])
            climits = (cmin, cmax)

        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                             projection=ct.crs.PlateCarree(central_longitude=central_longitude))
        print(climits)
        drawOnGlobe(ax,
                    ct.crs.PlateCarree(),
                    weights_train,
                    lat,
                    lon,
                    fastBool=True,
                    vmin=climits[0],
                    vmax=climits[1],
                    cmap=cmap,
                    extent=None,
                    cbarBool=cbarBool,
                    landfacecolor=landfacecolor,
                    )
        if region_bool:
            reg = regions.get_region_dict(settings["target_region_name"])
            if reg == "NorthEast":
                highlight_states = ['Connecticut', 'Delaware', 'Maine', 'Maryland', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York', 'Pennsylvania', 'Rhode Island', 'Vermont']
                shpfilename = shpreader.natural_earth(resolution='110m',
                                           category='cultural',
                                           name='admin_1_states_provinces')
                reader = shpreader.Reader(shpfilename)

                # Create a list to store the geometries of the highlighted states
                highlighted_states_geometries = []

                # Iterate over the states and add the geometries of the highlighted states to the list
                for state in reader.records():
                    state_name = state.attributes['name']
                    if state_name in highlight_states:
                        highlighted_states_geometries.append(state.geometry)

                # Merge the geometries into a single geometry
                merged_geometry = cascaded_union(highlighted_states_geometries)

                # Add the boundary of the merged geometry to the map
                ax.add_geometries([merged_geometry], ccrs.PlateCarree(), facecolor='none', edgecolor=edgecolor, linewidth=3)
            else:
                rect = mpl.patches.Rectangle((reg["lon_range"][0], reg["lat_range"][0]),
                                         reg["lon_range"][1] - reg["lon_range"][0],
                                         reg["lat_range"][1] - reg["lat_range"][0],
                                         transform=ct.crs.PlateCarree(),
                                         facecolor='None',
                                         edgecolor=edgecolor,
                                         color=None,
                                         linewidth=2.5,
                                         zorder=200,
                                         )
            ax.add_patch(rect)
            if settings["target_region_name"] == "north pdo":
                rect = mpl.patches.Rectangle((150, -30),
                                             50,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=100,
                                             )
                # ax.add_patch(rect)
                rect = mpl.patches.Rectangle((125, 5),
                                             180-125,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=101,
                                             )
                # ax.add_patch(rect)

        plt.title(title_text)
        if use_text:
            plt.text(0.01, .02, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
                    + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
                    fontsize=6, color="gray", va="bottom", ha="left", fontfamily="monospace", backgroundcolor="white",
                    transform=ax.transAxes,
                    )

        return ax, climits
def plot_interp_masks(fig, settings, weights_train, lat, lon, region_bool=True, climits=None, central_longitude=215.,
                      title_text=None, subplot=(1, 1, 1), cmap=None, use_text=True, edgecolor="turquoise", 
                      cbarBool=True, style = "seaborn-v0_8"):
#here weights_train is of shape lat x lon
    plt.style.use(style)
    if cmap is None:
        cmap = get_mycolormap()

    if settings["maskout_landocean_input"] == "ocean":
        landfacecolor = "k"
    else:
        landfacecolor = "None"

    for channel in [0, 1]:

        if climits is None:
            cmin = 0.  # np.min(weights_train[:])
            cmax = np.max(weights_train[:])
            climits = (cmin, cmax)

        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                             projection=ct.crs.PlateCarree(central_longitude=central_longitude))

        drawOnGlobe(ax,
                    ct.crs.PlateCarree(),
                    weights_train,
                    lat,
                    lon,
                    fastBool=True,
                    vmin=climits[0],
                    vmax=climits[1],
                    cmap=cmap,
                    extent=None,
                    cbarBool=cbarBool,
                    landfacecolor=landfacecolor,
                    )
        if region_bool:
            reg = regions.get_region_dict(settings["target_region_name"])
            rect = mpl.patches.Rectangle((reg["lon_range"][0], reg["lat_range"][0]),
                                         reg["lon_range"][1] - reg["lon_range"][0],
                                         reg["lat_range"][1] - reg["lat_range"][0],
                                         transform=ct.crs.PlateCarree(),
                                         facecolor='None',
                                         edgecolor=edgecolor,
                                         color=None,
                                         linewidth=2.5,
                                         zorder=200,
                                         )
            ax.add_patch(rect)
            if settings["target_region_name"] == "north pdo":
                rect = mpl.patches.Rectangle((150, -30),
                                             50,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=100,
                                             )
                # ax.add_patch(rect)
                rect = mpl.patches.Rectangle((125, 5),
                                             180-125,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=101,
                                             )
                # ax.add_patch(rect)

        plt.title(title_text)
        if use_text:
            plt.text(0.01, .02, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
                    + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
                    fontsize=6, color="gray", va="bottom", ha="left", fontfamily="monospace", backgroundcolor="white",
                    transform=ax.transAxes,
                    )

        return ax, climits
def JBL_maps_plot(fig, settings, weights_train, lat, lon, region_bool=True, climits=None, central_longitude=215.,
                      title_text=None, subplot=(1, 1, 1), cmap=None, use_text=True, edgecolor="turquoise", 
                      cbarBool=True):
#here weights_train is of shape lat x lon
    cmap = mpl.colormaps['PiYG']
    cmap.set_bad(color="grey")
    #cmap.set_under(color='white')
    if settings["maskout_landocean_input"] == "ocean":
        landfacecolor = "k"
    else:
        landfacecolor = "None"
    if cmap is None:
        cmap = get_mycolormap()
    avg_skill = round(np.nanmean(weights_train),2)
    area_skill = round((np.sum(weights_train>0)/np.sum(np.isfinite(weights_train)))*100,2)
    for channel in [0, 1]:

        if climits is None:
            cmin = 0.  # np.min(weights_train[:])
            cmax = np.max(weights_train[:])
            climits = (cmin, cmax)

        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                             projection=ct.crs.PlateCarree(central_longitude=central_longitude))

        drawOnGlobe(ax,
                    ct.crs.PlateCarree(),
                    weights_train,
                    lat,
                    lon,
                    fastBool=True,
                    vmin=climits[0],
                    vmax=climits[1],
                    cmap=cmap,
                    extent=None,
                    cbarBool=cbarBool,
                    landfacecolor=landfacecolor,
                    )
        if region_bool:
            reg = regions.get_region_dict(settings["target_region_name"])
            rect = mpl.patches.Rectangle((reg["lon_range"][0], reg["lat_range"][0]),
                                         reg["lon_range"][1] - reg["lon_range"][0],
                                         reg["lat_range"][1] - reg["lat_range"][0],
                                         transform=ct.crs.PlateCarree(),
                                         facecolor='None',
                                         edgecolor=edgecolor,
                                         color=None,
                                         linewidth=2.5,
                                         zorder=200,
                                         )
            ax.add_patch(rect)
            if settings["target_region_name"] == "north pdo":
                rect = mpl.patches.Rectangle((150, -30),
                                             50,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=100,
                                             )
                # ax.add_patch(rect)
                rect = mpl.patches.Rectangle((125, 5),
                                             180-125,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=101,
                                             )
                # ax.add_patch(rect)

        plt.title(title_text)
        if use_text:
            plt.text(0.01, .02, ' ' + settings["savename_prefix"] + '\n avg skill: ' + str(avg_skill) + ' skill area: ' + str(area_skill) +"%",
                    fontsize=6, color="gray", va="bottom", ha="left", fontfamily="monospace", backgroundcolor="white",
                    transform=ax.transAxes,
                    )

        return ax, climits

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap  
def JBL_plot_interp_masks(fig, settings, weights_train, lat, lon, region_bool=True, climits=None, central_longitude=215.,
                      title_text="", subplot=(2, 2, 1), cmap=None, use_text=0, edgecolor="turquoise", 
                      cbarBool=True):
#here weights_train is of shape lat x lon
    # if cmap is None:
    #     cmap = get_mycolormap()
    # elif cmap == "test":

    #     #orig_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    #     orig_cmap = mpl.colormaps["coolwarm_r"]
    #     if climits !=None:
    #         mp = climits[1]/(climits[1] - climits[0])
    #     else:
    #         mp = .5
    #     shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mp, name='shifted')
    #     cmap = shifted_cmap
    cmap = mpl.colormaps['coolwarm_r']
    if len(np.argwhere(np.isnan(weights_train)))>0:
        cmap.set_bad(color="lightgreen")
    if settings["maskout_landocean_input"] == "ocean":
        landfacecolor = "k"
        #landfacecolor = None
    else:
        landfacecolor = None

    for channel in [0, 1]:

        if climits is None:
            cmin = 0.  # np.min(weights_train[:])
            cmax = np.percentile(weights_train[:], 90)
            climits = (cmin, cmax)
        else:
            cmin = climits[0]
            cmax = climits[1]
            maxmax = max(abs(cmin),abs(cmax))
            climits = (-1*maxmax,maxmax)

        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                             projection=ct.crs.PlateCarree(central_longitude=central_longitude))

        drawOnGlobe(ax,
                    ct.crs.PlateCarree(),
                    weights_train,
                    lat,
                    lon,
                    fastBool=True,
                    vmin=climits[0],
                    vmax=climits[1],
                    cmap=cmap,
                    extent=None,
                    cbarBool=cbarBool,
                    landfacecolor=landfacecolor,
                    )
        if region_bool:
            reg = regions.get_region_dict(settings["target_region_name"])
            rect = mpl.patches.Rectangle((reg["lon_range"][0], reg["lat_range"][0]),
                                         reg["lon_range"][1] - reg["lon_range"][0],
                                         reg["lat_range"][1] - reg["lat_range"][0],
                                         transform=ct.crs.PlateCarree(),
                                         facecolor='None',
                                         edgecolor=edgecolor,
                                         color=None,
                                         linewidth=2.5,
                                         zorder=200,
                                         )
            ax.add_patch(rect)
            if settings["target_region_name"] == "north pdo":
                rect = mpl.patches.Rectangle((150, -30),
                                             50,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=100,
                                             )
                # ax.add_patch(rect)
                rect = mpl.patches.Rectangle((125, 5),
                                             180-125,
                                             25,
                                             transform=ct.crs.PlateCarree(),
                                             facecolor='None',
                                             edgecolor=edgecolor,
                                             linestyle="--",
                                             color=None,
                                             linewidth=2,
                                             zorder=101,
                                             )
                # ax.add_patch(rect)
        plt.title(title_text)
        if use_text:
            plt.text(0.01, .02, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
                    + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
                    fontsize=6, color="gray", va="bottom", ha="left", fontfamily="monospace", backgroundcolor="white",
                    transform=ax.transAxes,
                    )


        return ax, climits


def plot_history(settings, history):
    fontsize = 12
    colors = ("#7570b3", "#e7298a")

    best_epoch = np.argmin(history.history["val_loss"])

    plt.figure(figsize=(14, 10))
    # Plot the training and validations loss history.
    plt.subplot(2, 2, 1)
    plt.plot(
        history.history["loss"],
        "-o",
        color=colors[0],
        markersize=3,
        linewidth=1,
        label="training",
    )

    plt.plot(
        history.history["val_loss"],
        "-o",
        color=colors[1],
        markersize=3,
        linewidth=1,
        label="validation",
    )
    try:
        ymin = 0.97*np.min([history.history["val_loss"], history.history["loss"]])
        ymax = 1.025*np.max([history.history["val_loss"][5], history.history["loss"][25]])
    except:
        ymin = 0.
        ymax = 0.05

    plt.ylim(ymin, ymax)
    plt.yscale("log")
    plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
    plt.title("loss during training")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend(frameon=True, fontsize=fontsize)
    plt.tight_layout()
    print(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                '_training_history.png')
    plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                '_training_history.png', dpi=dpiFig, bbox_inches='tight')
    plt.close()



def uncertainty_whiskers(analogue_vector, network_error, analog_match_error, prediction_spread, settings, baseline_error=[], baseline_analog_match=[],
                          baseline_spread=[], random_error = None, random_spread = None,bins1=[0], bins2=[0]):
    plt.style.use("default")
    for analog_idx in range(len(analogue_vector)-1):
        y_data = network_error[:, analog_idx+1]
        x_data = prediction_spread[:, analog_idx+1]
        z = analog_match_error[:, analog_idx+1]
        if len(baseline_error)>0:
            NH_y_data = baseline_error[0][:, analog_idx+1]
            NH_x_data = baseline_spread[0][:, analog_idx+1]
            NH_z = baseline_analog_match[0][:, analog_idx+1]
            NH_x_sorted_indices = np.argsort(NH_x_data)
            NH_y_sorted_by_x = NH_y_data[NH_x_sorted_indices]
            NH_z_sorted_indices = np.argsort(NH_z)
            NH_y_sorted_by_z = NH_y_data[NH_z_sorted_indices]
            NH_y_means_x = []
            NH_y_means_z = []
            if len(baseline_error)>1:
                global_y_data = baseline_error[1][:, analog_idx+1]
                global_x_data = baseline_spread[1][:, analog_idx+1]
                global_z = baseline_analog_match[1][:, analog_idx+1]
                global_x_sorted_indices = np.argsort(global_x_data)
                global_y_sorted_by_x = global_y_data[global_x_sorted_indices]
                global_z_sorted_indices = np.argsort(global_z)
                global_y_sorted_by_z = global_y_data[global_z_sorted_indices]
                global_y_means_x = []
                global_y_means_z = []
        if random_error is not None:
            baseline_y_data = random_error[:, analog_idx+1]
            baseline_x_data = random_spread[:, analog_idx+1]
            random_sorted_indices = np.argsort(baseline_x_data)
            random_sorted = baseline_y_data[random_sorted_indices]
            random_y_means_x = []
        x_sorted_indices = np.argsort(x_data)
        y_sorted_by_x = y_data[x_sorted_indices]
        z_sorted_indices = np.argsort(z)
        y_sorted_by_z = y_data[z_sorted_indices]
        xz_sorted_indices = np.argsort(x_data * z)
        y_sorted_by_xz = y_data[xz_sorted_indices]
        # Percentages to calculate means for
        percentages = np.arange(100, 4, -1)

        # Lists to store the means
        y_means_x = []
        y_means_z = []
        #y_means_xz = []

        # Calculate means for each percentage
        for p in percentages:
            cutoff_index = int(len(x_data) * (p / 100))
            y_subset_x = y_sorted_by_x[:cutoff_index]
            y_means_x.append(np.mean(y_subset_x))
            y_subset_z = y_sorted_by_z[:cutoff_index]
            y_means_z.append(np.mean(y_subset_z))
            if len(baseline_error)>0:
                NH_y_subset_x = NH_y_sorted_by_x[:cutoff_index]
                NH_y_means_x.append(np.mean(NH_y_subset_x))
                NH_y_subset_z = NH_y_sorted_by_z[:cutoff_index]
                NH_y_means_z.append(np.mean(NH_y_subset_z))
                if len(baseline_error)>1:
                    global_y_subset_x = global_y_sorted_by_x[:cutoff_index]
                    global_y_means_x.append(np.mean(global_y_subset_x))
                    global_y_subset_z = global_y_sorted_by_z[:cutoff_index]
                    global_y_means_z.append(np.mean(global_y_subset_z))
            if random_error is not None:
                random_y_subset_x = random_sorted[:cutoff_index]
                random_y_means_x.append(np.mean(random_y_subset_x))
            #y_subset_xz = y_sorted_by_xz[:cutoff_index]
            #y_means_xz.append(np.mean(y_subset_xz))
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(percentages, 100*(1-np.array(y_means_z)), linestyle='-', linewidth = 7, color='forestgreen', label='Analog Matching Error')
        plt.plot(percentages, 100*(1-np.array(y_means_x)), linestyle='-', linewidth = 7, color='deepskyblue', label='Prediction Spread')
        if len(baseline_error)>0:
            #plt.plot(percentages, baseline_y_means_z, linestyle='dashed', linewidth = 5, color='forestgreen', label='NH Analog Matching Error')
            plt.plot(percentages, 100*(1-np.array(NH_y_means_x)), linestyle='dashed', linewidth = 5, color='skyblue', label='NH Prediction Spread')
            if len(baseline_error)>1:
                plt.plot(percentages, 100*(1-np.array(global_y_means_x)), linestyle='dotted', linewidth = 5, color='lightskyblue', label='Global Prediction Spread')
        if random_error is not None:
            plt.plot(percentages, 100*(1-np.array(random_y_means_x)), linestyle='dotted', linewidth = 5, color='coral', label='Random Prediction Spread')
        #plt.plot(percentages, y_means_xz, linestyle='-', linewidth = 7, color='gold', label='Analog Match Error x Prediction Spread')
        # Increase the font size of the labels and title
        plt.xlabel('Percent Most Confident', fontsize=14)
        plt.ylabel('Percent Accuracy', fontsize=14)
        plt.title('Discard Plot for Prediction Spread (' + str(analogue_vector[analog_idx+1]) + " analogs)", fontsize=16)
        ax.legend()

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        plt.gca().invert_xaxis()
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        '_' + "discard_plot" + str(analogue_vector[analog_idx+1]) + '.png', dpi=dpiFig, bbox_inches='tight')
        plt.close(fig)
        # Show the plot


def get_shades(base_color, num_shades):
    base = np.array(colors.to_rgb(base_color))
    return [colors.to_hex(base * (1 - i / num_shades)) for i in range(num_shades)]

def confidence_plot(analogue_vector, error_dictionary, settings, error_climotol = None):
    plt.style.use("default")
    for analog_idx in range(1,len(analogue_vector)):
        fig, ax = plt.subplots(figsize=(12, 6))
        for mask_type_name, mask_type_values in error_dictionary.items():
            error = mask_type_values[0]
            confidence_dictionary = mask_type_values[1]
            line_type = mask_type_values[2]
            base_color = mask_type_values[3]
            num_shades = len(confidence_dictionary)
            shades = get_shades(base_color, num_shades)
            for i, (confidence_name, confidence_values) in enumerate(confidence_dictionary.items()):
                y_data = error[:, analog_idx]
                x_data = confidence_values[:, analog_idx]
                x_sorted_indices = np.argsort(x_data)
                if confidence_name == "Entropy" or confidence_name == "Modal Fraction":
                    x_sorted_indices = x_sorted_indices[::-1]
                y_sorted_by_x = y_data[x_sorted_indices]
                percentages = np.arange(100, 4, -1)
                y_means_x = []
                for p in percentages:
                    cutoff_index = int(len(x_data) * (p / 100))
                    y_subset_x = y_sorted_by_x[:cutoff_index]
                    y_means_x.append(np.mean(y_subset_x))
                label_name =  label_name = f"{mask_type_name}: {confidence_name}"
                if type(error_climotol) == type(None):
                    plt.plot(percentages, 100*(1-np.array(y_means_x)), linestyle=line_type, linewidth = 4, color=shades[i], label=label_name, alpha = .8)
                else:
                    plt.plot(percentages, 100*(1-np.array(y_means_x)/np.mean(np.array(error_climotol))), linestyle=line_type, linewidth = 4, color=shades[i], label=label_name, alpha = .8)
        plt.xlabel('Percent Most Confident', fontsize=14)
        plt.ylabel('Percent Accuracy', fontsize=14)
        plt.title('Discard Plot for Prediction Spread (' + str(analogue_vector[analog_idx]) + " analogs)", fontsize=16)
        ax.legend(loc="upper left")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.gca().invert_xaxis()
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        '_' + "discard_plot" + str(analogue_vector[analog_idx]) + '.png', dpi=dpiFig, bbox_inches='tight')
        plt.close(fig)

def yearly_analysis(soi_year_repeated,best_analog_years,year_length, settings):
    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(soi_year_repeated, best_analog_years, bins=[int(year_length/5), int(year_length/5)])
    
    # Normalize the histogram along each row
    row_sums = hist.sum(axis=1)
    # Avoid division by zero by replacing 0s in row_sums with 1
    row_sums[row_sums == 0] = 1
    row_normalized_hist = hist / row_sums[:, np.newaxis]

    # Normalize the histogram along each column
    column_sums = row_normalized_hist.sum(axis=0)
    # Avoid division by zero by replacing 0s in column_sums with 1
    column_sums[column_sums == 0] = 1
    normalized_hist = row_normalized_hist / column_sums[np.newaxis, :]
    
    # Plot the normalized 2D histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(normalized_hist.T, origin='lower', cmap='Reds', aspect='auto', extent=[0, year_length, 0, year_length])
    
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, label='Normalized Counts')
    
    # Set labels and title
    ax.set_xlabel('SOI Years')
    ax.set_ylabel('Best Analog Years')
    ax.set_title('SOI vs Best Analog Years')
    
    # Save the figure
    print(dir_settings["figure_diag_directory"] + settings["savename_prefix"] + '_' + "yearly_analysis" + '.png')
    fig.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] + '_' + "yearly_analysis" + '.png', dpi=dpiFig, bbox_inches='tight')
    plt.close(fig)


def monthly_analysis(soi_months_repeated,best_analog_months, settings):
    fig, ax = plt.subplots(figsize=(12, 6))
# Define the desired order of months dynamically
    unique_months = np.unique(np.concatenate([soi_months_repeated, best_analog_months]))
    sorted_months = np.sort(unique_months)
    gaps = np.diff(np.append(sorted_months, sorted_months[0] + 12))  # consider wrap-around gap
    largest_gap_index = np.argmax(gaps)
    start_month_index = (largest_gap_index + 1) % len(sorted_months)
    ordered_months = np.roll(sorted_months, -start_month_index)
    month_mapping = {month: i for i, month in enumerate(ordered_months)}

    # Remap the months based on the new order
    soi_months_mapped = np.array([month_mapping[month] for month in soi_months_repeated])
    best_analog_mapped = np.array([month_mapping[month] for month in best_analog_months])

    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(soi_months_mapped, best_analog_mapped, bins=[len(ordered_months), len(ordered_months)])

    # Normalize the histogram along each row
    row_sums = hist.sum(axis=1)
    # Avoid division by zero by replacing 0s in row_sums with 1
    row_sums[row_sums == 0] = 1
    row_normalized_hist = hist / row_sums[:, np.newaxis]

    # Normalize the histogram along each column
    column_sums = row_normalized_hist.sum(axis=0)
    # Avoid division by zero by replacing 0s in column_sums with 1
    column_sums[column_sums == 0] = 1
    normalized_hist = row_normalized_hist / column_sums[np.newaxis, :]

    # Plot the normalized 2D histogram
    plt.imshow(normalized_hist.T, origin='lower', cmap='Reds', aspect='auto', extent=[0, len(ordered_months), 0, len(ordered_months)])
    plt.colorbar(label='Normalized Counts')

    # Setting custom ticks
    plt.xticks(ticks=np.arange(len(ordered_months)) + 0.5, labels=ordered_months)
    plt.yticks(ticks=np.arange(len(ordered_months)) + 0.5, labels=ordered_months)

    plt.xlabel('SOI Months')
    plt.ylabel('Best Analog Months')
    plt.title('2D Histogram with Column-Normalized Counts')
    print(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
    '_' + "monthly_analysis" + '.png')
    plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
    '_' + "monthly_analysis" + '.png', dpi=dpiFig, bbox_inches='tight')
    plt.close(fig)