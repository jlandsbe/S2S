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
import matplotlib.colors as clr
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


def summarize_skill_score(metrics_dict, error_type):
    marker_size = 15
    alpha = .8

    x_plot = metrics.eval_function(metrics_dict["error_climo"])
    x_climatology_baseline = x_plot.copy()
    plt.axhline(y=0, linewidth=1, linestyle='-', color="k", alpha=alpha)

    if error_type == "field":
        plt.plot(metrics_dict["analogue_vector"], 1-(np.repeat(np.array([x_plot]), len(metrics_dict["analogue_vector"]))), '.-', markersize=marker_size, label='climatology',
             color="black", alpha=alpha)
        
    x_plot = metrics.eval_function(metrics_dict["error_network"])
    x_plot = 1. - skill_score_helper(x_plot, x_climatology_baseline, error_type)
    plt.plot(metrics_dict["analogue_vector"], x_plot, '.-', markersize=marker_size, label='masked analog',
             color="orange", alpha=alpha)
    
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
    
    #x_plot = metrics.eval_function(metrics_dict["error_persist"])

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

    plt.ylabel('skill score')
    plt.xlabel('number of analogues averaged')
    plt.xlim(0, np.max(metrics_dict["analogue_vector"])*1.01)

    plt.ylim(-0.5, 1.0)
    plt.grid(False)
    plt.legend(fontsize=8)
    if error_type == "field":
        plt.title('Anomaly Correlation Coefficient')
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
                      cbarBool=True):
#here weights_train is of shape lat x lon
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


def uncertainty_plots(analogue_vector, error_network, analog_match_error, prediction_spread, settings):
    for i in range(0,len(analogue_vector)):
        #plot the analog match error vs error in prediction

        x_data = error_network[:, i]
        y_data = analog_match_error[:, i]

        # Compute the line of best fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

        # Compute the p-value for the slope
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        # Compute the 99% confidence interval for the slope
        alpha = 0.01  # Significance level (1 - confidence level)
        t_value = 2.626  # T-value for a two-tailed test with 48 degrees of freedom
        confidence_interval = t_value * std_err

        # Plot the scatter plot
        plt.figure()
        plt.scatter(x_data, y_data, alpha=0.5, label='Data Points')

        # Plot the line of best fit
        plt.plot(x_data, slope*x_data + intercept, color='red', label=f'Line of Best Fit (Slope={slope:.4f} \u00B1 {confidence_interval:.4f})')

        # Add labels and title
        plt.xlabel('RMSE Prediction Error')
        plt.ylabel('Analog Match Error')
        plt.title('Analog Match Error vs RMSE Prediction Error(' + str(analogue_vector[i]) + " analogs)")

        # Display the plot
        plt.legend()
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        '_' + "goodness_of_match_" + str(analogue_vector[i]) + '.png', dpi=dpiFig, bbox_inches='tight')

        #plot the spread of the predictions vs the error in the network

        x_data = error_network[:, i]
        y_data = prediction_spread[:, i]

        # Compute the line of best fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

        # Compute the p-value for the slope
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        # Compute the 99% confidence interval for the slope
        alpha = 0.01  # Significance level (1 - confidence level)
        t_value = 2.626  # T-value for a two-tailed test with 48 degrees of freedom
        confidence_interval = t_value * std_err
        plt.figure()
        # Plot the scatter plot
        plt.scatter(x_data, y_data, alpha=0.5, label='Data Points')

        # Plot the line of best fit
        plt.plot(x_data, slope*x_data + intercept, color='red', label=f'Line of Best Fit (Slope={slope:.4f} \u00B1 {confidence_interval:.4f})')

        # Add labels and title
        plt.xlabel('RMSE Prediction Error')
        plt.ylabel('Variance of Predictions')
        plt.title('Variance of Predictions vs RMSE Prediction Error(' + str(analogue_vector[i]) + " analogs)")

        # Display the plot
        plt.legend()
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        '_' + "prediction_spread_" + str(analogue_vector[i]) + '.png', dpi=dpiFig, bbox_inches='tight')

        #weighted plot of prediction spread vs network error


        x_data = error_network[:, i]
        y_data = prediction_spread[:, i]
        z = analog_match_error[:, i]
        z = (z -np.mean(z))/np.std(z)
        z = (z - np.min(z)) / (np.max(z) - np.min(z))
        # Compute the line of best fit
        slope, intercept = np.polyfit(x_data, y_data, 1)

        # Compute the p-value for the slope
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        # Compute the 99% confidence interval for the slope
        alpha = 0.01  # Significance level (1 - confidence level)
        t_value = 2.626  # T-value for a two-tailed test with 48 degrees of freedom
        confidence_interval = t_value * std_err
        plt.figure()
        # Plot the scatter plot
        plt.scatter(x_data, y_data, alpha=.5, label='Data Points', c = z, cmap = 'RdYlGn')

        # Plot the line of best fit
        plt.plot(x_data, slope*x_data + intercept, color='red', label=f'Line of Best Fit (Slope={slope:.4f} \u00B1 {confidence_interval:.4f})')

        # Add labels and title
        plt.xlabel('RMSE Prediction Error')
        plt.ylabel('Variance of Predictions')
        plt.title('Variance of Predictions vs Prediction Error - Color = Analog Match Error (' + str(analogue_vector[i]) + " analogs)")

        # Display the plot
        plt.legend()
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        '_' + "prediction_spread_weighted" + str(analogue_vector[i]) + '.png', dpi=dpiFig, bbox_inches='tight')


def uncertainty_whiskers(analogue_vector, error_network, analog_match_error, prediction_spread, settings, baseline_error=None, baseline_analog_match=None,
                          baseline_spread=None, random_error = None, random_spread = None,bins1=[0], bins2=[0]):
    plt.style.use("default")
    for analog_idx in range(len(analogue_vector)-1):
        # # Plot the analog match error vs error in prediction
        # bins_an = list(bins1[analog_idx+1])  # Create a copy of bins to add max value
        # y_data = error_network[:, analog_idx+1]
        # x_data = analog_match_error[:, analog_idx+1]
        # bins_an.append(np.max(x_data)+.000001)
        # filtered_data_list = []
        # mins = np.zeros((len(bins_an) - 1))
        # maxs = np.zeros((len(bins_an) - 1))
        
        # for i in range(len(bins_an) - 1):
        #     bin_min = bins_an[i]
        #     bin_max = bins_an[i + 1]
        #     filtered_data = y_data[(x_data >= bin_min) & (x_data < bin_max)]
        #     mins[i] = np.min(filtered_data)
        #     maxs[i] = np.max(filtered_data)
        #     filtered_data_list.append(filtered_data)
        
        # x_positions = (np.array(bins_an)[1:] + np.array(bins_an)[:-1]) / 2
        # color_data = .5 * np.ones(len(x_positions))
        
        # norm = clr.Normalize(vmin=color_data.min(), vmax=color_data.max())
        # color_data = norm(color_data)
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.set_facecolor('white')
        # # Hide top and right spines
        # ax.grid(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # if len(filtered_data_list) > 1:
        #     widths = .5 * np.diff(bins_an)
        # else:
        #     widths = [.5]
        # # Plot each box plot
        # boxes = ax.boxplot(filtered_data_list, positions=x_positions, patch_artist=True, medianprops=dict(color="black"), widths=widths, labels=np.round(x_positions,1))
        # delta = (np.max(maxs) - np.min(mins))/10
        # tot_min = max(np.min(mins)-1.3*delta,0)
        # plt.ylim(tot_min, np.max(maxs)+1.3*delta)
        # ax.set_xlim(max([x_positions[0]-.6*widths[0],0]), x_positions[-1]+.502*widths[-1])  # Setting x-axis limits
        # ax.set_ylim(tot_min, np.max(maxs)+1.3*delta)  # Setting y-axis limits
        # ax.set_xticks(np.round(x_positions,1))
        
        #         # Set colors for the boxes based on color_data
        # for i, box in enumerate(boxes['boxes']):
        #     color_value = color_data[i]
        #     box.set_facecolor(cm.Blues(color_value))  # Using colormap 'viridis' to map color_value to a color

        # for i, data in enumerate(filtered_data_list):
        #     ax.text(x_positions[i], maxs[i]+delta/1.3, f'{len(data)}', ha='center', va='top', color='black', weight = "bold",
        #             path_effects=[pe.withStroke(linewidth=1, foreground="w")])
        # # Plot the scatter plot
        # plt.xlabel('Analog Match Error')
        # plt.ylabel('RMSE Prediction Error')
        # plt.title(f'Analog Match Error vs RMSE Prediction Error ({analogue_vector[analog_idx+1]} analogs)')
        
        # # Save the figure
        # plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        #             '_' + "goodness_of_match_" + str(analogue_vector[analog_idx+1]) + '.png', dpi=dpiFig, bbox_inches='tight')

        # plt.close(fig)
        #     #plot the spread of the predictions vs the error in the network
        

        #bins_an = list(bins2[analog_idx+1])  # Create a copy of bins to add max value
        y_data = error_network[:, analog_idx+1]
        x_data = prediction_spread[:, analog_idx+1]
        z = analog_match_error[:, analog_idx+1]
        baseline_exist = baseline_error is not None
        if baseline_exist:
            baseline_y_data = baseline_error[:, analog_idx+1]
            baseline_x_data = baseline_spread[:, analog_idx+1]
            baseline_z = baseline_analog_match[:, analog_idx+1]
            baseline_x_sorted_indices = np.argsort(baseline_x_data)
            baseline_y_sorted_by_x = baseline_y_data[baseline_x_sorted_indices]
            baseline_z_sorted_indices = np.argsort(baseline_z)
            baseline_y_sorted_by_z = baseline_y_data[baseline_z_sorted_indices]
            baseline_y_means_x = []
            baseline_y_means_z = []
        if random_error is not None:
            baseline_y_data = random_error[:, analog_idx+1]
            baseline_x_data = random_spread[:, analog_idx+1]
            random_sorted_indices = np.argsort(baseline_x_data)
            random_sorted = baseline_y_data[random_sorted_indices]
            random_y_means_x = []
        # z_low = np.round(np.min(z),2)
        # z_high = np.round(np.max(z),2)
        # z = (z - np.min(z))/(np.max(z)-np.min(z))
        # bins_an.append(np.max(x_data)+.000001)
        # filtered_data_list = []
        # color_data = np.zeros(len(bins_an) - 1)
        # mins = np.zeros((len(bins_an) - 1))
        # maxs = np.zeros((len(bins_an) - 1))
        # for i in range(len(bins_an) - 1):
        #     bin_min = bins_an[i]
        #     bin_max = bins_an[i + 1]
        #     filtered_data = y_data[(x_data >= bin_min) & (x_data < bin_max)]
        #     filtered_data_list.append(filtered_data)
        #     filtered_colors = z[(x_data >= bin_min) & (x_data < bin_max)]
        #     filtered_colors_mean = np.mean(filtered_colors)
        #     mins[i] = np.min(filtered_data)
        #     maxs[i] = np.max(filtered_data)
        #     color_data[i] = filtered_colors_mean
        
        # x_positions = (np.array(bins_an)[1:] + np.array(bins_an)[:-1]) / 2
        
        # norm = clr.Normalize(vmin=color_data.min(), vmax=color_data.max())
        # color_data = norm(color_data)
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # # Hide top and right spines
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # if len(filtered_data_list) > 1:
        #     widths = .5 * np.diff(bins_an)
        # else:
        #     widths = [.5]
        
        # # Plot each box plot
        
        # boxes = ax.boxplot(filtered_data_list, positions=x_positions, patch_artist=True, medianprops=dict(color="black"), widths=widths, labels=np.round(x_positions,1))
        # #boxes = ax.violinplot(filtered_data_list, positions=x_positions, widths=widths, showmeans=True)
        # ax.set_xticks(np.round(x_positions,1))
        
        #         # Set colors for the boxes based on color_data
        # for i, box in enumerate(boxes['boxes']):
        #     color_value = color_data[i]
        #     rgba_color = cm.PiYG_r(color_value)
        #     box.set_facecolor((rgba_color[0], rgba_color[1], rgba_color[2], 1.0))  # Set alpha 
            

        # delta = (np.max(maxs) - np.min(mins))/10
        # tot_min = max(np.min(mins)-1.3*delta,0)

        # plt.ylim(tot_min, np.max(maxs)+1.3*delta)
        # ax.set_xlim(max([x_positions[0]-.6*widths[0],0]), x_positions[-1]+.502*widths[-1])  # Setting x-axis limits
        # ax.set_ylim(tot_min, np.max(maxs)+1.3*delta)  # Setting y-axis limits
        # for i, data in enumerate(filtered_data_list):
        #     ax.text(x_positions[i], maxs[i]+delta/1.3, f'{len(data)}', ha='center', va='top', color='black', weight="bold", 
        #             path_effects=[pe.withStroke(linewidth=1, foreground="w")])
        # # Add labels and title
        # plt.xlabel('Variance of Predictions')
        # plt.ylabel('RMSE Prediction Error')
        # plt.title('Variance of Predictions vs Prediction Error (' + str(analogue_vector[analog_idx+1]) + " analogs)")

        # # Display the plot
        #     # Add colorbar
        # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm.PiYG_r), ax=ax)
        # vmin = cbar.vmin
        # vmax = cbar.vmax
        # cbar.set_ticks([vmin, vmax])  # Set the colorbar ticks
        # cbar.set_ticklabels([z_low, z_high])  # Set the tick labels
        # cbar.set_label('Analog Matching Error', rotation=270)
        # #cbar.ax.set_yticklabels(['Good Analog Match', 'Poor Analog Match'])  
        # # Add text annotations to colorbar
        # # cbar.ax.text(0, 1.05, 'Low Analog Match Error', horizontalalignment='left', verticalalignment='center')
        # # cbar.ax.text(1, 1.05, 'High Analog Match Error', horizontalalignment='right', verticalalignment='center')
        # plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
        # '_' + "prediction_spread_weighted" + str(analogue_vector[analog_idx+1]) + '.png', dpi=dpiFig, bbox_inches='tight')
        # plt.close(fig)
        # Sort x and y based on x values
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
            if baseline_exist:
                baseline_y_subset_x = baseline_y_sorted_by_x[:cutoff_index]
                baseline_y_means_x.append(np.mean(baseline_y_subset_x))
                baseline_y_subset_z = baseline_y_sorted_by_z[:cutoff_index]
                baseline_y_means_z.append(np.mean(baseline_y_subset_z))
            if random_error is not None:
                random_y_subset_x = random_sorted[:cutoff_index]
                random_y_means_x.append(np.mean(random_y_subset_x))
            #y_subset_xz = y_sorted_by_xz[:cutoff_index]
            #y_means_xz.append(np.mean(y_subset_xz))
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(percentages, y_means_z, linestyle='-', linewidth = 7, color='forestgreen', label='Analog Matching Error')
        plt.plot(percentages, y_means_x, linestyle='-', linewidth = 7, color='cornflowerblue', label='Prediction Spread')
        if baseline_exist:
            plt.plot(percentages, baseline_y_means_z, linestyle='dashed', linewidth = 5, color='forestgreen', label='Baseline Analog Matching Error')
            plt.plot(percentages, baseline_y_means_x, linestyle='dashed', linewidth = 5, color='cornflowerblue', label='Baseline Prediction Spread')
        if random_error is not None:
            plt.plot(percentages, random_y_means_x, linestyle='dotted', linewidth = 5, color='coral', label='Random Prediction Spread')
        #plt.plot(percentages, y_means_xz, linestyle='-', linewidth = 7, color='gold', label='Analog Match Error x Prediction Spread')
        # Increase the font size of the labels and title
        plt.xlabel('Percent Most Confident', fontsize=14)
        plt.ylabel('Error in Prediction', fontsize=14)
        plt.title('Discard Plot for Analog Matching Error and Prediction Spread (' + str(analogue_vector[analog_idx+1]) + " analogs)", fontsize=16)
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
