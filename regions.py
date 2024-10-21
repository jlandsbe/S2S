"""Region definitions.

Functions
---------

"""


def get_region_dict(region=None):
    regions_dict = {

        "globe": {
            "lat_range": [-90, 90],
            "lon_range": [0, 360],
        },

        "northern_hemisphere": {
            "lat_range": [0, 90],
            "lon_range": [0, 360],
        },

        "southern_hemisphere": {
            "lat_range": [-90, 0],
            "lon_range": [0, 360],
        },

        "north_atlantic": {
            "lat_range": [40, 60],
            "lon_range": [360 - 70, 360 - 10],
        },

        "eastern_europe": {
            "lat_range": [40, 60],
            "lon_range": [0, 30],
        },

        "western_us": {
            "lat_range": [31, 49],
            "lon_range": [360-125, 360-110],
        },
        "california": {
            "lat_range": [32, 37],
            "lon_range": [360-121, 360-116],
        },


        "india": {
            "lat_range": [10, 30],
            "lon_range": [70, 85],
        },

        "north pdo": {
            "lat_range": [48., 60.],
            "lon_range": [360-165, 360-124],
        },

        "nino34" :{
            "lat_range":[-5., 5.],
            "lon_range":[360-170, 360-120],
        },

        "indopac" :{
            "lat_range":[-30., 30.],
            "lon_range":[30, 360-80],
        },

        "sams" :{
            "lat_range":[-20., -5.],
            "lon_range":[300., 320.],

        },

        "southern europe" :{
            "lat_range":[29., 49.],
            "lon_range":[16., 36.],
        },

        "n_atlantic" :{
            "lat_range":[30., 60.],
            "lon_range":[360.-70., 360],
        },
        "n_pacific" :{
            "lat_range":[0., 65.],
            "lon_range":[360.-205., 360-125],
        },

        "trop_indopac" :{
            "lat_range":[-23.5, 23.5],
            "lon_range":[30, 360-80],
        },
         "alaska" :{
            "lat_range":[57, 70],
            "lon_range":[360-166, 360-135],
        },

        "australlia_oc":{
            "lat_range":[-60,-50],
            "lon_range": [200,220]
        },

        "trop_pac_precip" :{
            "lat_range":[-5., 5.],
            "lon_range":[170., 200.],
        },

        "mjo" :{
            "lat_range":[-15., 15.],
            "lon_range":[0., 360.],
          },        

          "tristate" :{
              
            "lat_range":[39., 45.],
            "lon_range":[360-80.5, 360-72.],
          },

          "midwest" :{          
                "lat_range":[36., 49.],
                "lon_range":[254., 270.],
             },

            "cont_us" :{          
                "lat_range":[25., 48.],
                "lon_range":[360.-124., 360.-70.],
             },
            "n_atlantic_ext" :{
                "lat_range":[25., 65.],
                "lon_range":[360.-82., 360],
        },
          }

    if region is None:
        return regions_dict
    else:
        return regions_dict[region]
