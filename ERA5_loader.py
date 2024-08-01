save_directory = "/Users/jlandsbe/Downloads/WeightedMaskAnalogForecasting-main/ERA5/Z500"
save_directory = "scratch/jlandsbe/WeightedMaskAnalogForecasting-main/ERA5/Z500"
import cdsapi
import calendar
import os
def generate_days_list():
  days_list = []
  for month in range(1, 13):
    # Get the number of days in the month
    num_days = calendar.monthrange(2023, month)[1]
    # Add days to the list for the current month
    days_list.extend(f"{day:02d}" for day in range(1, num_days + 1))
  return days_list
days_list = generate_days_list()
c = cdsapi.Client(key='328224:81451bca-e5dd-42e7-b7ae-1207223a23ec', url='https://cds.climate.copernicus.eu/api/v2')
MONTHS = [
 "01", "02", "03", "04", "05", "06",
 "07", "08", "09", "10", "11", "12"
 ]
YEARS = [str(year) for year in range(1942, 2024)]
days_list = generate_days_list()
DAYS = days_list
for year in YEARS:
  for month in MONTHS:
    month_int = int(month) # Convert month to integer
    year_int = int(year)
    num_days = calendar.monthrange(year_int, month_int)[1]
    days_in_month = [f"{day:02d}" for day in range(1, num_days + 1)]
    for day in days_in_month:
      print(f"{year}, {month}, {day}")
      filename = os.path.join(save_directory, f"ERA5_{year}{month}{day}_500geopotential.nc")
      filename = save_directory
      result = c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type": "reanalysis",
        "variable": "geopotential",
        "pressure_level": "500",
        "year": year,
        "month": month,
        "day": day,
        'time': [
          '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
          '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
          '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
          '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        "grid": [2.5,2.5],
        "area": "global",
        "format": "netcdf"
      }, filename)
c.download(result)
#Script for generating the days of the month:
