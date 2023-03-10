# Import packages
import xarray as xr
import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
## Load in the necessary python packages to train a neural network
import cartopy, cartopy.crs as ccrs 
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
z500 = xr.open_mfdataset('geopotential_500*.nc', combine='by_coords').isel(time=slice(None, None, 12))
#z500.load()
#z500.z.isel(time=0).plot()

# training dataset selection
train_years = slice('1979', '2015')
# validation dataset selection (this dataset helps with overfitting)
valid_years = slice('2016', '2016')
# test dataset selection
test_years = slice('2017', '2018')

def computed_weighted_rmse(fc, gt):
    """Error metric to compute the area averaged RMSE."""
    error = fc - gt
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(('time', 'lat', 'lon')))
    return rmse

# pick the forecast lead time
lead_time_steps = 10 # 5 day forecast because considering midday and midnight

# compute persistent forecast 
persistence_fc = z500.sel(time=test_years).isel(time=slice(0, -lead_time_steps))
persistence_fc['time'] = persistence_fc.time + np.timedelta64(5, 'D').astype('timedelta64[ns]')

# target data
target = z500.sel(time=test_years)['z']
# compute RMSE
rmse_data = computed_weighted_rmse(persistence_fc, target)
print(rmse_data['z'].values)
clim = z500.sel(time=train_years).groupby('time.dayofyear').mean()
# compute RMSE


rmse_clim = computed_weighted_rmse(clim.sel(dayofyear=z500.sel(time=test_years).time.dt.dayofyear), z500)
print(rmse_clim['z'].values)

def get_train_valid_test_dataset(lead_steps, z500_dataset):
    # Split train, valid and test dataset
    train_data = z500_dataset.sel(time=train_years)
    valid_data = z500_dataset.sel(time=valid_years)
    test_data = z500_dataset.sel(time=test_years)

    # Normalize the data using the mean and standard deviation of the training data
    mean = train_data.mean()
    std = train_data.std()

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    mean = mean['z'].values # extract numerical value from xarray Dataset
    std = std['z'].values # extract numerical value from xarray Dataset

    # Create inputs and outputs that are shifted by lead_steps
    X_train = train_data.z.isel(time=slice(None, -lead_steps)).values[..., None]
    Y_train = train_data.z.isel(time=slice(lead_steps, None)).values[..., None]
    X_valid = valid_data.z.isel(time=slice(None, -lead_steps)).values[..., None]
    Y_valid = valid_data.z.isel(time=slice(lead_steps, None)).values[..., None]  
    X_test = test_data.z.isel(time=slice(None, -lead_steps)).values[..., None]
    Y_test = test_data.z.isel(time=slice(lead_steps, None)).values[..., None]
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, mean, std = get_train_valid_test_dataset(lead_time_steps, z500)



# CNN

model = keras.Sequential([
    keras.layers.Conv2D(32, 5, padding='same'),   # 32 channels with a 5x5 convolution
    keras.layers.ELU(),  # Slightly smoother alternative to ReLU
    keras.layers.Conv2D(32, 5, padding='same'),   # Same padding keeps the size identical.
    keras.layers.ELU(),
    keras.layers.Conv2D(1, 5, padding='same'),
    # No activation since we are solving a regression problem
])


model.build(X_train[:32].shape)
model.compile(keras.optimizers.Adam(1e-4), 'mse')

# With .summary() we can check the shape of the model




# Finally we can fit the model.
# For each epoch, the entire training dataset has passed through the neural network exactly once
# Each epoch should take about 10s
model.fit(X_train, Y_train, batch_size=32, epochs=10)

# Convert predictions backto xarray
pred_test = X_test[:, :, :, 0].copy()
pred_test[:] = model.predict(X_test).squeeze()   # To remove channel dimension which is 1

# Unnormalize
pred_test = pred_test * std + mean
# compute RMSE
computed_weighted_rmse(pred_test, target.isel(time=slice(lead_time_steps, None)))

# Note first you need to modify your predictions so they are an xarray instead of a numpy array
# This way you can access the latitude, longitude and time for each point in the array

# We do this by taking a copy of the original z500 object which has the correct time, 
# latitude and longitude, and replacing the data in this array with the predictions
pred_xarray = z500.z.sel(time=test_years).isel(time=slice(lead_time_steps, None)).copy()
pred_xarray.data = pred_test



t = xr.DataArray(np.datetime64('2017-10-01T00'))
fig, ax = plt.subplots(1, 3, figsize=(15, 15),subplot_kw=dict(projection=ccrs.PlateCarree()))

z500.z.sel(time=t).plot(ax=ax[0])
ax[0].set_title('Ground truth')

clim.z.sel(dayofyear=t.dt.dayofyear).plot(ax=ax[1])
ax[1].set_title('Climatology')

pred_xarray.sel(time=t).plot(ax=ax[2])
ax[2].set_title('Prediction')

