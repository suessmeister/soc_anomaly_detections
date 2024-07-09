import numpy as np
import pandas as pd
import matplotlib.pylab as plt

#1. Scaling data so that the X and Y are similar
# option a. using the StandardScaler we can get the axis somewhat similar.
# however, outliers are still a problem in this method and does not really get rid of them...

#we can use quantiles to get a robust preprocessing if we have outliers
# i.e. split data into 4ths and map accordingly.

# this is option b. using the QuantileTransformer()
# this sets the scale on both axis pretty much perfectly, but it is more invariant.
# typically better for the outliers

# now onto metrics. these help when determining which model to use

# test