import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# plot x lim
episode=range(0,99000,1000)
#r=[11.4, 5.7, 7.1, 5.0, 7.4, 7.7, 7.3, 9.2, 7.6, 10.3, 7.3, 7.4, 8.3, 7.0, 5.7, 7.0, 8.3, 7.5, 7.0, 6.3, 9.9, 6.8, 7.9, 6.4, 6.9, 4.6, 8.8, 5.7, 6.4, 6.6, 7.4, 8.5, 7.8, 6.6, 5.9, 6.3, 7.5, 6.1, 6.8, 6.8, 8.1, 9.3, 5.4, 5.8, 8.9, 7.4, 7.5, 7.9, 8.1, 8.0, 8.0, 5.9, 6.2, 7.4, 7.5, 8.5, 7.5, 7.0, 5.7, 8.4, 6.1, 10.8, 6.5, 6.7, 7.3, 8.7, 5.6, 6.6, 7.3, 7.9, 7.1, 6.6, 5.1, 7.0, 6.4, 6.9, 7.3, 7.8, 7.1, 8.0, 7.3, 7.8, 8.2, 8.0, 7.6, 9.5, 6.4, 6.6, 6.3, 7.4, 7.1, 7.6, 9.0, 6.7, 6.3, 7.1, 7.5, 8.1, 8.2, 6.4, 7.2, 6.6, 7.1, 7.3, 6.2, 6.7, 8.0, 6.2, 7.3, 6.4, 7.5, 7.1, 7.7, 5.5, 8.9, 6.8, 7.4, 5.7, 6.8, 7.0, 7.3, 6.9, 7.5, 6.4, 6.4, 5.7, 6.2, 6.6, 9.6, 6.1, 4.8, 8.4, 6.9, 8.7, 6.8, 7.1, 6.7, 8.0, 6.9, 6.7, 6.6, 6.8, 7.8, 7.2, 7.7, 5.6, 6.8, 5.9, 9.7, 6.5, 8.3, 7.3, 5.3, 7.9, 7.8, 6.9, 7.3, 7.4, 7.0, 6.4, 8.8, 7.0, 5.8, 8.3, 6.9, 6.5, 7.7, 6.2, 7.3, 7.9, 6.5, 7.2, 7.9, 6.9, 7.9, 5.8, 7.6, 6.0, 6.4, 9.3, 6.7, 7.4, 5.1, 9.2, 7.7, 6.5, 6.0, 7.2, 7.2, 8.6, 5.3, 7.0, 6.9, 6.9, 7.9, 6.7, 7.8, 8.0, 6.5, 8.2, 5.9, 6.0, 7.3, 8.2, 7.0, 7.3, 7.2, 6.5, 5.5, 5.9, 6.1, 7.0, 7.2, 7.6, 6.4, 4.1, 6.5, 8.5, 6.2, 7.1, 6.4, 6.1, 5.1, 7.0, 6.9, 8.4, 7.6, 7.6, 7.8, 6.0, 7.3, 8.9, 7.7, 5.4, 7.2, 8.0, 8.0, 7.7, 5.5, 7.3, 7.5, 6.5, 7.0, 5.8, 7.2, 7.8, 8.5, 9.3, 7.5, 7.0, 6.2, 5.7, 7.2, 9.2, 9.0, 9.2, 8.8, 6.7, 7.3, 8.4, 7.0, 6.9, 10.0, 8.7, 8.4, 5.2, 5.1, 7.6, 8.9, 5.5, 6.1, 7.4, 7.6, 5.3, 7.8, 10.3, 6.8, 7.4, 6.6, 7.8, 6.9, 7.4, 5.6, 8.0, 8.3, 6.3, 5.5, 7.3, 6.6, 7.5, 7.1, 6.6, 8.6, 6.8, 8.3, 9.1, 5.6, 5.7, 7.5, 7.9, 6.7, 8.3, 7.4, 8.2, 7.2, 6.1, 7.4, 7.8, 8.6, 7.3, 8.5, 7.1, 7.6, 9.0, 7.6, 7.6, 6.9, 6.1, 5.4, 6.5, 5.8, 5.7, 7.8, 9.1, 7.2, 7.6, 7.1, 6.1, 6.6, 5.7, 7.4, 7.1, 8.5, 7.6, 5.2, 7.1, 7.9, 6.2, 5.4, 7.0, 5.0, 7.3, 6.9, 8.4, 7.3, 5.8, 9.5, 7.5, 5.4, 7.4, 8.4, 6.6, 6.6, 8.0, 6.4, 5.6, 6.8, 5.1, 7.5, 7.5, 6.2, 6.8, 5.4, 9.1, 7.9, 7.1, 6.5, 6.4, 7.8, 6.6, 7.1, 5.9, 5.5, 7.8, 6.4, 7.3, 6.1, 6.5, 8.2, 6.6, 5.1, 7.5, 6.5, 6.9, 6.2, 5.6, 7.5, 6.9, 7.4, 6.7, 7.7, 6.2, 7.5, 7.5, 7.0, 6.3, 5.9, 7.8, 5.6, 7.5, 7.6, 7.6, 7.6, 6.0, 7.9, 6.1, 6.8, 5.4, 6.1, 6.4, 7.5, 8.3, 6.8, 7.2, 5.9, 7.1, 5.0, 6.8, 7.8, 6.8, 8.2, 8.7, 7.0, 7.9, 6.2, 7.2, 6.3, 7.6, 9.1, 6.6, 9.2, 5.6, 5.9, 6.6, 7.1, 7.0, 7.8, 6.0, 7.6, 6.9, 8.2, 6.8, 8.5, 5.4, 6.5, 6.4, 6.1, 6.2, 7.6, 7.5, 6.8, 8.9, 6.1, 6.5, 6.5, 7.1, 7.4, 7.1, 7.9, 6.1, 6.9, 5.8, 8.0, 5.9, 7.0, 8.4, 5.4, 7.8, 6.0, 6.8, 8.6, 6.6, 7.0, 5.3, 6.2, 7.0, 7.6, 7.7, 6.0, 5.5, 5.7, 7.2, 7.2, 7.3, 7.6, 6.6, 7.7, 8.5, 7.8, 7.8, 5.6, 7.4, 6.1, 7.3, 6.4, 7.7, 5.5, 6.2, 7.3, 7.8, 7.1, 6.4, 8.6, 7.5, 5.7, 5.8, 7.3, 7.4, 6.9, 6.5, 7.1, 5.9, 6.7, 7.2, 6.1, 7.8, 7.5, 6.7, 7.7, 7.4, 5.2, 7.8, 7.3, 5.9, 10.4, 6.2, 6.9, 7.3, 7.7, 7.6, 7.8, 6.2, 6.2, 6.6, 6.5, 5.8, 7.8, 7.2, 7.8, 8.5, 5.3, 6.2, 6.4, 6.9, 9.1, 6.9, 7.6, 6.9, 8.7, 7.4, 7.3, 8.1, 6.2, 7.2, 5.5, 5.5, 7.1, 6.4, 6.6, 7.6, 7.2, 7.1, 7.8, 5.2, 7.4, 7.7, 6.2, 8.2, 7.5, 6.3, 7.0, 6.0, 6.5, 8.5, 7.4, 5.8, 6.6, 6.7, 6.8, 7.6, 8.0, 10.3, 6.7, 8.3, 6.0, 10.1, 7.8, 7.3, 6.9, 6.8, 6.3, 7.1, 6.3, 7.7, 7.4, 5.7, 5.4, 8.1, 9.7, 8.4, 8.0, 6.6, 6.7, 6.6, 5.6, 7.3, 7.0, 7.3, 8.3, 7.5, 7.3, 9.3, 6.4, 8.7, 6.8, 6.3, 8.5, 5.5, 6.3, 7.5, 6.6, 6.2, 5.4, 6.7, 6.4, 6.2, 5.4, 7.4, 7.1, 8.6, 7.7, 7.6, 5.6, 9.4, 7.0, 7.7, 7.3, 7.0, 6.9, 7.7, 8.1, 7.6, 6.0, 6.5, 7.8, 6.8, 6.9, 8.2, 7.2, 9.0, 5.3, 7.0, 7.6, 5.5, 6.7, 7.4, 7.0, 7.9, 7.0, 8.0, 6.7, 6.4, 8.0, 7.1, 7.8, 5.8, 6.8, 9.7, 7.4, 8.8, 7.5, 6.0, 6.9, 8.1, 9.2, 6.4, 5.9, 6.8, 6.7, 6.8, 8.4, 6.8, 7.9, 6.8, 7.0, 7.6, 7.5, 7.5, 6.9, 8.1, 6.9, 7.1, 7.0, 8.9, 8.2, 5.8, 6.7, 5.6, 7.5, 7.0, 7.0, 8.0, 6.2, 8.2, 7.3, 7.6, 6.7, 6.2, 6.9, 8.1, 8.1, 6.7, 6.9, 7.0, 6.9, 6.5, 8.1, 7.1, 6.5, 9.4, 7.5, 5.9, 7.6, 8.8, 5.2, 7.7, 5.2, 5.9, 6.0, 7.2, 7.3, 9.6, 7.0, 6.6, 6.3, 6.1, 6.0, 7.6, 5.5, 8.1, 7.1, 7.1, 8.1, 7.3, 6.8, 6.4, 7.6, 5.8, 8.4, 6.6, 7.5, 7.7, 7.4, 5.5, 6.5, 7.6, 7.3, 8.7, 6.7, 8.2, 7.8, 4.9, 7.7, 7.0, 5.8, 7.2, 6.6, 5.9, 7.7, 5.4, 5.3, 6.7, 7.6, 9.8, 5.7, 8.2, 5.9, 6.9, 8.0, 4.9, 7.3, 8.1, 6.5, 4.9, 8.1, 5.8, 7.9, 6.0, 7.3, 6.9, 7.9, 8.3, 8.8, 9.6, 7.6, 7.9, 5.9, 6.7, 6.6, 5.0, 7.2, 6.9, 6.2, 7.4, 9.2, 8.5, 6.7, 6.8, 8.3, 6.3, 7.7, 10.2, 7.9, 5.4, 8.1, 6.0, 6.0, 8.2, 6.8, 8.3, 7.4, 5.4, 8.5, 8.2, 6.4, 7.6, 9.8, 7.3, 6.1, 8.2, 6.9, 7.1, 7.1, 8.0, 8.5, 7.7, 7.4, 6.5, 7.3, 7.1, 7.7, 6.7, 6.5, 6.0, 8.0, 6.8, 6.4, 6.3, 6.1, 7.4, 6.7, 7.5, 8.2, 7.0, 7.1, 8.0, 5.2, 6.1, 8.7, 7.9, 8.1, 8.7, 6.6, 6.8, 5.8, 6.1, 6.3, 5.9, 5.9, 5.8, 7.5, 7.8, 6.8, 7.9, 7.4, 7.7, 7.1, 7.2, 8.0, 7.0, 7.6, 6.8, 7.1, 7.6, 5.7, 6.0, 7.8, 6.2, 6.0, 7.6, 6.0, 5.8, 5.2, 8.2, 6.5, 5.7, 7.6, 6.9, 6.6, 7.4, 7.6, 8.2, 7.3, 6.1, 6.7, 5.9, 6.9, 6.2, 6.5, 6.5, 7.7, 8.8, 8.2, 6.2, 7.0, 7.4, 8.0, 7.2, 7.0, 5.8, 8.1, 6.7, 9.0, 7.7, 6.2, 6.4, 6.3, 7.6, 6.3, 9.0, 6.4, 5.7, 6.4, 7.3, 7.9, 5.5, 6.7, 5.9, 5.8, 9.5, 7.7, 6.6, 7.0, 7.1, 5.4, 6.7, 6.9, 7.3, 6.9, 7.6, 8.6, 7.5, 7.1, 6.2, 7.1, 5.8, 6.6, 9.1, 7.2, 6.4, 7.1, 6.2, 6.7, 7.5, 6.6, 6.7, 6.6, 7.9, 7.1, 7.8, 7.1, 7.2, 7.1, 7.1, 7.4, 6.1, 6.3, 6.9, 9.6, 6.9, 7.0, 6.7, 7.3, 7.6, 8.0, 5.5, 7.6, 8.1, 7.3, 7.9, 7.3, 8.2, 6.5, 5.8, 8.7, 7.1, 7.4, 6.3, 5.7]
lr = ['0.0001', '0.0005','0.001']

#batch:64,width:30,lr:0.0001,count:245
#p1=[26.31, 28.176, 36.292, 37.234, 36.976, 42.948, 42.554, 42.386, 42.862, 38.894, 45.134, 45.59, 46.692, 41.1, 41.526, 45.298, 46.032, 46.63, 48.324, 47.808, 45.264, 44.494, 47.948, 49.222, 45.334, 45.164, 48.482, 45.82, 49.656, 49.968, 46.664, 50.362, 51.546, 48.318, 47.628, 47.862, 48.892, 51.892, 54.974, 49.99, 51.752, 50.25, 55.998, 55.376, 53.466, 54.518, 56.614, 57.56, 56.822, 52.616, 55.44, 55.242, 58.044, 56.782, 59.634, 55.096, 54.966, 57.464, 54.042, 58.9, 52.49, 58.288, 57.43, 51.636, 57.17, 59.992, 58.508, 59.156, 58.398, 53.662, 55.406, 59.5, 55.922, 56.85, 55.876, 57.976, 53.326, 57.256, 56.576, 58.712, 59.648, 60.802, 58.69, 58.126, 61.4, 55.554, 64.878, 61.798, 58.318, 59.402, 58.718, 57.654, 58.836, 61.482, 65.234, 58.48, 57.51, 56.13, 53.568]
#batch:64,width:30,lr:0.0005,count:268
#p2=[35.246, 44.832, 44.99, 46.572, 48.526, 46.414, 52.702, 49.444, 49.038, 49.566, 47.892, 54.452, 50.27, 56.246, 57.438, 54.838, 53.974, 52.418, 60.522, 51.798, 58.468, 56.644, 55.524, 60.008, 54.64, 56.434, 57.374, 59.672, 58.928, 59.688, 60.174, 59.084, 59.126, 58.834, 57.31, 59.88, 60.324, 57.556, 55.986, 56.42, 58.946, 55.132, 54.9, 58.13, 55.416, 57.384, 61.24, 55.14, 58.5, 62.57, 58.99, 55.622, 63.914, 59.606, 58.452, 57.554, 61.114, 58.934, 61.17, 60.274, 58.196, 63.418, 63.888, 62.788, 60.718, 59.17, 61.17, 64.458, 56.064, 59.536, 59.868, 57.338, 61.644, 60.91, 58.416, 65.008, 65.014, 61.534, 61.77, 59.196, 63.14, 61.204, 63.752, 61.692, 68.002, 64.104, 63.372, 64.518, 67.21, 65.362, 64.11, 65.252, 64.512, 66.99, 66.114, 63.146, 66.55, 62.23, 63.842]
#batch:64,width:30,lr:0.001,count:218
#p3=[37.442, 37.548, 41.554, 41.3, 39.282, 45.082, 47.724, 50.34, 45.504, 48.514, 49.142, 46.56, 48.042, 50.276, 51.196, 48.792, 47.774, 47.742, 47.572, 49.514, 48.854, 47.42, 49.534, 49.712, 48.496, 51.304, 50.286, 48.678, 48.246, 53.002, 49.046, 46.782, 48.774, 48.828, 48.348, 50.192, 53.654, 52.264, 50.28, 48.964, 53.094, 52.45, 51.548, 51.494, 50.736, 54.33, 49.788, 52.348, 49.71, 49.264, 48.272, 48.248, 53.336, 50.972, 52.588, 51.452, 52.286, 49.52, 50.064, 50.08, 51.792, 51.846, 50.802, 57.604, 55.428, 56.444, 55.222, 56.602, 56.462, 57.012, 54.722, 58.386, 58.82, 60.34, 60.898, 59.148, 57.974, 58.668, 57.404, 57.67, 58.14, 57.22, 58.42, 58.676, 58.712, 58.1, 61.224, 55.55, 62.568, 61.03, 62.438, 61.958, 61.262, 57.612, 61.936, 65.006, 62.268, 59.872, 60.984]
#batch:128,width:30,lr:0.0001,count:214
#p1=[25.02, 35.082, 36.902, 38.58, 40.1, 42.768, 47.846, 43.182, 48.936, 48.262, 54.546, 51.838, 55.136, 59.73, 52.186, 53.53, 54.906, 55.104, 51.066, 54.678, 51.94, 50.006, 47.11, 51.608, 52.05, 50.694, 52.496, 48.342, 51.434, 48.556, 48.306, 50.132, 47.35, 50.604, 52.696, 51.786, 55.478, 50.558, 51.692, 51.618, 51.844, 50.466, 49.938, 50.104, 53.86, 48.336, 52.078, 51.934, 52.776, 52.446, 56.578, 52.68, 57.598, 53.884, 56.196, 52.662, 51.904, 49.37, 55.212, 52.054, 52.916, 51.958, 54.91, 52.06, 52.03, 51.678, 51.43, 53.456, 56.222, 55.01, 53.16, 55.906, 54.286, 51.7, 55.116, 47.854, 45.67, 52.868, 53.288, 52.228, 55.34, 54.226, 53.19, 54.358, 54.882, 52.74, 51.468, 51.398, 50.208, 51.72, 52.02, 44.866, 46.43, 49.672, 50.792, 51.726, 51.924, 48.158, 53.038]
#batch:128,width:30,lr:0.0005,count:276
#p2=[42.99, 47.986, 54.836, 61.78, 62.188, 58.932, 57.562, 61.634, 61.606, 60.484, 61.43, 60.676, 64.612, 62.48, 68.694, 64.72, 66.858, 65.786, 64.572, 70.812, 68.072, 67.418, 68.272, 69.264, 67.996, 72.662, 65.962, 73.104, 67.702, 70.894, 73.58, 70.31, 72.866, 66.322, 73.996, 69.044, 70.952, 70.262, 72.098, 71.582, 70.748, 73.694, 69.736, 69.988, 70.724, 70.432, 74.162, 73.238, 72.268, 72.11, 67.632, 67.216, 71.274, 70.83, 73.212, 73.188, 71.748, 68.548, 76.784, 74.648, 70.08, 69.246, 69.188, 72.122, 75.244, 70.838, 71.444, 71.642, 71.718, 74.41, 68.728, 68.944, 70.092, 66.218, 68.646, 69.298, 71.822, 69.0, 71.328, 68.558, 70.076, 70.738, 71.706, 71.496, 66.94, 72.772, 63.784, 69.878, 70.848, 72.306, 65.092, 67.844, 67.564, 68.204, 70.384, 65.612, 65.58, 70.802, 69.8]
#batch:128,width:30,lr:0.001,count:179
#p3=[33.144, 39.754, 48.072, 48.138, 52.08, 50.746, 61.832, 59.322, 55.266, 60.69, 60.704, 59.97, 56.638, 57.49, 57.944, 56.536, 62.204, 56.598, 61.83, 59.22, 58.548, 60.008, 57.008, 59.238, 60.11, 59.8, 59.86, 57.26, 58.214, 63.974, 55.002, 64.122, 61.54, 58.53, 59.26, 60.114, 65.366, 59.008, 61.922, 62.518, 59.038, 59.718, 63.138, 60.158, 62.96, 58.202, 58.772, 62.57, 60.664, 58.828, 63.652, 57.406, 62.654, 59.672, 60.33, 61.14, 59.058, 62.45, 58.86, 59.62, 60.39, 59.556, 59.87, 58.374, 56.266, 59.35, 61.304, 60.84, 61.346, 58.89, 58.226, 60.906, 62.288, 60.824, 60.652, 64.962, 55.652, 57.466, 57.35, 57.628, 60.52, 60.028, 57.038, 59.346, 58.69, 60.652, 62.468, 57.546, 60.826, 56.9, 58.346, 58.704, 60.312, 61.532, 58.55, 59.684, 56.078, 56.36, 60.906]

#batch:64,width:60,lr:0.0001,count:209
#p1=[33.746, 44.076, 41.484, 44.954, 40.228, 41.602, 47.894, 42.642, 44.552, 47.268, 49.04, 46.414, 46.398, 44.464, 47.088, 45.4, 46.78, 48.178, 49.402, 54.846, 50.66, 52.444, 46.438, 51.636, 52.06, 50.06, 58.4, 50.422, 56.662, 49.984, 53.81, 54.76, 53.818, 55.222, 58.884, 58.788, 52.242, 52.722, 57.882, 58.644, 58.112, 60.69, 58.8, 59.834, 57.424, 59.202, 56.284, 56.19, 60.42, 56.348, 59.274, 55.298, 60.22, 59.014, 60.674, 64.34, 58.078, 59.686, 56.052, 56.986, 58.584, 60.004, 57.754, 60.132, 54.958, 53.3, 55.252, 56.262, 59.918, 62.874, 62.868, 55.648, 59.324, 59.254, 61.364, 55.856, 58.02, 61.116, 57.406, 54.04, 55.45, 54.086, 54.61, 54.422, 58.756, 52.586, 50.954, 54.762, 51.986, 52.376, 56.44, 58.004, 53.614, 58.358, 52.856, 57.618, 56.114, 54.798, 53.968]
#batch:64,width:60,lr:0.0005,count:225
#p2=[39.234, 45.81, 47.01, 44.89, 49.038, 51.416, 48.222, 49.918, 52.434, 53.002, 50.684, 49.108, 53.814, 51.392, 51.318, 50.154, 53.764, 55.58, 56.03, 50.838, 57.846, 60.576, 52.592, 53.506, 50.352, 55.628, 51.358, 57.526, 54.472, 57.072, 53.62, 57.766, 56.798, 55.12, 56.588, 58.872, 60.112, 56.782, 55.88, 56.554, 54.508, 56.054, 53.36, 56.332, 53.678, 55.294, 58.252, 54.948, 56.898, 55.194, 55.578, 56.9, 56.062, 57.726, 57.412, 57.288, 56.352, 56.29, 55.446, 56.838, 57.712, 53.722, 55.522, 55.224, 56.764, 57.048, 57.198, 55.238, 56.784, 58.538, 56.362, 57.346, 55.744, 57.81, 58.922, 56.348, 60.566, 52.822, 57.59, 59.378, 58.768, 56.438, 54.136, 58.784, 58.856, 56.316, 60.192, 58.036, 58.57, 58.834, 57.37, 56.312, 58.058, 58.054, 58.424, 58.376, 58.366, 59.25, 60.064]
#batch:64,width:60,lr:0.001,count:250
#p3=[32.88, 42.038, 51.29, 55.584, 58.938, 64.234, 65.094, 71.128, 69.51, 70.448, 68.57, 72.816, 67.332, 64.87, 65.908, 62.202, 70.534, 69.736, 72.632, 69.54, 73.182, 70.922, 68.6, 70.288, 72.758, 73.124, 68.426, 74.58, 74.164, 71.006, 69.214, 69.078, 69.056, 65.0, 68.356, 71.694, 69.562, 65.798, 68.804, 69.036, 67.718, 71.628, 67.402, 71.86, 65.154, 68.21, 69.152, 68.384, 63.702, 65.004, 65.446, 66.41, 63.07, 64.996, 66.424, 65.626, 63.66, 68.12, 68.968, 66.994, 66.784, 68.592, 69.104, 68.68, 73.048, 73.054, 68.096, 69.4, 69.248, 67.81, 74.628, 69.702, 72.518, 68.03, 67.626, 68.582, 70.97, 73.366, 70.954, 68.968, 68.01, 71.014, 75.162, 70.592, 71.478, 66.896, 69.284, 71.224, 69.85, 71.742, 63.48, 68.44, 62.918, 66.238, 66.276, 71.946, 69.448, 63.88, 68.76]
#batch:128,width:60,lr:0.0001,count:146
#p1=[29.9, 32.548, 33.078, 31.954, 33.34, 35.946, 35.488, 36.138, 34.328, 38.444, 38.644, 35.758, 38.298, 39.804, 41.718, 38.048, 39.228, 44.006, 40.388, 43.442, 44.12, 41.954, 45.262, 43.188, 44.934, 45.948, 45.93, 42.75, 46.298, 43.676, 43.944, 44.932, 47.394, 44.05, 41.308, 45.098, 43.862, 41.38, 45.566, 44.488, 45.602, 44.37, 47.31, 47.59, 48.306, 41.566, 45.088, 43.312, 42.178, 43.804, 46.604, 42.874, 43.314, 45.6, 45.388, 41.696, 44.454, 44.206, 45.28, 45.362, 46.666, 45.904, 44.168, 42.072, 45.296, 45.63, 44.368, 45.602, 47.508, 43.468, 44.606, 46.58, 43.818, 44.33, 41.336, 45.412, 45.206, 41.454, 44.348, 43.85, 46.018, 45.958, 45.082, 47.0, 44.696, 41.682, 47.62, 47.3, 47.37, 44.138, 43.474, 46.604, 44.11, 47.088, 46.508, 44.294, 44.976, 47.66, 45.692]
#batch:128,width:60,lr:0.0005,count:253
#p2=[34.704, 42.48, 42.486, 48.608, 46.192, 47.516, 49.088, 47.844, 50.218, 53.254, 54.776, 53.202, 54.526, 52.822, 53.088, 56.342, 56.248, 55.81, 54.012, 53.082, 58.016, 53.686, 56.404, 58.862, 59.662, 60.956, 55.486, 56.24, 52.494, 57.9, 58.27, 57.282, 60.92, 60.344, 56.574, 54.312, 59.286, 57.132, 54.646, 58.28, 61.248, 59.0, 59.364, 60.08, 58.284, 62.354, 59.71, 57.984, 64.32, 62.548, 68.878, 65.592, 62.04, 69.298, 65.02, 67.782, 66.984, 62.968, 68.588, 69.654, 70.962, 66.906, 71.728, 67.512, 70.134, 70.688, 71.616, 71.342, 73.186, 68.592, 73.0, 70.334, 67.03, 66.804, 72.306, 72.364, 71.42, 72.67, 71.988, 73.236, 73.72, 73.442, 73.814, 72.968, 73.784, 78.488, 73.402, 77.984, 67.33, 70.144, 76.242, 71.714, 80.116, 71.79, 69.474, 72.272, 70.982, 71.224, 71.948]
#batch:128,width:60,lr:0.001,count:274
#p3=[34.694, 42.582, 51.03, 48.426, 54.442, 51.342, 52.534, 48.0, 50.51, 48.608, 48.916, 54.776, 50.972, 51.044, 50.706, 50.652, 54.852, 52.382, 47.938, 52.972, 52.138, 51.628, 53.054, 50.376, 53.404, 52.816, 51.764, 52.572, 57.51, 56.5, 57.56, 61.2, 56.136, 57.79, 55.156, 60.248, 55.414, 56.47, 56.426, 57.106, 57.576, 55.048, 59.286, 52.796, 59.014, 52.638, 57.602, 58.682, 58.646, 54.158, 53.906, 57.324, 55.318, 54.832, 52.104, 54.476, 59.314, 56.29, 56.186, 59.978, 61.646, 58.328, 60.35, 58.842, 55.202, 57.928, 61.77, 58.358, 54.454, 59.606, 58.956, 57.908, 53.794, 57.898, 56.228, 54.76, 57.466, 55.042, 55.238, 59.006, 55.462, 60.856, 56.982, 53.872, 55.422, 55.92, 59.828, 56.712, 54.098, 59.074, 54.458, 54.742, 63.516, 60.426, 57.668, 63.752, 66.144, 65.618, 63.456]

#batch:64,width:120,lr:0.0001,count:216
#p1=[36.012, 45.028, 44.186, 48.968, 44.706, 48.838, 44.072, 45.996, 47.884, 51.094, 46.682, 50.5, 52.956, 52.2, 48.434, 52.164, 52.112, 54.346, 50.688, 50.972, 52.682, 53.242, 53.284, 55.11, 53.57, 52.09, 51.826, 51.25, 52.03, 56.534, 53.908, 55.2, 51.046, 53.664, 55.46, 53.018, 52.718, 51.338, 53.876, 50.986, 52.434, 52.734, 53.082, 52.564, 51.61, 55.404, 52.806, 52.64, 54.824, 55.158, 48.072, 51.762, 51.194, 53.466, 52.926, 51.758, 52.164, 47.264, 53.734, 51.308, 55.168, 52.668, 51.888, 51.058, 54.804, 51.094, 52.05, 54.33, 50.45, 54.408, 50.84, 51.142, 53.836, 51.116, 50.438, 50.582, 52.464, 54.56, 50.972, 52.59, 51.126, 54.502, 49.176, 53.07, 54.87, 53.3, 56.736, 50.674, 56.05, 55.746, 55.558, 52.598, 55.114, 49.624, 52.664, 54.842, 53.402, 52.712, 53.656]
#batch:64,width:120,lr:0.0005,count:242
#p2=[36.758, 44.864, 53.85, 57.078, 59.232, 62.606, 60.88, 58.704, 56.368, 56.556, 58.718, 61.21, 56.138, 58.642, 60.838, 57.098, 57.596, 58.622, 61.274, 60.68, 60.042, 61.848, 59.53, 60.292, 63.728, 59.78, 60.446, 63.216, 61.388, 60.76, 65.3, 61.188, 68.432, 63.43, 66.02, 65.458, 64.846, 66.75, 66.394, 64.954, 66.688, 65.158, 70.322, 68.528, 65.664, 69.966, 67.038, 69.508, 70.59, 68.364, 65.238, 62.102, 65.672, 68.544, 65.592, 67.032, 67.626, 71.136, 69.724, 64.252, 70.322, 67.836, 70.566, 65.686, 66.294, 67.074, 67.348, 68.688, 67.394, 67.31, 67.238, 65.824, 66.842, 69.094, 65.52, 68.152, 71.522, 66.372, 63.652, 67.574, 64.976, 66.048, 69.81, 69.842, 65.462, 65.346, 64.126, 67.112, 67.462, 68.4, 65.12, 66.722, 67.518, 63.464, 71.798, 64.504, 71.756, 67.044, 67.55]
#batch:64,width:120,lr:0.001,count:351
#p3=[37.708, 47.658, 53.474, 55.782, 57.428, 58.91, 58.172, 61.35, 61.584, 63.782, 67.612, 66.766, 72.548, 70.986, 76.552, 71.248, 71.174, 76.032, 73.924, 73.974, 72.248, 77.974, 75.598, 78.644, 76.712, 76.25, 72.93, 73.254, 76.772, 76.83, 76.3, 79.356, 75.158, 79.38, 80.238, 73.066, 75.658, 73.726, 79.128, 78.718, 72.874, 75.198, 77.958, 77.48, 77.904, 73.73, 76.798, 75.794, 74.14, 79.804, 75.358, 70.824, 75.23, 74.292, 77.276, 73.072, 75.302, 73.248, 74.862, 74.75, 71.594, 72.374, 74.16, 72.802, 67.744, 72.022, 74.762, 74.472, 78.132, 72.912, 72.946, 75.108, 75.056, 76.422, 74.844, 80.154, 77.32, 75.684, 76.766, 78.882, 80.3, 75.26, 76.84, 75.812, 80.628, 76.388, 75.312, 76.628, 75.232, 76.512, 76.64, 72.926, 80.92, 73.48, 73.406, 77.682, 75.042, 76.988, 77.432]
#batch:128,width:120,lr:0.0001,count:568
#p1=[33.564, 43.328, 44.27, 43.454, 46.304, 51.898, 49.382, 47.758, 49.208, 55.456, 60.4, 58.95, 56.612, 57.412, 54.838, 54.346, 57.2, 61.894, 65.964, 64.26, 64.804, 70.69, 72.338, 71.324, 72.536, 72.298, 77.164, 76.19, 80.79, 79.076, 78.56, 88.068, 82.51, 83.756, 82.026, 82.094, 83.842, 79.85, 87.434, 85.642, 88.166, 86.918, 88.326, 89.096, 85.378, 83.764, 83.28, 86.5, 84.644, 84.422, 87.946, 85.33, 89.324, 89.242, 85.226, 88.834, 84.848, 88.706, 88.748, 86.234, 88.61, 88.212, 90.356, 90.784, 88.94, 85.614, 90.726, 89.478, 87.458, 85.668, 88.55, 89.104, 88.458, 86.296, 87.898, 87.724, 86.824, 87.996, 89.37, 91.218, 87.684, 89.892, 92.382, 89.862, 90.444, 83.886, 88.736, 91.126, 88.018, 90.86, 91.192, 89.886, 88.822, 89.532, 87.662, 87.722, 86.48, 90.736, 89.218]
#batch:128,width:120,lr:0.001,count:210
#p2=[39.124, 43.94, 44.17, 50.868, 53.552, 55.648, 55.728, 55.094, 55.878, 59.888, 61.26, 57.574, 61.642, 59.038, 58.436, 56.996, 60.172, 59.244, 56.424, 60.698, 58.832, 57.254, 60.356, 64.102, 61.966, 64.038, 63.896, 61.4, 65.338, 64.748, 63.334, 62.158, 66.08, 61.564, 63.468, 60.594, 63.852, 60.012, 66.636, 60.73, 60.446, 60.946, 62.354, 60.734, 60.106, 63.728, 61.876, 58.54, 63.566, 61.652, 59.976, 62.108, 63.432, 67.532, 57.984, 67.194, 64.432, 63.334, 64.064, 62.464, 61.024, 60.938, 59.836, 63.526, 64.226, 64.296, 63.504, 64.088, 64.536, 65.96, 61.96, 64.394, 64.862, 65.284, 60.986, 63.784, 64.306, 67.624, 64.17, 66.746, 65.702, 62.326, 63.638, 66.144, 63.82, 61.678, 68.25, 64.818, 62.19, 66.574, 63.894, 62.706, 63.408, 67.452, 64.334, 69.228, 65.248, 69.162, 65.564]
#batch:128,width:120,lr:0.0005,count:266
p3=[33.072, 44.266, 45.526, 52.664, 54.066, 55.224, 54.424, 53.874, 60.646, 59.598, 61.872, 61.592, 68.48, 64.412, 66.75, 70.37, 65.758, 65.226, 68.534, 67.14, 66.846, 68.576, 65.924, 66.784, 68.992, 69.15, 65.816, 66.278, 67.49, 69.252, 67.61, 71.778, 72.698, 72.06, 69.04, 67.618, 70.22, 67.16, 70.83, 71.384, 73.77, 70.974, 69.276, 73.444, 70.364, 73.108, 69.092, 72.536, 67.122, 73.27, 69.902, 68.626, 72.76, 75.532, 74.118, 72.69, 73.646, 69.286, 70.538, 66.272, 69.124, 68.806, 68.278, 67.032, 75.142, 70.39, 69.956, 70.956, 70.94, 68.28, 69.632, 71.582, 71.982, 68.644, 70.826, 69.418, 72.204, 72.31, 73.312, 71.43, 71.742, 71.974, 73.384, 71.776, 73.218, 69.116, 71.758, 68.366, 70.578, 67.888, 66.832, 73.212, 74.664, 71.938, 72.972, 70.604, 69.644, 70.366, 72.938]
#y1=[0.245,0.268,0.218]y2=[0.209,0.225,0.250]y3=[0.216,0.242,0.351]
y1=[0.214,0.276,0.179]
y2=[0.146,0.253,0.274]
y3=[0.568,0.210,0.266]

ChooseFig=['TSA','PRF']
Choose='PRF'# your choice

if Choose=='TSA':
    figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties = 'Times New Roman', fontsize=4)
    plt.yticks(fontproperties = 'Times New Roman', fontsize=4)
    plt.axis([0,100000,0,125])

    plt.plot(episode,p1, 'darkorange', label='lr=0.0001',linewidth = '.3')
    plt.plot(episode,p2, 'purple', label='lr=0.0005',linewidth = '.3')
    plt.plot(episode,p3, 'blue', label='lr=0.001',linewidth = '.3')
    #plt.plot(episode,r, 'red', label='random',linewidth = '.3')

    plt.legend(loc='upper right', prop={'family':'Times New Roman', 'size':4})
    plt.title('width=120, batch size=128', fontdict={'family' : 'Times New Roman', 'size':4})

    plt.xlabel('episode', fontdict={'family' : 'Times New Roman', 'size':4})
    plt.ylabel('ave_reward', fontdict={'family' : 'Times New Roman', 'size':4})
    plt.show()

if Choose=='PRF':
    figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties = 'Times New Roman', fontsize=4)
    plt.yticks(fontproperties = 'Times New Roman', fontsize=4)
    plt.plot(lr,y1, 'blue',marker= '+',label='width=30',linewidth = '.8')
    plt.plot(lr,y2, 'darkorange',marker= '+',label='width=60',linewidth = '.8')
    plt.plot(lr,y3, 'purple', marker='+',label='width=120',linewidth = '.8')
    #plt.plot(lr,y4, 'cyan',marker= '+',label='width=240',linewidth = '.8')
    plt.ylim([0,0.7])
    plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':4})
    plt.title('The frequency of reaching 125 steps with 1 hidden layer (batchsize=128)', fontdict={'family' : 'Times New Roman', 'size':4})

    plt.xlabel('learning_rate', fontdict={'family' : 'Times New Roman', 'size':4})
    plt.ylabel('times', fontdict={'family' : 'Times New Roman', 'size':4})
    plt.show()