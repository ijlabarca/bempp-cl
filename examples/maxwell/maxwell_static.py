# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---



# +
import bempp_cl.api
import numpy as np

bempp_cl.api.enable_console_logging()
# bempp_cl.api.pool.create_device_pool("AMD")
# -

# For this notebook we will use two spheres of radius 1.0, centered at the origin.

mesh = bempp_cl.api.shapes.sphere(r=1., origin=(0, 0, 0), h=0.2)

# +
theta = np.pi / 4  # Incident field travelling at a 45 degree angle
direction = np.array([np.cos(theta), np.sin(theta), 0])
polarization = np.array([0, 0, 1.0])


def incident_field(point):
    return polarization * np.dot(point, direction)


@bempp_cl.api.real_callable
def tangential_trace(point, n, domain_index, result):
    value = polarization * np.dot(point, direction)
    result[:] = np.cross(value, n)


@bempp_cl.api.real_callable
def neumann_trace(point, n, domain_index, result):
    value = np.cross(direction, polarization) * np.dot(point, direction)
    result[:] = np.cross(value, n)


# -


# +
from bempp_cl.api.operators.boundary.maxwell import single_layer,double_layer


div_space  = bempp_cl.api.function_space(mesh, "RWG", 0)
curl_space = bempp_cl.api.function_space(mesh, "SNC", 0)

# Next, we define the Maxwell electric field boundary operator and the identity operator. For Maxwell problems, the ``domain`` and ``range`` spaces should be div-conforming, while the ``dual_to_range`` space should be curl conforming.

SL = bempp_cl.api.operators.boundary.maxwell.single_layer(div_space, div_space, curl_space)
DL = bempp_cl.api.operators.boundary.maxwell.double_layer(div_space, div_space, curl_space)
II = bempp_cl.api.operators.boundary.sparse.identity(div_space, div_space, curl_space)

# TODO: Define operator A

# -

# +

# The following code assembles the right-hand sides.

RHS = bempp_cl.api.GridFunction(div_space, fun=tangential_trace, dual_space=curl_space)

# LU Direct Solver

from bempp_cl.api.linalg import lu

SOL = lu(A, RHS)

# -
