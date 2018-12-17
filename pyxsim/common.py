from yt.utilities.cosmology import Cosmology
from pyxsim.utils import mylog
import numpy as np
from yt.funcs import iterable
from collections import defaultdict
from yt.units.yt_array import YTArray
from pyxsim.utils import parse_value
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    communication_system, parallel_objects

new_photon_units = {"energy": "keV",
                    "dx": "kpc",
                    "pos": "kpc",
                    "vel": "km/s"}

comm = communication_system.communicators[-1]


def find_object_bounds(data_source):
    """
    This logic is required to determine the bounds of the object, which is 
    solely for fixing coordinates at periodic boundaries
    """

    if hasattr(data_source, "base_object"):
        # This a cut region so we'll figure out
        # its bounds from its parent object
        data_src = data_source.base_object
    else:
        data_src = data_source

    if hasattr(data_src, "left_edge"):
        # Region or grid
        c = 0.5 * (data_src.left_edge + data_src.right_edge)
        w = data_src.right_edge - data_src.left_edge
        le = -0.5 * w + c
        re = 0.5 * w + c
    elif hasattr(data_src, "radius") and not hasattr(data_src, "height"):
        # Sphere
        le = -data_src.radius + data_src.center
        re = data_src.radius + data_src.center
    else:
        # Not sure what to do with any other object yet, so just
        # return the domain edges and punt.
        mylog.warning("You are using a region that is not currently "
                      "supported for straddling periodic boundaries. "
                      "Check to make sure that your region does not "
                      "do so.")
        le = data_source.ds.domain_left_edge
        re = data_source.ds.domain_right_edge

    return le.to("kpc"), re.to("kpc")


def make_hsml(source_type):
    def _smoothing_length(field, data):
        hsml = data[source_type, "particle_mass"] / data[source_type, "density"]
        hsml *= 3.0 / (4.0 * np.pi)
        hsml **= 1. / 3.
        return 2.5 * hsml
    return _smoothing_length


def determine_fields(ds, source_type, point_sources):
    ds_type = ds.index.__class__.__name__
    if "ParticleIndex" in ds_type:
        position_fields = [(source_type, "particle_position_%s" % ax) for ax in "xyz"]
        velocity_fields = [(source_type, "particle_velocity_%s" % ax) for ax in "xyz"]
        if source_type in ["PartType0", "gas", "Gas"]:
            width_field = (source_type, "smoothing_length")
            if width_field not in ds.field_info and not point_sources:
                _smoothing_length = make_hsml(source_type)
                width_field = (source_type, "pyxsim_smoothing_length")
                ds.add_field(width_field, _smoothing_length, particle_type=True,
                             units='code_length')
        else:
            width_field = None
    else:
        position_fields = [("index", ax) for ax in "xyz"]
        velocity_fields = [(source_type, "velocity_%s" % ax) for ax in "xyz"]
        width_field = ("index", "dx")
    if point_sources:
        width_field = None
    return position_fields, velocity_fields, width_field


def concatenate_photons(ds, photons, photon_units):
    for key in ["pos", "vel", "dx", "energy", "num_photons"]:
        if key in photons and len(photons[key]) > 0:
            if key in ["pos", "vel"]:
                photons[key] = np.swapaxes(np.concatenate(photons[key],
                                                          axis=1), 0, 1)
            else:
                photons[key] = np.concatenate(photons[key])
            if key in photon_units:
                photons[key] = ds.arr(photons[key], photon_units[key])
                photons[key].convert_to_units(new_photon_units[key])
        elif key == "num_photons":
            photons[key] = np.array([])
        else:
            photons[key] = YTArray([], new_photon_units[key])


def _from_data_source(data_source, redshift, area, exp_time,
                      source_model, shift_normal=None,
                      point_sources=False, parameters=None,
                      center=None, dist=None, cosmology=None,
                      velocity_fields=None):

    ds = data_source.ds

    if parameters is None:
        parameters = {}
    if cosmology is None:
        if hasattr(ds, 'cosmology'):
            cosmo = ds.cosmology
        else:
            cosmo = Cosmology()
    else:
        cosmo = cosmology
    if dist is None:
        if redshift <= 0.0:
            msg = "If redshift <= 0.0, you must specify a distance to the " \
                  "source using the 'dist' argument!"
            mylog.error(msg)
            raise ValueError(msg)
        D_A = cosmo.angular_diameter_distance(0.0, redshift).in_units("Mpc")
    else:
        D_A = parse_value(dist, "kpc")
        if redshift > 0.0:
            mylog.warning("Redshift must be zero for nearby sources. "
                          "Resetting redshift to 0.0.")
            redshift = 0.0

    if isinstance(center, str):
        if center == "center" or center == "c":
            parameters["center"] = ds.domain_center
        elif center == "max" or center == "m":
            parameters["center"] = ds.find_max("density")[-1]
    elif iterable(center):
        if isinstance(center, YTArray):
            parameters["center"] = center.in_units("code_length")
        elif isinstance(center, tuple):
            if center[0] == "min":
                parameters["center"] = ds.find_min(center[1])[-1]
            elif center[0] == "max":
                parameters["center"] = ds.find_max(center[1])[-1]
            else:
                raise RuntimeError
        else:
            parameters["center"] = ds.arr(center, "code_length")
    elif center is None:
        if hasattr(data_source, "left_edge"):
            parameters["center"] = 0.5 * (data_source.left_edge + data_source.right_edge)
        else:
            parameters["center"] = data_source.get_field_parameter("center")

    parameters["fid_exp_time"] = parse_value(exp_time, "s")
    parameters["fid_area"] = parse_value(area, "cm**2")
    parameters["fid_redshift"] = redshift
    parameters["fid_d_a"] = D_A
    parameters["hubble"] = cosmo.hubble_constant
    parameters["omega_matter"] = cosmo.omega_matter
    parameters["omega_lambda"] = cosmo.omega_lambda

    if redshift > 0.0:
        mylog.info("Cosmology: h = %g, omega_matter = %g, omega_lambda = %g" %
                   (cosmo.hubble_constant, cosmo.omega_matter, cosmo.omega_lambda))
    else:
        mylog.info("Observing local source at distance %s." % D_A)

    D_A = parameters["fid_d_a"].in_cgs()
    dist_fac = 1.0 / (4. * np.pi * D_A.value * D_A.value * (1. + redshift) ** 2)
    spectral_norm = parameters["fid_area"].v * parameters["fid_exp_time"].v * dist_fac

    source_model.setup_model(data_source, redshift, spectral_norm)

    p_fields, v_fields, w_field = determine_fields(ds,
                                                   source_model.source_type,
                                                   point_sources)

    if velocity_fields is not None:
        v_fields = velocity_fields

    if p_fields[0] == ("index", "x"):
        parameters["data_type"] = "cells"
    else:
        parameters["data_type"] = "particles"

    citer = data_source.chunks([], "io")

    photons = defaultdict(list)

    for chunk in parallel_objects(citer):

        chunk_data = source_model(chunk)

        if chunk_data is not None:
            ncells, number_of_photons, idxs, energies = chunk_data
            photons["num_photons"].append(number_of_photons)
            photons["energy"].append(energies)
            photons["pos"].append(np.array([chunk[p_fields[0]].d[idxs],
                                            chunk[p_fields[1]].d[idxs],
                                            chunk[p_fields[2]].d[idxs]]))
            photons["vel"].append(np.array([chunk[v_fields[0]].d[idxs],
                                            chunk[v_fields[1]].d[idxs],
                                            chunk[v_fields[2]].d[idxs]]))
            if w_field is None:
                photons["dx"].append(np.zeros(ncells))
            else:
                photons["dx"].append(chunk[w_field].d[idxs])

    source_model.cleanup_model()

    photon_units = {"pos": ds.field_info[p_fields[0]].units,
                    "vel": ds.field_info[v_fields[0]].units,
                    "energy": "keV"}
    if w_field is None:
        photon_units["dx"] = "kpc"
    else:
        photon_units["dx"] = ds.field_info[w_field].units

    concatenate_photons(ds, photons, photon_units)

    nphotons = photons["num_photons"].sum()
    all_nphotons = comm.mpi_allreduce(nphotons)
    if all_nphotons == 0:
        raise RuntimeError("No photons were generated!!")

    c = parameters["center"].to("kpc")

    if sum(ds.periodicity) > 0:
        # Fix photon coordinates for regions crossing a periodic boundary
        dw = ds.domain_width.to("kpc")
        le, re = find_object_bounds(data_source)
        for i in range(3):
            if ds.periodicity[i] and photons["pos"].shape[0] > 0:
                tfl = photons["pos"][:, i] < le[i]
                tfr = photons["pos"][:, i] > re[i]
                photons["pos"][tfl, i] += dw[i]
                photons["pos"][tfr, i] -= dw[i]

    # Re-center all coordinates
    if photons["pos"].shape[0] > 0:
        photons["pos"] -= c

    mylog.info("Finished generating photons.")
    mylog.info("Number of photons generated: %d" % int(np.sum(photons["num_photons"])))
    mylog.info("Number of cells with photons: %d" % photons["dx"].size)

    return photons, parameters, cosmo
