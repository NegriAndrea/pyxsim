"""
Classes for generating lists of photons
"""
import numpy as np
from pyxsim.lib.sky_functions import pixel_to_cel, \
    scatter_events, doppler_shift
from yt.utilities.physical_constants import clight
from yt.utilities.cosmology import Cosmology
from yt.utilities.orientation import Orientation
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    communication_system, get_mpi_type, parallel_capable
from yt.units.yt_array import YTQuantity, YTArray, uconcatenate
import h5py
from pyxsim.spectral_models import absorb_models
from pyxsim.utils import force_unicode, validate_parameters, \
    key_warning, ParameterDict, mylog
from pyxsim.event_list import EventList, MultiEventList
from soxs.utils import parse_prng
from pyxsim.common import _from_data_source

comm = communication_system.communicators[-1]

new_photon_units = {"energy": "keV",
                    "dx": "kpc",
                    "pos": "kpc",
                    "vel": "km/s"}

old_photon_keys = {"Energy": "energy",
                   "NumberOfPhotons": "num_photons"}
old_parameter_keys = {"FiducialExposureTime": "fid_exp_time",
                      "FiducialArea": "fid_area",
                      "FiducialRedshift": "fid_redshift",
                      "FiducialAngularDiameterDistance": "fid_d_a",
                      "HubbleConstant": "hubble",
                      "OmegaLambda": "omega_lambda",
                      "OmegaMatter": "omega_matter",
                      "DataType": "data_type"}


class PhotonList(object):

    def __init__(self, photons, parameters, cosmo):
        self.photons = photons
        self.parameters = ParameterDict(parameters, "PhotonList", 
                                        old_parameter_keys)
        self.cosmo = cosmo
        self.num_cells = len(photons["dx"])

        p_bins = np.cumsum(photons["num_photons"])
        self.p_bins = np.insert(p_bins, 0, [np.int64(0)])

    def keys(self):
        return self.photons.keys()

    def items(self):
        ret = []
        for k, v in self.photons.items():
            if k == "energy":
                ret.append((k, self[k]))
            else:
                ret.append((k,v))
        return ret

    def values(self):
        ret = []
        for k, v in self.photons.items():
            if k == "energy":
                ret.append(self[k])
            else:
                ret.append(v)
        return ret

    def __getitem__(self, key):
        if key in old_photon_keys:
            k = old_photon_keys[key]
            mylog.warning(key_warning % ("PhotonList", k))
        else:
            k = key
        if k == "energy":
            return [self.photons["energy"][self.p_bins[i]:self.p_bins[i+1]]
                    for i in range(self.num_cells)]
        else:
            return self.photons[k]

    def __contains__(self, key):
        if key in old_photon_keys:
            mylog.warning(key_warning % ("PhotonList", old_photon_keys[key]))
            return True
        return key in self.photons

    def __iter__(self):
        return iter(self.photons)

    def __repr__(self):
        return self.photons.__repr__()

    def __add__(self, other):
        validate_parameters(self.parameters, other.parameters)
        for param in ["hubble_constant", "omega_matter", 
                      "omega_lambda", "omega_curvature"]:
            v1 = getattr(self.cosmo, param)
            v2 = getattr(other.cosmo, param)
            check_equal = np.allclose(np.array(v1), np.array(v2), rtol=0.0, atol=1.0e-10)
            if not check_equal:
                raise RuntimeError("The values for the parameter '%s' in the two" % param +
                                   " cosmologies are not identical (%s vs. %s)!" % (v1, v2))
        photons = {}
        for item1, item2 in zip(self.photons.items(), other.photons.items()):
            k1, v1 = item1
            k2, v2 = item2
            photons[k1] = uconcatenate([v1,v2])
        return PhotonList(photons, self.parameters, self.cosmo)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize a :class:`~pyxsim.photon_list.PhotonList` from 
        the HDF5 file *filename*.
        """
        mylog.info("Reading photons from %s." % filename)

        photons = {}
        parameters = {}

        f = h5py.File(filename, "r")

        p = f["/parameters"]
        parameters["fid_exp_time"] = YTQuantity(p["fid_exp_time"].value, "s")
        parameters["fid_area"] = YTQuantity(p["fid_area"].value, "cm**2")
        parameters["fid_redshift"] = p["fid_redshift"].value
        parameters["fid_d_a"] = YTQuantity(p["fid_d_a"].value, "Mpc")
        parameters["hubble"] = p["hubble"].value
        parameters["omega_matter"] = p["omega_matter"].value
        parameters["omega_lambda"] = p["omega_lambda"].value
        if "data_type" in p:
            parameters["data_type"] = force_unicode(p["data_type"].value)
        else:
            parameters["data_type"] = "cells"

        d = f["/data"]

        num_cells = d["x"].size
        start_c = comm.rank*num_cells//comm.size
        end_c = (comm.rank+1)*num_cells//comm.size

        photons["pos"] = YTArray(np.zeros((num_cells, 3)), "kpc")
        photons["vel"] = YTArray(np.zeros((num_cells, 3)), "km/s")
        photons["pos"][:, 0] = d["x"][start_c:end_c]
        photons["pos"][:, 1] = d["y"][start_c:end_c]
        photons["pos"][:, 2] = d["z"][start_c:end_c]
        photons["vel"][:, 0] = d["vx"][start_c:end_c]
        photons["vel"][:, 1] = d["vy"][start_c:end_c]
        photons["vel"][:, 2] = d["vz"][start_c:end_c]
        photons["dx"] = YTArray(d["dx"][start_c:end_c], "kpc")

        n_ph = d["num_photons"][:]

        if comm.rank == 0:
            start_e = np.int64(0)
        else:
            start_e = n_ph[:start_c].sum()
        end_e = start_e + np.int64(n_ph[start_c:end_c].sum())

        photons["num_photons"] = n_ph[start_c:end_c]
        photons["energy"] = YTArray(d["energy"][start_e:end_e], "keV")

        f.close()

        cosmo = Cosmology(hubble_constant=parameters["hubble"],
                          omega_matter=parameters["omega_matter"],
                          omega_lambda=parameters["omega_lambda"])

        mylog.info("Read %d photons from %d %s." % (n_ph.sum(), num_cells, 
                                                    parameters["data_type"]))

        return cls(photons, parameters, cosmo)

    @classmethod
    def from_data_source(cls, data_source, redshift, area,
                         exp_time, source_model, point_sources=False,
                         parameters=None, center=None, dist=None, 
                         cosmology=None, velocity_fields=None):
        r"""
        Initialize a :class:`~pyxsim.photon_list.PhotonList` from a yt data
        source. The redshift, collecting area, exposure time, and cosmology
        are stored in the *parameters* dictionary which is passed to the
        *source_model* function.

        Parameters
        ----------
        data_source : :class:`~yt.data_objects.data_containers.YTSelectionContainer`
            The data source from which the photons will be generated.
        redshift : float
            The cosmological redshift for the photons.
        area : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`
            The collecting area to determine the number of photons. If units are
            not specified, it is assumed to be in cm^2.
        exp_time : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`
            The exposure time to determine the number of photons. If units are
            not specified, it is assumed to be in seconds.
        source_model : :class:`~pyxsim.source_models.SourceModel`
            A source model used to generate the photons.
        point_sources : boolean, optional
            If True, the photons will be assumed to be generated from the exact
            positions of the cells or particles and not smeared around within
            a volume. Default: False
        parameters : dict, optional
            A dictionary of parameters to be passed for the source model to use,
            if necessary.
        center : string or array_like, optional
            The origin of the photon spatial coordinates. Accepts "c", "max", or
            a coordinate. If not specified, pyxsim attempts to use the "center"
            field parameter of the data_source.
        dist : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`, optional
            The angular diameter distance, used for nearby sources. This may be
            optionally supplied instead of it being determined from the
            *redshift* and given *cosmology*. If units are not specified, it is
            assumed to be in kpc. To use this, the redshift must be set to zero.
        cosmology : :class:`~yt.utilities.cosmology.Cosmology`, optional
            Cosmological information. If not supplied, we try to get the
            cosmology from the dataset. Otherwise, LCDM with the default yt
            parameters is assumed.
        velocity_fields : list of fields, optional
            The yt fields to use for the velocity. If not specified, the
            following will be assumed:
            ['velocity_x', 'velocity_y', 'velocity_z'] for grid datasets
            ['particle_velocity_x', 'particle_velocity_y', 'particle_velocity_z'] for particle datasets

        Examples
        --------
        >>> thermal_model = ThermalSourceModel(apec_model, Zmet=0.3)
        >>> redshift = 0.05
        >>> area = 6000.0 # assumed here in cm**2
        >>> time = 2.0e5 # assumed here in seconds
        >>> sp = ds.sphere("c", (500., "kpc"))
        >>> my_photons = PhotonList.from_data_source(sp, redshift, area,
        ...                                          time, thermal_model)
        """
        return cls(*_from_data_source(data_source, redshift, area,
                                      exp_time, source_model,
                                      point_sources=point_sources,
                                      parameters=parameters, center=center,
                                      dist=dist, cosmology=cosmology,
                                      velocity_fields=velocity_fields))

    def write_h5_file(self, photonfile):
        """
        Write the :class:`~pyxsim.photon_list.PhotonList` to the HDF5 
        file *photonfile*.
        """
        if parallel_capable:

            mpi_long = get_mpi_type("int64")
            mpi_double = get_mpi_type("float64")

            local_num_cells = len(self.photons["x"])
            sizes_c = comm.comm.gather(local_num_cells, root=0)

            local_num_photons = np.sum(self.photons["num_photons"])
            sizes_p = comm.comm.gather(local_num_photons, root=0)

            if comm.rank == 0:
                num_cells = sum(sizes_c)
                num_photons = sum(sizes_p)
                disps_c = [sum(sizes_c[:i]) for i in range(len(sizes_c))]
                disps_p = [sum(sizes_p[:i]) for i in range(len(sizes_p))]
                x = np.zeros(num_cells)
                y = np.zeros(num_cells)
                z = np.zeros(num_cells)
                vx = np.zeros(num_cells)
                vy = np.zeros(num_cells)
                vz = np.zeros(num_cells)
                dx = np.zeros(num_cells)
                n_ph = np.zeros(num_cells, dtype="int64")
                e = np.zeros(num_photons)
            else:
                sizes_c = []
                sizes_p = []
                disps_c = []
                disps_p = []
                x = np.empty([])
                y = np.empty([])
                z = np.empty([])
                vx = np.empty([])
                vy = np.empty([])
                vz = np.empty([])
                dx = np.empty([])
                n_ph = np.empty([])
                e = np.empty([])

            comm.comm.Gatherv([self.photons["pos"][:,0].d, local_num_cells, mpi_double],
                              [x, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["pos"][:,1].d, local_num_cells, mpi_double],
                              [y, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["pos"][:,2].d, local_num_cells, mpi_double],
                              [z, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["vel"][:,0].d, local_num_cells, mpi_double],
                              [vx, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["vel"][:,1].d, local_num_cells, mpi_double],
                              [vy, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["vel"][:,2].d, local_num_cells, mpi_double],
                              [vz, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["dx"].d, local_num_cells, mpi_double],
                              [dx, (sizes_c, disps_c), mpi_double], root=0)
            comm.comm.Gatherv([self.photons["num_photons"], local_num_cells, mpi_long],
                              [n_ph, (sizes_c, disps_c), mpi_long], root=0)
            comm.comm.Gatherv([self.photons["energy"].d, local_num_photons, mpi_double],
                              [e, (sizes_p, disps_p), mpi_double], root=0)

        else:

            x = self.photons["pos"][:,0].d
            y = self.photons["pos"][:,1].d
            z = self.photons["pos"][:,2].d
            vx = self.photons["vel"][:,0].d
            vy = self.photons["vel"][:,1].d
            vz = self.photons["vel"][:,2].d
            dx = self.photons["dx"].d
            n_ph = self.photons["num_photons"]
            e = self.photons["energy"].d

        if comm.rank == 0:

            f = h5py.File(photonfile, "w")

            # Parameters

            p = f.create_group("parameters")
            p.create_dataset("fid_area", data=float(self.parameters["fid_area"]))
            p.create_dataset("fid_exp_time", data=float(self.parameters["fid_exp_time"]))
            p.create_dataset("fid_redshift", data=self.parameters["fid_redshift"])
            p.create_dataset("hubble", data=self.parameters["hubble"])
            p.create_dataset("omega_matter", data=self.parameters["omega_matter"])
            p.create_dataset("omega_lambda", data=self.parameters["omega_lambda"])
            p.create_dataset("fid_d_a", data=float(self.parameters["fid_d_a"]))
            p.create_dataset("data_type", data=self.parameters["data_type"])

            # Data

            d = f.create_group("data")
            d.create_dataset("x", data=x)
            d.create_dataset("y", data=y)
            d.create_dataset("z", data=z)
            d.create_dataset("vx", data=vx)
            d.create_dataset("vy", data=vy)
            d.create_dataset("vz", data=vz)
            d.create_dataset("dx", data=dx)
            d.create_dataset("num_photons", data=n_ph)
            d.create_dataset("energy", data=e)

            f.close()

        comm.barrier()

    def project_photons(self, normal, sky_center, absorb_model=None,
                        nH=None, no_shifting=False, north_vector=None,
                        sigma_pos=None, kernel="top_hat", prng=None):
        r"""
        Projects photons onto an image plane given a line of sight.
        Returns a new :class:`~pyxsim.event_list.EventList`.

        Parameters
        ----------
        normal : character or array-like
            Normal vector to the plane of projection. If "x", "y", or "z", will
            assume to be along that axis (and will probably be faster). Otherwise,
            should be an off-axis normal vector, e.g [1.0, 2.0, -3.0]
        sky_center : array-like
            Center RA, Dec of the events in degrees.
        absorb_model : string or :class:`~pyxsim.spectral_models.AbsorptionModel`
            A model for foreground galactic absorption, to simulate the
            absorption of events before being detected. This cannot be applied
            here if you already did this step previously in the creation of the
            :class:`~pyxsim.photon_list.PhotonList` instance. Known options for
            strings are "wabs" and "tbabs".
        nH : float, optional
            The foreground column density in units of 10^22 cm^{-2}. Only used
            if absorption is applied.
        no_shifting : boolean, optional
            If set, the photon energies will not be Doppler shifted.
        north_vector : a sequence of floats
            A vector defining the "up" direction. This option sets the
            orientation of the plane of projection. If not set, an arbitrary
            grid-aligned north_vector is chosen. Ignored in the case where a
            particular axis (e.g., "x", "y", or "z") is explicitly specified.
        sigma_pos : float, optional
            Apply a gaussian smoothing operation to the sky positions of the
            events. This may be useful when the binned events appear blocky due
            to their uniform distribution within simulation cells. However, this
            will move the events away from their originating position on the
            sky, and so may distort surface brightness profiles and/or spectra.
            Should probably only be used for visualization purposes. Supply a
            float here to smooth with a standard deviation with this fraction
            of the cell size. Default: None
        kernel : string, optional
            The kernel used when smoothing positions of X-rays originating from
            SPH particles, "gaussian" or "top_hat". Default: "top_hat".
        prng : integer or :class:`~numpy.random.RandomState` object 
            A pseudo-random number generator. Typically will only be specified
            if you have a reason to generate the same set of random numbers,
            such as for a test. Default is to use the :mod:`numpy.random`
            module.

        Examples
        --------
        >>> L = np.array([0.1,-0.2,0.3])
        >>> events = my_photons.project_photons(L, [30., 45.])
        """
        prng = parse_prng(prng)

        scale_shift = -1.0/clight.to("km/s")

        if sigma_pos is not None and self.parameters["data_type"] == "particles":
            raise RuntimeError("The 'sigma_pos' argument should not be used with "
                               "particle-based datasets!")

        if isinstance(absorb_model, str):
            if absorb_model not in absorb_models:
                raise KeyError("%s is not a known absorption model!" % absorb_model)
            absorb_model = absorb_models[absorb_model]
        if absorb_model is not None:
            if nH is None:
                raise RuntimeError("You specified an absorption model, but didn't "
                                   "specify a value for nH!")
            absorb_model = absorb_model(nH)

        sky_center = YTArray(sky_center, "degree")

        n_ph = self.photons["num_photons"]

        if not isinstance(normal, str):
            L = np.array(normal)
            orient = Orientation(L, north_vector=north_vector)
            x_hat = orient.unit_vectors[0]
            y_hat = orient.unit_vectors[1]
            z_hat = orient.unit_vectors[2]
        else:
            x_hat = np.zeros(3)
            y_hat = np.zeros(3)
            z_hat = np.zeros(3)

        parameters = {}

        D_A = self.parameters["fid_d_a"]

        events = {}

        eobs = self.photons["energy"].v

        if not no_shifting:
            if comm.rank == 0:
                mylog.info("Doppler-shifting photon energies.")
            if isinstance(normal, str):
                shift = self.photons["vel"][:,"xyz".index(normal)]*scale_shift
            else:
                shift = np.dot(self.photons["vel"], z_hat)*scale_shift
            doppler_shift(shift, n_ph, eobs)

        if absorb_model is None:
            det = np.ones(eobs.size, dtype='bool')
            num_det = eobs.size
        else:
            if comm.rank == 0:
                mylog.info("Foreground galactic absorption: using "
                           "the %s model and nH = %g." % (absorb_model._name, nH))
            det = absorb_model.absorb_photons(eobs, prng=prng)
            num_det = det.sum()

        events["eobs"] = YTArray(eobs[det], "keV")

        num_events = comm.mpi_allreduce(num_det)

        if comm.rank == 0:
            mylog.info("%d events have been detected." % num_events)

        if num_det > 0:

            if comm.rank == 0:
                mylog.info("Assigning positions to events.")

            if isinstance(normal, str):
                norm = "xyz".index(normal)
            else:
                norm = normal

            xsky, ysky = scatter_events(norm, prng, kernel,
                                        self.parameters["data_type"],
                                        num_det, det, self.photons["num_photons"],
                                        self.photons["pos"].d, self.photons["dx"].d,
                                        x_hat, y_hat)

            if self.parameters["data_type"] == "cells" and sigma_pos is not None:
                if comm.rank == 0:
                    mylog.info("Optionally smoothing sky positions.")
                sigma = sigma_pos*np.repeat(self.photons["dx"].d, n_ph)[det]
                xsky += sigma * prng.normal(loc=0.0, scale=1.0, size=num_det)
                ysky += sigma * prng.normal(loc=0.0, scale=1.0, size=num_det)

            d_a = D_A.to("kpc").v
            xsky /= d_a
            ysky /= d_a

            if comm.rank == 0:
                mylog.info("Converting pixel to sky coordinates.")

            pixel_to_cel(xsky, ysky, sky_center)

        else:

            xsky = []
            ysky = []

        events["xsky"] = YTArray(xsky, "degree")
        events["ysky"] = YTArray(ysky, "degree")

        parameters["exp_time"] = self.parameters["fid_exp_time"]
        parameters["area"] = self.parameters["fid_area"]
        parameters["sky_center"] = sky_center

        return EventList(events, parameters)


class MultiPhotonList(object):

    def __init__(self, photon_lists):
        self.photon_lists = photon_lists
        self.num_lists = len(photon_lists)

    @classmethod
    def from_files(cls, basename):
        r"""
        Initialize a :class:`~pyxsim.photon_list.MultiPhotonList`
        from a set of HDF5 files with the prefix *basename*.
        """
        import glob
        photon_lists = []
        fns = glob.glob("{}.[0-9][0-9].h5".format(basename))
        fns.sort()
        for fn in fns:
            photons = PhotonList.from_file(fn)
            photon_lists.append(photons)
        return cls(photon_lists)

    @classmethod
    def from_data_source(cls, num_lists, data_source, redshift, area,
                         exp_time, source_model, point_sources=False,
                         parameters=None, center=None, dist=None,
                         cosmology=None, velocity_fields=None):
        r"""
        Initialize a :class:`~pyxsim.photon_list.MultiPhotonList` from 
        a yt data source. The redshift, collecting area, exposure time, 
        and cosmology are stored in the *parameters* dictionary which 
        is passed to the *source_model* function.

        Parameters
        ----------
        num_lists : integer
            The number of :class:`~pyxsim.photon_list.PhotonList` instances
            to generate.
        data_source : :class:`~yt.data_objects.data_containers.YTSelectionContainer`
            The data source from which the photons will be generated.
        redshift : float
            The cosmological redshift for the photons.
        area : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`
            The collecting area to determine the number of photons. If units are
            not specified, it is assumed to be in cm^2.
        exp_time : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`
            The exposure time to determine the number of photons. If units are
            not specified, it is assumed to be in seconds.
        source_model : :class:`~pyxsim.source_models.SourceModel`
            A source model used to generate the photons.
        point_sources : boolean, optional
            If True, the photons will be assumed to be generated from the exact
            positions of the cells or particles and not smeared around within
            a volume. Default: False
        parameters : dict, optional
            A dictionary of parameters to be passed for the source model to use,
            if necessary.
        center : string or array_like, optional
            The origin of the photon spatial coordinates. Accepts "c", "max", or
            a coordinate. If not specified, pyxsim attempts to use the "center"
            field parameter of the data_source.
        dist : float, (value, unit) tuple, :class:`~yt.units.yt_array.YTQuantity`, or :class:`~astropy.units.Quantity`, optional
            The angular diameter distance, used for nearby sources. This may be
            optionally supplied instead of it being determined from the
            *redshift* and given *cosmology*. If units are not specified, it is
            assumed to be in kpc. To use this, the redshift must be set to zero.
        cosmology : :class:`~yt.utilities.cosmology.Cosmology`, optional
            Cosmological information. If not supplied, we try to get the
            cosmology from the dataset. Otherwise, LCDM with the default yt
            parameters is assumed.
        velocity_fields : list of fields, optional
            The yt fields to use for the velocity. If not specified, the
            following will be assumed:
            ['velocity_x', 'velocity_y', 'velocity_z'] for grid datasets
            ['particle_velocity_x', 'particle_velocity_y', 'particle_velocity_z'] for particle datasets
        """
        photon_lists = []

        my_exp_time = exp_time/num_lists

        for i in range(num_lists):
            mylog.info("Generating photons for PhotonList %d." % i)
            photons = PhotonList.from_data_source(data_source, redshift,
                                                  area, my_exp_time, source_model,
                                                  point_sources=point_sources,
                                                  parameters=parameters,
                                                  center=center, dist=dist,
                                                  cosmology=cosmology,
                                                  velocity_fields=velocity_fields)
            photons.parameters["exp_time"] = exp_time
            photon_lists.append(photons)

        return cls(photon_lists)

    def write_h5_files(self, basename):
        """
        Write the :class:`~pyxsim.photon_list.MultiPhotonList`
        to a set of HDF5 files with prefix *basename*.
        """
        for i, photons in enumerate(self.photon_lists):
            photons.write_h5_file("%s.%02d.h5" % (basename, i))

    def project_photons(self, normal, sky_center, absorb_model=None,
                        nH=None, no_shifting=False, north_vector=None,
                        sigma_pos=None, kernel="top_hat", prng=None, **kwargs):
        r"""
        Projects photons onto an image plane given a line of sight.
        Returns a new :class:`~pyxsim.event_list.EventList`.

        Parameters
        ----------
        normal : character or array-like
            Normal vector to the plane of projection. If "x", "y", or "z", will
            assume to be along that axis (and will probably be faster). Otherwise,
            should be an off-axis normal vector, e.g [1.0, 2.0, -3.0]
        sky_center : array-like
            Center RA, Dec of the events in degrees.
        absorb_model : string or :class:`~pyxsim.spectral_models.AbsorptionModel`
            A model for foreground galactic absorption, to simulate the
            absorption of events before being detected. This cannot be applied
            here if you already did this step previously in the creation of the
            :class:`~pyxsim.photon_list.PhotonList` instance. Known options for
            strings are "wabs" and "tbabs".
        nH : float, optional
            The foreground column density in units of 10^22 cm^{-2}. Only used
            if absorption is applied.
        no_shifting : boolean, optional
            If set, the photon energies will not be Doppler shifted.
        north_vector : a sequence of floats
            A vector defining the "up" direction. This option sets the
            orientation of the plane of projection. If not set, an arbitrary
            grid-aligned north_vector is chosen. Ignored in the case where a
            particular axis (e.g., "x", "y", or "z") is explicitly specified.
        sigma_pos : float, optional
            Apply a gaussian smoothing operation to the sky positions of the
            events. This may be useful when the binned events appear blocky due
            to their uniform distribution within simulation cells. However, this
            will move the events away from their originating position on the
            sky, and so may distort surface brightness profiles and/or spectra.
            Should probably only be used for visualization purposes. Supply a
            float here to smooth with a standard deviation with this fraction
            of the cell size. Default: None
        kernel : string, optional
            The kernel used when smoothing positions of X-rays originating from
            SPH particles, "gaussian" or "top_hat". Default: "top_hat".
        prng : integer or :class:`~numpy.random.RandomState` object 
            A pseudo-random number generator. Typically will only be specified
            if you have a reason to generate the same set of random numbers,
            such as for a test. Default is to use the :mod:`numpy.random`
            module.

        Examples
        --------
        >>> L = np.array([0.1,-0.2,0.3])
        >>> events = my_photons.project_photons(L, [30., 45.])
        """
        event_lists = []

        for i, photons in enumerate(self.photon_lists):
            mylog.info("Projecting photons from PhotonList %d." % i)
            events = photons.project_photons(normal, sky_center,
                                             absorb_model=absorb_model,
                                             nH=nH, no_shifting=no_shifting,
                                             north_vector=north_vector,
                                             sigma_pos=sigma_pos,
                                             kernel=kernel, prng=prng, **kwargs)
            event_lists.append(events)

        return MultiEventList(event_lists)
