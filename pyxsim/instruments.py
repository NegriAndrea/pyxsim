import numpy as np
from pyxsim.responses import AuxiliaryResponseFile, \
    RedistributionMatrixFile
from pyxsim.utils import mylog
from yt.funcs import get_pbar, ensure_numpy_array, \
    iterable
from yt.units.yt_array import YTQuantity, YTArray
from yt.utilities.on_demand_imports import _astropy

class InstrumentSimulator(object):
    def __init__(self, dtheta, nx, psf, arf,
                 rmf):
        self.dtheta = dtheta
        self.nx = nx
        self.psf = psf
        self.arf = AuxiliaryResponseFile(arf, rmffile=rmf)
        self.rmf = RedistributionMatrixFile(rmf)

    def __call__(self, events, reblock=True,
                 convolve_psf=True, convolve_arf=True, 
                 convolve_rmf=True, prng=None):
        if prng is None:
            prng = np.random
        if reblock:
            self.reblock(events)
        if convolve_psf:
            self.convolve_with_psf(events, prng)
        if convolve_arf:
            self.apply_effective_area(events, prng)
        if convolve_rmf:
            self.convolve_energies(events, prng)
        return events

    def reblock(self, events):
        """
        Reblock events to a new binning with the same celestial
        coordinates.
        """
        new_wcs = _astropy.pywcs.WCS(naxis=2)
        new_wcs.wcs.crval = events.parameters["sky_center"].d
        new_wcs.wcs.crpix = np.array([0.5*(self.nx+1)]*2)
        new_wcs.wcs.cdelt = [-self.dtheta, self.dtheta]
        new_wcs.wcs.ctype = ["RA---TAN","DEC--TAN"]
        new_wcs.wcs.cunit = ["deg"]*2
        xpix, ypix = new_wcs.wcs_world2pix(events["xsky"], events["ysky"], 1)
        events.events['xpix'] = xpix
        events.events['ypix'] = ypix
        xsky, ysky = new_wcs.wcs_pix2world(events["xpix"], events["ypix"], 1)
        events.events['xsky'] = xsky
        events.events['ysky'] = ysky
        events.parameters['pix_center'] = new_wcs.wcs.crpix[:]
        events.parameters['dtheta'] = YTQuantity(self.dtheta, "deg")
        events.wcs = new_wcs

    def convolve_with_psf(self, events, prng):
        r"""
        Convolve the event positions with a PSF.
        """
        if isinstance(self.psf, float):
            scale = self.psf
            psf = lambda n: prng.normal(scale=scale, size=n)
        events.events["xsky"] += psf(events.num_events)
        events.events["ysky"] += psf(events.num_events)
        xpix, ypix = events.wcs.wcs_world2pix(events["xsky"], events["ysky"], 1)
        events.events['xpix'] = xpix
        events.events['ypix'] = ypix

    def apply_effective_area(self, events, prng):
        """
        Convolve the events with a ARF file.
        """
        mylog.info("Applying energy-dependent effective area.")
        detected = self.arf.detect_events(events["eobs"], prng=prng)
        for key in ["xpix", "ypix", "xsky", "ysky", "eobs"]:
            events.events[key] = events[key][detected]

    def convolve_energies(self, events, prng):
        """
        Convolve the events with a RMF file.
        """
        mylog.info("Reading response matrix file (RMF): %s" % self.rmf.filename)

        elo = self.rmf.data["ENERG_LO"]
        ehi = self.rmf.data["ENERG_HI"]
        n_de = elo.shape[0]
        mylog.info("Number of energy bins in RMF: %d" % n_de)
        mylog.info("Energy limits: %g %g" % (min(elo), max(ehi)))

        n_ch = len(self.rmf.ebounds["CHANNEL"])
        mylog.info("Number of channels in RMF: %d" % n_ch)

        eidxs = np.argsort(events["eobs"])
        sorted_e = events["eobs"][eidxs].d

        detectedChannels = []

        # run through all photon energies and find which bin they go in
        fcurr = 0
        last = sorted_e.shape[0]

        pbar = get_pbar("Scattering energies with RMF", last)

        for (k, low), high in zip(enumerate(elo), ehi):
            # weight function for probabilities from RMF
            weights = np.nan_to_num(np.float64(self.rmf.data["MATRIX"][k]))
            weights /= weights.sum()
            # build channel number list associated to array value,
            # there are groups of channels in rmfs with nonzero probabilities
            trueChannel = []
            f_chan = ensure_numpy_array(np.nan_to_num(self.rmf.data["F_CHAN"][k]))
            n_chan = ensure_numpy_array(np.nan_to_num(self.rmf.data["N_CHAN"][k]))
            if not iterable(f_chan):
                f_chan = [f_chan]
                n_chan = [n_chan]
            for start, nchan in zip(f_chan, n_chan):
                if nchan == 0:
                    trueChannel.append(start)
                else:
                    trueChannel += list(range(start, start+nchan))
            if len(trueChannel) > 0:
                for q in range(fcurr, last):
                    if low <= sorted_e[q] < high:
                        channelInd = prng.choice(len(weights), p=weights)
                        fcurr += 1
                        pbar.update(fcurr)
                        detectedChannels.append(trueChannel[channelInd])
                    else:
                        break
        pbar.finish()

        for key in ["xpix", "ypix", "xsky", "ysky"]:
            events.events[key] = events[key][eidxs]

        events.events["eobs"] = YTArray(sorted_e, "keV")
        events.events[self.rmf.header["CHANTYPE"]] = np.array(detectedChannels, dtype="int")

        events.parameters["RMF"] = self.rmf.filename
        events.parameters["ChannelType"] = self.rmf.header["CHANTYPE"]
        events.parameters["Telescope"] = self.rmf.header["TELESCOP"]
        events.parameters["Instrument"] = self.rmf.header["INSTRUME"]
        events.parameters["Mission"] = self.rmf.header.get("MISSION","")

ACIS_S = InstrumentSimulator(0.0001366666666, 8192, 0.0,
                             "aciss_aimpt_cy17.arf",
                             "aciss_aimpt_cy17.rmf")
ACIS_I = InstrumentSimulator(0.0001366666666, 8192, 0.0,
                             "acisi_aimpt_cy17.arf",
                             "acisi_aimpt_cy17.rmf")
Hitomi_SXS = InstrumentSimulator(0.0, 6, 0.0, 
                                 "sxt-s_140505_ts02um_intallpxl.arf",
                                 "ah_sxs_5ev_20130806.rmf")
XRS_Imager = InstrumentSimulator(9.167325E-05, 4096, 0.0,
                                 "xrs_hdxi.arf", "xrs_hdxi.rmf")
XRS_Calorimeter = InstrumentSimulator(0.0002864789, 300, 0.0,
                                      "xrs_calorimeter.arf", 
                                      "xrs_calorimeter.rmf")