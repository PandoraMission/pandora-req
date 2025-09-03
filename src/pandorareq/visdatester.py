import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandorapsf as pp
import pandorasat as ps
from astropy import __version__ as __astropyversion__
from matplotlib.backends.backend_pdf import PdfPages

from . import __version__, config

REQUIRED_MAG = float(config["VISDA_REQUIREMENTS"]["required_mag"])
REQUIRED_TEFF = float(config["VISDA_REQUIREMENTS"]["required_teff"])
REQUIRED_SNR = float(config["VISDA_REQUIREMENTS"]["required_snr"])
REQUIRED_COADDS = int(config["VISDA_REQUIREMENTS"]["required_coadds"])
REQUIRED_TIME = int(config["VISDA_REQUIREMENTS"]["required_time"])


class VISDATester(object):
    def __init__(
        self,
        background_rate=None,
        dark=None,
        readnoise=None,
    ):
        # CBE of all the detector properties
        self.VISDA = ps.VisibleDetector()
        self.background_rate = background_rate
        self.dark = dark
        self.readnoise = readnoise

        for attr in [
            "background_rate",
            "dark",
            "readnoise",
        ]:
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(self.VISDA, attr))
            setattr(
                self,
                attr,
                u.Quantity(getattr(self, attr), getattr(self.VISDA, attr).unit),
            )
        self.psf = pp.PSF.from_name("visda_fallback")
        self.psf.extrapolate = True

        return

    def __repr__(self):
        return "VISDA Test Object:\n\t" + "\n\t".join(
            [
                f"{attr}: {getattr(self, attr)}"
                for attr in [
                    "background_rate",
                    "dark",
                    "readnoise",
                ]
            ]
        )

    @property
    def text_info(self):
        characteristics = "\n".join(
            [
                f"{attr}: {getattr(self, attr)}"
                for attr in [
                    "background_rate",
                    "dark",
                    "readnoise",
                ]
            ]
        )
        # Define the text content (Package Versions + Any Other Info)
        text_info = f"""
        Pandora Data Processing Center
        VISDA PSF Requirements Test Report
        ----------------------------------
        
        Package Versions:
        - pandora_requirements: {__version__}
        - pandorapsf: {pp.__version__}
        - pandorasat: {ps.__version__}
        - astropy: {__astropyversion__}

        Detector Characteristics:
        {characteristics}

        Test Requriements:
        - Target Test V band Magnitude: {REQUIRED_MAG}
        - Target Test T$_{{eff}}$: {REQUIRED_TEFF} K
        - Required Time on Target: {REQUIRED_TIME}s
        - Required SNR: {REQUIRED_SNR}

        Additional Information:
        - This document contains test results for a target star on the VISDA.
        - Generated using automated tests and visualizations.
        - This test uses CBE for all Pandora parameters
        - This test assumes no jitter motion between frames.

        """
        return text_info

    def get_page_one(self):
        # Create a figure for the first page
        fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard page size

        # Hide axes
        ax.axis("off")

        # Add text to the figure
        ax.text(
            0.5,
            0.5,
            self.text_info,
            fontsize=12,
            ha="center",
            va="center",
            wrap=True,
        )
        return fig

    def make_report(
        self,
        pdf_filename: str = "VISDA_requirements_test.pdf",
    ):
        # # Output plots
        # resolution_fig = self.test_resolution(return_plot=True)
        # all_resolutions_fig = self.make_resolution_array(return_plot=True)
        snr_figs = self.test_SNR(return_plot=True)

        with PdfPages(pdf_filename) as pdf:
            fig = self.get_page_one()
            # Save the first page
            pdf.savefig(fig)
            plt.close(fig)

            # fig = self.show_trace()
            # pdf.savefig(fig)
            # plt.close(fig)

            # fig = self.show_trace_1D()
            # pdf.savefig(fig)
            # plt.close(fig)

            # pdf.savefig(resolution_fig)
            # plt.close(resolution_fig)

            for fig in snr_figs:
                pdf.savefig(fig)
                plt.close(fig)

            # pdf.savefig(all_resolutions_fig)
            # plt.close(all_resolutions_fig)

    def test_SNR(self, return_plot=False):
        wav, spec = ps.phoenix.load_benchmark()
        background_signal = (
            self.background_rate * REQUIRED_TIME * u.s + self.dark * REQUIRED_TIME * u.s
        )
        signal = (
            np.trapz(spec * self.VISDA.sensitivity(wav), wav)
            * REQUIRED_TIME
            * u.s
            * self.psf.prf(0, 0)[2]
        )
        dmags = np.round(np.arange(5, 20, 0.1) - REQUIRED_MAG, 4)

        s = np.argsort(signal.ravel())[::-1]
        brightest_pixel, SNR = np.zeros((2, len(dmags)))
        for mdx, dmag in enumerate(dmags):
            rflux = 10 ** (dmag / -2.5)
            ar = signal.copy() * rflux
            npixels = np.cumsum(np.ones(np.prod(ar.shape)))
            flux = np.cumsum(ar.ravel()[s])
            photon_noise = np.sqrt(flux.value) * flux.unit
            background_noise_in_aperture = (
                np.sqrt(npixels * background_signal.value) * u.electron
            )
            readnoise_in_aperture = (
                np.sqrt(((REQUIRED_TIME * u.s) / self.VISDA.integration_time).value)
                * self.readnoise
                * npixels
                * u.pixel
            )
            noise = np.sqrt(
                background_noise_in_aperture**2
                + photon_noise**2
                + readnoise_in_aperture**2
            )
            SNR[mdx] = (flux / noise).max()
            brightest_pixel[mdx] = (
                (
                    (ar / ((REQUIRED_TIME * u.s) / self.VISDA.integration_time))
                    .max()
                    .value
                    * u.electron
                )
                / self.VISDA.gain
            ).value
        if return_plot:
            # Brightest pixel
            fig1, ax = plt.subplots()
            ax.plot(dmags + REQUIRED_MAG, brightest_pixel, c="k")
            ax.axhline(2**16 * 0.95, c="r", ls="--")
            ax.set(
                xlabel="V Mag",
                ylabel=f"Brightest Pixel Value in one {self.VISDA.integration_time} integration [DN]",
                ylim=(0, 80000),
                title="Brightest Pixel in 24 integrations vs. Magnitude",
            )
            plt.tight_layout()

            # SNR
            k = dmags > -3
            fig2, ax = plt.subplots()
            ax.plot(dmags[k] + REQUIRED_MAG, SNR[k], c="k")
            # ax.plot(dmags + REQUIRED_MAG, np.nanmax(phot_SNR, axis=0), c="k", ls="--")
            ax.scatter(REQUIRED_MAG, REQUIRED_SNR, c="r", marker="*", s=100)
            ax.set(
                xlabel="V Mag",
                ylabel=f"Theoretical SNR in {REQUIRED_TIME}s",
                ylim=(0, 10000),
                title="Signal to Noise Ratio vs. Magnitude",
            )
            plt.tight_layout()
            return fig1, fig2
        return SNR[(dmags + REQUIRED_MAG) == REQUIRED_MAG][0] > REQUIRED_SNR
