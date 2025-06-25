import astropy.units as u
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
import numpy as np
import pandorapsf as pp
import pandorasat as ps
from astropy import __version__ as __astropyversion__
from matplotlib.backends.backend_pdf import PdfPages

from . import __version__, config

REQUIRED_MAG = float(config["NIRDA_REQUIREMENTS"]["required_mag"])
REQUIRED_TEFF = float(config["NIRDA_REQUIREMENTS"]["required_teff"])
REQUIRED_SNR = float(config["NIRDA_REQUIREMENTS"]["required_snr"])
REQUIRED_R = float(config["NIRDA_REQUIREMENTS"]["required_r"])
REQUIRED_INTS = int(config["NIRDA_REQUIREMENTS"]["required_ints"])
LAM = (float(config["NIRDA_REQUIREMENTS"]["lam"]) * u.nm).to(u.micron)
NFOWLER = int(config["NIRDA_REQUIREMENTS"]["nfowler"])
NFOWLER_GROUPS = int(config["NIRDA_REQUIREMENTS"]["nfowler_groups"])
NFRAMES = int(config["NIRDA_REQUIREMENTS"]["nframes"])


class NIRDATester(object):
    def __init__(
        self,
        zodiacal_background_rate=None,
        stray_light_rate=None,
        thermal_background_rate=None,
        dark_rate=None,
        read_noise=None,
        throughput=1,
    ):
        # CBE of all the detector properties
        self.NIRDA = ps.NIRDetector()
        self.zodiacal_background_rate = zodiacal_background_rate
        self.stray_light_rate = stray_light_rate
        self.thermal_background_rate = thermal_background_rate
        self.dark_rate = dark_rate
        self.read_noise = read_noise
        self.frame_time = self.NIRDA.frame_time((400, 80))
        self.throughput = throughput

        for attr in [
            "zodiacal_background_rate",
            "stray_light_rate",
            "thermal_background_rate",
            "dark_rate",
            "read_noise",
        ]:
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(self.NIRDA, attr))
            setattr(
                self,
                attr,
                u.Quantity(getattr(self, attr), getattr(self.NIRDA, attr).unit),
            )
        self.throughput = u.Quantity(self.throughput, "")

        self.psf = pp.PSF.from_name("nirda_fallback")
        self.psf.extrapolate = True

        self.ts = pp.TraceScene(np.asarray([(300, 40)]), psf=self.psf)

        dy = LAM / REQUIRED_R

        self.microns_per_pixel = np.interp(
            LAM.to(u.micron).value,
            self.psf.trace_wavelength.value,
            np.gradient(self.psf.trace_wavelength.value, self.psf.trace_pixel.value),
        )
        self.pix_det, self.wav_det = (
            self.psf.trace_pixel + 300 * u.pixel,
            self.psf.trace_wavelength,
        )

        self.bounds = (
            int(np.floor(self.wav2pix(LAM - dy / 2).value)),
            int(np.ceil(self.wav2pix(LAM + dy / 2).value)),
        )

        return

    def __repr__(self):
        return "NIRDA Test Object:\n\t" + "\n\t".join(
            [
                f"{attr}: {getattr(self, attr)}"
                for attr in [
                    "zodiacal_background_rate",
                    "stray_light_rate",
                    "thermal_background_rate",
                    "dark_rate",
                    "read_noise",
                    "throughput",
                ]
            ]
        )

    @property
    def text_info(self):
        characteristics = "\n".join(
            [
                f"{attr}: {getattr(self, attr)}"
                for attr in [
                    "zodiacal_background_rate",
                    "stray_light_rate",
                    "thermal_background_rate",
                    "dark_rate",
                    "read_noise",
                ]
            ]
        )
        # Define the text content (Package Versions + Any Other Info)
        text_info = f"""
        Pandora Data Processing Center
        NIRDA PSF Requirements Test Report
        ----------------------------------
        
        Package Versions:
        - pandora_requirements: {__version__}
        - pandorapsf: {pp.__version__}
        - pandorasat: {ps.__version__}
        - astropy: {__astropyversion__}

        Detector Characteristics:
        {characteristics}

        Test Requriements:
        - Target Test j band Magnitude: {REQUIRED_MAG}
        - Target Test T$_{{eff}}$: {REQUIRED_TEFF} K
        - Required Number of Integrations: {REQUIRED_INTS}
        - Required SNR: {REQUIRED_SNR}
        - Required Wavelength: {LAM} micron
        - Required R: {REQUIRED_R}

        Additional Information:
        - This document contains test results for a target star on the NIRDA.
        - Generated using automated tests and visualizations.
        - This test uses CBE for all Pandora parameters
        - This test assumes no jitter motion between frames.

        """
        return text_info

    def wav2pix(self, x):
        return np.interp(x, self.wav_det, self.pix_det)

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

    def show_trace(self):
        wav, spec = ps.utils.load_benchmark()
        spec_det = self.ts.integrate_spectrum(wav, spec)
        ar = self.ts.model((spec_det[:, None] * self.frame_time).to(u.electron))[0]

        # Trace
        fig, ax = plt.subplots()
        im = ax.imshow(ar.value, vmin=0, vmax=10, origin="lower")
        ax.set(xlabel="Column Pixel", ylabel="Row Pixel")
        plt.axhline(self.bounds[0], c="r", ls="--", lw=1)
        plt.axhline(self.bounds[1], c="r", ls="--", lw=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Electrons Per Frame")
        return fig

    def show_trace_1D(self):
        wav, spec = ps.utils.load_benchmark()
        spec_det = self.ts.integrate_spectrum(wav, spec)
        ar = self.ts.model((spec_det[:, None] * self.frame_time).to(u.electron))[0]

        # Trace summed in each axis
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(ar.sum(axis=0).value, c="k")
        axs[0].set(xlabel="Column Pixel", ylabel="Flux Summed over Row [e$^-$]")

        ax = axs[1]
        ax.plot(ar.sum(axis=1).value, c="k")
        ax.set(xlabel="Row Pixel", ylabel="Flux Summed over Column [e$^-$]")
        plt.subplots_adjust(wspace=0.5)

        ax.spines[["right", "top"]].set_visible(True)
        ax_p = ax.twiny()
        ax_p.set(xticks=ax.get_xticks(), xlim=ax.get_xlim())
        ax_p.set_xlabel(xlabel="Wavelength [micron]", color="grey")
        ax_p.set_xticklabels(
            labels=list(
                np.round(
                    np.interp(
                        ax.get_xticks(),
                        self.pix_det.value,
                        self.wav_det.value,
                    ),
                    2,
                )
            ),
            rotation=45,
            color="grey",
        )
        plt.tight_layout()
        return fig

    def _get_measured_r(self, lam):

        psf = self.psf.psf(wavelength=lam)
        psr = self.psf.psf_row.value

        # get the column-averaged prf, normalize the average
        psf_slice = psf.mean(axis=1)
        psf_slice /= psf_slice.sum()

        # make sure the row array has the slice centered
        psr -= np.average(psr, weights=psf_slice)

        # split the PRF into two halves, left and right
        s1 = np.where(psr < 0)[0][::-1]
        s2 = np.where(psr > 0)[0]

        # width of the PSF by averaging the left and right half
        width = np.sqrt(2) * np.mean(
            [
                np.interp(0.34, np.cumsum(psf_slice[s2]), psr[s2]),
                np.interp(0.34, np.cumsum(psf_slice[s1]), -psr[s1]),
            ]
        )
        measured_R = lam.to(u.micron).value / (width * self.microns_per_pixel)
        return measured_R

    def test_resolution(self, return_plot=False):

        measured_R = self._get_measured_r(LAM)
        width = (
            ((LAM / measured_R) / (self.microns_per_pixel * u.micron / u.pixel))
            .to(u.pixel)
            .value
        )

        ts_short = pp.TraceScene(
            np.asarray(
                [
                    (self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2, 40),
                    (
                        self.bounds[0] + (self.bounds[1] - self.bounds[0] + width) / 2,
                        40,
                    ),
                ]
            ),
            psf=self.psf,
        )
        spec_one = np.zeros((ts_short.nwav, 2))
        spec_one[400, 0] = 1
        spec_two = np.zeros((ts_short.nwav, 2))
        spec_two[400, :] = 1

        if return_plot:
            # PRF single + double
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(ts_short.model(spec_one)[0], vmin=0, vmax=0.1)
            axs[0].set(
                ylim=(230, 280),
                title="Single Line",
                xlabel="Subarray Column",
                ylabel="Subarray Row\n(Spectral Dimension)",
                aspect="equal",
            )
            axs[1].imshow(ts_short.model(spec_two)[0], vmin=0, vmax=0.1)
            axs[1].set(
                ylim=(230, 280),
                title=f"Double Line, Resolved at R={int(np.round(measured_R))}\n({np.round(width, 2)} pixels)",
                xlabel="Subarray Column",
            )
            return fig
        return measured_R > REQUIRED_R

    def make_resolution_array(
        self, lam0=0.875 * u.micron, lam1=1.630 * u.micron, lamsteps=200, return_plot=False
    ):
        lams = np.linspace(lam0, lam1, lamsteps)

        lam_arr = np.array([self._get_measured_r(lam) for lam in lams])

        if return_plot:
            fig, ax = plt.subplots()
            kernel = Box1DKernel(width=10)
            y_smooth = convolve(lam_arr, kernel, boundary='extend')
            ax.scatter(lams, lam_arr, s=8, color='r', label="Modeled resolution")
            ax.plot(lams, y_smooth, label='Smoothed', linewidth=2)
            ax.set_xlabel("Wavelength(micron)")
            ax.set_ylabel(r"Spectral resolution")
            ax.legend()
            ax.set_title("NIRDA Spectral Resolution")
            ax.grid()
            plt.tight_layout()
            return fig
        return np.array([lams, lam_arr])

    def test_SNR(self, return_plot=False):
        zodi = self.zodiacal_background_rate * (NFRAMES - NFOWLER) * self.frame_time
        stray_light = self.stray_light_rate * (NFRAMES - NFOWLER) * self.frame_time
        thermal = self.thermal_background_rate * (NFRAMES - NFOWLER) * self.frame_time
        dark = self.dark_rate * (NFRAMES - NFOWLER) * self.frame_time

        background_signal = zodi + stray_light + thermal + dark

        # mags = np.arange(5, 14, 0.5)
        dmags = np.arange(5, 14, 0.5) - REQUIRED_MAG
        SNR = np.zeros((40, len(dmags)))
        phot_SNR = np.zeros((40, len(dmags)))
        brightest_pixel = np.zeros(len(dmags))
        wav, spec = ps.utils.load_benchmark()
        spec_det = self.ts.integrate_spectrum(wav, spec) * self.throughput
        for mdx, dmag in enumerate(dmags):
            rflux = 10 ** (dmag / -2.5)
            # this is the total image per frame
            ar = self.ts.model(
                (NFRAMES - NFOWLER)
                * (rflux * spec_det[:, None] * self.frame_time).to(u.electron)
            )[0]

            # Trace shape on detector
            th = ar.max(axis=0)
            s = np.argsort(th)[::-1]
            for npix in np.arange(1, len(SNR)):
                aper = np.in1d(th, th[s[:npix]])
                flux = ar[self.bounds[0] : self.bounds[1], aper].sum()
                photon_noise = np.sqrt(flux.value) * flux.unit

                npixels = (self.bounds[1] - self.bounds[0]) * aper.sum() * u.pixel

                background_noise_in_aperture = (
                    np.sqrt(npixels.value * background_signal.value) * u.electron
                )
                read_noise = (
                    np.sqrt(
                        (
                            NFOWLER
                            * NFOWLER_GROUPS
                            * self.read_noise.value**2
                            * npixels.value
                        )
                    )
                    * u.electron
                )
                # total noise sources
                noise = np.sqrt(
                    background_noise_in_aperture**2 + photon_noise**2 + read_noise**2
                )
                SNR[npix, mdx] = (REQUIRED_INTS / np.sqrt(REQUIRED_INTS)) * flux / noise
                phot_SNR[npix, mdx] = (
                    (REQUIRED_INTS / np.sqrt(REQUIRED_INTS)) * flux / photon_noise
                )

            brightest_pixel[mdx] = np.nanmax(ar.value)

        if return_plot:
            # Brightest pixel
            fig1, ax = plt.subplots()
            ax.plot(dmags + REQUIRED_MAG, brightest_pixel, c="k")
            ax.set(
                xlabel="J Mag",
                ylabel="Brightest Pixel Value [electrons]",
                ylim=(0, 80000),
                title="Brightest Pixel in 24 integrations vs. Magnitude",
            )
            plt.tight_layout()

            # SNR
            fig2, ax = plt.subplots()
            ax.plot(dmags + REQUIRED_MAG, np.nanmax(SNR, axis=0), c="k")
            ax.plot(dmags + REQUIRED_MAG, np.nanmax(phot_SNR, axis=0), c="k", ls="--")
            ax.scatter(REQUIRED_MAG, REQUIRED_SNR, c="r", marker="*", s=100)
            ax.set(
                xlabel="J Mag",
                ylabel="Theoretical SNR",
                ylim=(0, 17000),
                title="Signal to Noise Ratio vs. Magnitude",
            )
            plt.tight_layout()

            # SNR
            fig3, ax = plt.subplots()
            ax.plot(
                SNR[:, (dmags + REQUIRED_MAG) == REQUIRED_MAG], c="k", label="jmag = 9"
            )
            ax.set(
                xlabel="Box width [pixels]",
                ylabel="Theoretical SNR",
                ylim=(0, 11000),
                title="Signal to Noise Ratio vs. Aperture Size",
            )
            ax.axhline(REQUIRED_SNR, c="r", ls="--")
            ax.legend()
            plt.tight_layout()
            return fig1, fig2, fig3
        return (SNR[:, (dmags + REQUIRED_MAG) == REQUIRED_MAG] > REQUIRED_SNR).any()

    def make_report(
        self,
        pdf_filename: str = "NIRDA_requirements_test.pdf",
    ):
        # Output plots
        resolution_fig = self.test_resolution(return_plot=True)
        all_resolutions_fig = self.make_resolution_array(return_plot=True)
        snr_figs = self.test_SNR(return_plot=True)

        with PdfPages(pdf_filename) as pdf:
            fig = self.get_page_one()
            # Save the first page
            pdf.savefig(fig)
            plt.close(fig)

            fig = self.show_trace()
            pdf.savefig(fig)
            plt.close(fig)

            fig = self.show_trace_1D()
            pdf.savefig(fig)
            plt.close(fig)

            pdf.savefig(resolution_fig)
            plt.close(resolution_fig)

            for fig in snr_figs:
                pdf.savefig(fig)
                plt.close(fig)

            pdf.savefig(all_resolutions_fig)
            plt.close(all_resolutions_fig)

    # def test_psf(self):
    #     # Test width
    #     psr, _, psf_slice = self.psf.prf(
    #         row=self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2,
    #         column=0,
    #         wavelength=LAM,
    #     )
    #     psr = psr.astype(float)
    #     psr -= self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2

    #     psf_slice = psf_slice.mean(axis=1)
    #     psf_slice /= psf_slice.sum()

    #     psr -= np.average(psr, weights=psf_slice)

    #     s1 = np.where(psr < 0)[0][::-1]
    #     s2 = np.where(psr > 0)[0]
    #     width = np.sqrt(2) * np.mean(
    #         [
    #             np.interp(0.34, np.cumsum(psf_slice[s2]), psr[s2]),
    #             np.interp(0.34, np.cumsum(psf_slice[s1]), -psr[s1]),
    #         ]
    #     )
    #     measured_R = LAM.to(u.micron).value / (width * self.microns_per_pixel)

    #     ts_short = pp.TraceScene(
    #         np.asarray(
    #             [
    #                 (self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2, 40),
    #                 (
    #                     self.bounds[0] + (self.bounds[1] - self.bounds[0] + width) / 2,
    #                     40,
    #                 ),
    #             ]
    #         ),
    #         psf=self.psf,
    #     )
    #     spec_one = np.zeros((ts_short.nwav, 2))
    #     spec_one[400, 0] = 1
    #     spec_two = np.zeros((ts_short.nwav, 2))
    #     spec_two[400, :] = 1

    #     # Test SNR
    #     # this is the total image per frame

    #     # SIGNAL
    #     zodi = self.zodiacal_background_rate * (NFRAMES - NFOWLER) * self.frame_time
    #     # SIGNAL
    #     stray_light = self.stray_light_rate * (NFRAMES - NFOWLER) * self.frame_time
    #     # SIGNAL
    #     thermal = self.thermal_background_rate * (NFRAMES - NFOWLER) * self.frame_time
    #     # SIGNAL
    #     dark = self.dark_rate * (NFRAMES - NFOWLER) * self.frame_time

    #     background_signal = zodi + stray_light + thermal + dark
    #     # background_error = np.sqrt(background_signal.value) * background_signal.unit

    #     mags = np.arange(5, 14, 0.5)
    #     SNR = np.zeros((40, len(mags)))
    #     phot_SNR = np.zeros((40, len(mags)))
    #     brightest_pixel = np.zeros(len(mags))
    #     for mdx, mag in enumerate(mags):
    #         wav, spec = ps.utils.SED(teff=REQUIRED_TEFF, jmag=mag)
    #         spec_det = self.ts.integrate_spectrum(wav, spec) * self.throughput
    #         # this is the total image per frame
    #         ar = self.ts.model(
    #             (NFRAMES - NFOWLER)
    #             * (spec_det[:, None] * self.frame_time).to(u.electron)
    #         )[0]

    #         # # Error on the measurement of the background signal
    #         # bkg_aper = ar.value < (0.5 * detector_signal.value)
    #         # nbkg_pixels = (bkg_aper).sum()
    #         # bkg_measurement_err = (detector_noise) / np.sqrt(nbkg_pixels) * u.pixel

    #         # Trace shape on detector
    #         th = ar.max(axis=0)
    #         s = np.argsort(th)[::-1]
    #         for npix in np.arange(1, len(SNR)):
    #             aper = np.in1d(th, th[s[:npix]])

    #             # For the integration
    #             flux = ar[self.bounds[0] : self.bounds[1], aper].sum()
    #             photon_noise = np.sqrt(flux.value) * flux.unit

    #             npixels = (self.bounds[1] - self.bounds[0]) * aper.sum() * u.pixel
    #             background_noise_in_aperture = (
    #                 np.sqrt(npixels.value * background_signal.value) * u.electron
    #             )
    #             read_noise = (
    #                 np.sqrt(
    #                     (
    #                         NFOWLER
    #                         * NFOWLER_GROUPS
    #                         * self.read_noise.value**2
    #                         * npixels.value
    #                     )
    #                 )
    #                 * u.electron
    #             )

    #             noise = np.sqrt(
    #                 background_noise_in_aperture**2 + photon_noise**2 + read_noise**2
    #             )
    #             SNR[npix, mdx] = (REQUIRED_INTS / np.sqrt(REQUIRED_INTS)) * flux / noise
    #             phot_SNR[npix, mdx] = (
    #                 (REQUIRED_INTS / np.sqrt(REQUIRED_INTS)) * flux / photon_noise
    #             )

    #         brightest_pixel[mdx] = np.nanmax(ar.value)

    #     # for plotting purposes we remake just one frame
    #     wav, spec = ps.utils.get_phoenix_model(REQUIRED_TEFF, jmag=REQUIRED_MAG)
    #     spec_det = self.ts.integrate_spectrum(wav, spec)
    #     # this is the total image per frame
    #     ar = self.ts.model((spec_det[:, None] * self.frame_time).to(u.electron))[0]

    #     # Output plots
    #     with PdfPages("out.pdf") as pdf:
    #         fig = self.get_page_one()
    #         # Save the first page
    #         pdf.savefig(fig)
    #         plt.close(fig)

    #         # PRF
    #         fig, ax = plt.subplots()
    #         r, c, y = self.psf.prf(
    #             row=self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2,
    #             column=0,
    #             wavelength=LAM,
    #         )
    #         im = ax.pcolormesh(c, r, 100 * y, vmin=0, vmax=10)
    #         plt.gca().set_aspect("equal")
    #         ax.set(
    #             xlabel="Column Position",
    #             ylabel="Row position",
    #             title=f"PRF at $\lambda$={LAM.to(u.micron).value}$\mu m$",
    #         )
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label("Flux [%]")
    #         pdf.savefig()
    #         plt.close()

    #         # PRF single + double
    #         fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #         axs[0].imshow(ts_short.model(spec_one)[0], vmin=0, vmax=0.1)
    #         axs[0].set(
    #             ylim=(230, 280),
    #             title="Single Line",
    #             xlabel="Subarray Column",
    #             ylabel="Subarray Row\n(Spectral Dimension)",
    #             aspect="equal",
    #         )
    #         axs[1].imshow(ts_short.model(spec_two)[0], vmin=0, vmax=0.1)
    #         axs[1].set(
    #             ylim=(230, 280),
    #             title=f"Double Line, Resolved at R={int(np.round(measured_R))}\n({np.round(width, 2)} pixels)",
    #             xlabel="Subarray Column",
    #         )
    #         pdf.savefig()
    #         plt.close()

    #         # Trace
    #         fig, ax = plt.subplots()
    #         im = ax.imshow(ar.value, vmin=0, vmax=10, origin="lower")
    #         ax.set(xlabel="Column Pixel", ylabel="Row Pixel")
    #         plt.axhline(self.bounds[0], c="r", ls="--", lw=1)
    #         plt.axhline(self.bounds[1], c="r", ls="--", lw=1)
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label("Electrons Per Frame")
    #         pdf.savefig()
    #         plt.close()

    #         # Trace summed in each axis
    #         fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #         im = axs[0].plot(ar.sum(axis=0).value, c="k")
    #         axs[0].set(xlabel="Column Pixel", ylabel="Flux Summed over Row [e$^-$]")

    #         ax = axs[1]
    #         im = ax.plot(ar.sum(axis=1).value, c="k")
    #         ax.set(xlabel="Row Pixel", ylabel="Flux Summed over Column [e$^-$]")
    #         plt.subplots_adjust(wspace=0.5)

    #         ax.spines[["right", "top"]].set_visible(True)
    #         ax_p = ax.twiny()
    #         ax_p.set(xticks=ax.get_xticks(), xlim=ax.get_xlim())
    #         ax_p.set_xlabel(xlabel="Wavelength [micron]", color="grey")
    #         ax_p.set_xticklabels(
    #             labels=list(
    #                 np.round(
    #                     np.interp(
    #                         ax.get_xticks(),
    #                         self.pix_det.value,
    #                         self.wav_det.value,
    #                     ),
    #                     2,
    #                 )
    #             ),
    #             rotation=45,
    #             color="grey",
    #         )
    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()

    #         # Brightest pixel
    #         fig, ax = plt.subplots()
    #         ax.plot(mags, brightest_pixel, c="k")
    #         ax.set(
    #             xlabel="J Mag",
    #             ylabel="Brightest Pixel Value [electrons]",
    #             ylim=(0, 80000),
    #             title="Brightest Pixel in 24 integrations vs. Magnitude",
    #         )
    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()

    #         # SNR
    #         fig, ax = plt.subplots()
    #         ax.plot(mags, np.nanmax(SNR, axis=0), c="k")
    #         ax.plot(mags, np.nanmax(phot_SNR, axis=0), c="k", ls="--")
    #         ax.scatter(REQUIRED_MAG, REQUIRED_SNR, c="r", marker="*", s=100)
    #         ax.set(
    #             xlabel="J Mag",
    #             ylabel="Theoretical SNR",
    #             ylim=(0, 17000),
    #             title="Signal to Noise Ratio vs. Magnitude",
    #         )
    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()

    #         # SNR
    #         fig, ax = plt.subplots()
    #         ax.plot(SNR[:, mags == 9], c="k", label="jmag = 9")
    #         ax.set(
    #             xlabel="Box width [pixels]",
    #             ylabel="Theoretical SNR",
    #             ylim=(0, 11000),
    #             title="Signal to Noise Ratio vs. Aperture Size",
    #         )
    #         ax.axhline(REQUIRED_SNR, c="r", ls="--")
    #         ax.legend()
    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()
    #     return (SNR[:, mags == REQUIRED_MAG] > REQUIRED_SNR).any()
