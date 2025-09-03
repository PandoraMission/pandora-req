from pandorareq.visdatester import VISDATester
from pandorareq import TESTDIR


def test_visda_psf():
    vt = VISDATester()
    vt.make_report(
        pdf_filename=TESTDIR + "output/VISDA_requirements_test.pdf",
    )
