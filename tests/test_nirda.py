from pandorareq.nirdatester import NIRDATester
from pandorareq import TESTDIR


def test_nirda_psf():
    nt = NIRDATester()
    nt.make_report(
        pdf_filename=TESTDIR + "output/NIRDA_requirements_test.pdf",
    )
