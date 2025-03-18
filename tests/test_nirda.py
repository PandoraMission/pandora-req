import pandorareq as pr
from pandorareq import TESTDIR


def test_nirda_psf():
    nt = pr.NIRDATester()
    nt.make_report(
        pdf_filename=TESTDIR + "output/NIRDA_requirements_test.pdf",
    )
