from test import *

if __name__ == "__main__":
    test_dmp_vs_prodmp_identical()
    dmp_quantitative_test(plot=True)
    promp_quantitative_test(plot=True)
    prodmp_quantitative_test(plot=True)