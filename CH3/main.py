import CH3.bsm_functions as bsm
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

if __name__ == '__main__':

    V0 = 17.6639
    r = 0.01

    h5 = pd.HDFStore("/home/vlemoine/data/python_for_finance/data/vstoxx_data_31032014.h5")
    futures_data = h5['futures_data']
    options_data = h5['options_data']
    h5.close()

    options_data['IMP_VOL'] = 0.0
    tol = 0.5

    futures_data['DATE'] = futures_data['DATE'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
    futures_data['MATURITY'] = futures_data['MATURITY'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))

    options_data['DATE'] = options_data['DATE'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))
    options_data['MATURITY'] = options_data['MATURITY'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e9))

    options_data['IMP_VOL'] = 0.0
    tol = 0.5  # tolerance level for moneyness
    for option in options_data.index:
        # iterating over all option quotes
        forward = futures_data[futures_data['MATURITY'] == \
                               options_data.loc[option]['MATURITY']]['PRICE'].values[0]
        # picking the right futures value
        if (forward * (1 - tol) < options_data.loc[option]['STRIKE']
                < forward * (1 + tol)):
            # only for options with moneyness within tolerance
            imp_vol = bsm.bsm_call_imp_vol(
                V0,  # VSTOXX value
                options_data.loc[option]['STRIKE'],
                options_data.loc[option]['TTM'],
                r,  # short rate
                options_data.loc[option]['PRICE'],
                sigma_est=2.,  # estimate for implied volatility
                it=100)
            options_data.ix[option, 'IMP_VOL'] = imp_vol

    plot_data = options_data[options_data['IMP_VOL'] > 0]

    maturities = sorted(set(options_data['MATURITY']))

    plt.figure(figsize=(8, 6))
    for maturity in maturities:
        data = plot_data[options_data.MATURITY == maturity]
        # select data for this maturity
        plt.plot(data['STRIKE'], data['IMP_VOL'],
                 label=maturity.date(), lw=1.5)
        plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.', label='')
    plt.grid(True)
    plt.xlabel('strike')
    plt.ylabel('implied volatility of volatility')
    plt.legend()
    plt.show()
    # tag: vs_imp_vol
    # title: Implied volatilities (of volatility) for European call options on the VSTOXX on 31. March 2014