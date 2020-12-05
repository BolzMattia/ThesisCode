from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import pandas as pd
import numpy as np

# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
# check if the package rmgarch in installed or not, in case it installs, otherwise it just imports it
try:
    rmgarch = importr('rmgarch')
except:
    utils.install_packages('rmgarch')
    print('installing rmgarch package')
    rmgarch = importr('rmgarch')


# the function used to run the DCC inference.
def rmgarch_inference(file_name=None,
                      df=None,
                      ar_order=0,
                      ma_order=0,
                      p_order=1,
                      q_order=1,
                      dcc_a_order=1,
                      dcc_b_order=1,
                      model_GARCH="eGARCH",
                      pdf_GARCH="norm",
                      pdf_DCC="mvnorm",
                      forecast_dt=1,
                      out_sample=1,
                      file_script=r'./rmgarch_code_4py_col.R'):
    '''This function uses the script file-->file_script to run a DCC inference.
    data comes from the .csv saved in-->file_name or the pandas dataframe-->df.
    The others parameters represent the specific choice for the DCC implementation.'''

    if file_name is None:
        if df is None:
            Exception('A DataFrame or a .csv path must be supplied.')
        else:
            file_name = r'./__df__rmgarch_inference.csv'
            if type(df) != pd.DataFrame:
                df = pd.DataFrame(df)
            df.to_csv(file_name, index=0)
            instrumental_csv = True
    with open(file_script, 'r') as file:
        script = file.read().replace('__fileName__', file_name)
        script = script.replace('__ar_order__', str(ar_order))
        script = script.replace('__ma_order__', str(ma_order))
        script = script.replace('__g_order__', str(p_order))
        script = script.replace('__arch_order__', str(q_order))
        script = script.replace('__dcc_a__', str(dcc_a_order))
        script = script.replace('__dcc_b__', str(dcc_b_order))
        script = script.replace('__modelGARCH__', str(model_GARCH))
        script = script.replace('__pdfGARCH__', str(pdf_GARCH))
        script = script.replace('__pdfDCC__', str(pdf_DCC))
        script = script.replace('__forecast_dt__', str(forecast_dt))
        script = script.replace('__out_sample__', str(out_sample))

    results = robjects.r(script)
    # deletes the .csv file if created here
    if instrumental_csv:
        import os
        os.remove(file_name)
    # here shapes the coefficient into a dictionary of np arrays
    # it suppose that n.ahead=1 in the R code.
    # Extensions are not implemented, due to known incoherence of DCC in dt>1 predictions.
    ret = dict(zip(results.names, list(results)))
    for var_name in ret.keys():
        ret[var_name] = pandas2ri.ri2py(ret[var_name])
    if (model_GARCH == "eGARCH") | (model_GARCH == "gjrGARCH"):  # in this case there will be an extra parameter
        o_order = p_order
    else:
        o_order = 0
    N = ret['Vcv'].shape[0]
    ret['Vcv'] = ret['Vcv'].transpose([2, 0, 1])
    ret['Ey'] = ret['Ey']
    ret['Corr'] = ret['Corr'].transpose([2, 0, 1])
    num_forecasts = np.max([forecast_dt, out_sample + 1])
    Vcv_forecast = np.zeros([num_forecasts, N, N])
    for fdt in range(num_forecasts):
        for n in range(N):
            Vcv_forecast[fdt, n, :] = ret['Vcv_forecast'][fdt][n * N:n * N + N]
    ret['Vcv_forecast'] = Vcv_forecast
    ret['Ey_forecast'] = ret['Ey_forecast'].transpose([2, 1, 0])[:, :, 0]
    print(ret['Ey_forecast'].shape)
    ret['a'] = ret['Coef'][-2]
    ret['b'] = ret['Coef'][-1]
    ret['Coef'] = ret['Coef'][:-(dcc_a_order + dcc_b_order)].reshape(
        [N, 1 + ar_order + ma_order + 1 + p_order + q_order + o_order])

    return ret
