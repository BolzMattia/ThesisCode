
require("rmgarch")
#input parameters, will be replaced with the values by the Python code calling it.
df=read.csv("__fileName__")
N=NCOL(df)
armaOrders=c(__ar_order__,__ma_order__)
garchOrders = c(__g_order__,__arch_order__)
dccOrders=c(__dcc_a__,__dcc_b__)
model_type="__modelGARCH__"
distributionModel="__pdfGARCH__"
dcc_distribution="__pdfDCC__"
forecast_dt=__forecast_dt__
out_sample=__out_sample__
#creates the model specifications
xspec = ugarchspec(mean.model = list(armaOrder = armaOrders), variance.model = list(garchOrder = garchOrders, model = model_type), distribution.model = distributionModel)
uspec = multispec(c(replicate(N, xspec)))
spec1 = dccspec(uspec = uspec, dccOrder = dccOrders, distribution = dcc_distribution)
#fits it to the Dataframe df
fit1 = dccfit(spec1, data = df,out.sample=out_sample)
#forecasts values
forecast1=dccforecast(fit1,n.ahead = 1,n.roll = out_sample)
#write.csv(coef(fit1),"__coefs__")
#return results
list(Vcv=rcov(fit1),Ey=fitted(fit1),Corr=rcor(fit1),Coef=coef(fit1),Vcv_forecast=rcov(forecast1),Ey_forecast=fitted(forecast1))
